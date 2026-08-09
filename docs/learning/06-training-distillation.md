# 06 — Training & Distillation Deep Dive

> From paper §2.3–§2.5 (p4–7). Three stages: **Pre-Training → Middle-Training → Post-Training**.
> Page refs `[pN]` / `[§x]` are to `paper/2603.11911v3.pdf`; code refs are `path:line` in this repo.
>
> ⚠️ **Honest status — read this first.** The release is **inference-only.** There is no
> `train.py`, no dataset/dataloader, no optimizer, no `backward()` anywhere in the repo
> (`modules/`, `worldfm/`). The **entire 3-stage training pipeline is paper-only.** What the
> repo *does* ship is (a) the **inference config** `default.yaml` with exact inference-side
> hparams, and (b) the **already-distilled 1-/2-step student** plus a multi-step teacher sampler.
> So unlike the AlayaWorld reference (which released `ErrorBankConfig`/`DmdConfig` training
> schemas), here the "exact shipped hyperparameters" tables below are **inference/distillation-schedule**
> constants — not training-loop hparams (LR, batch size, optimizer, step counts are all unstated).
> Note also: `WorldFM.yaml` is a **conda environment lockfile**, not a model config.
>
> Verified 2026-07-29.

## The three-stage spine `[§1 p2; §2.2 p4]`

```
Stage I  Pre-Training    image-generation prior        (select PixArt-Σ DiT)        [§2.3]
   │
Stage II Middle-Training  controllable frame model      (+ tri-cond, PRoPE, memory)  [§2.4]
   │   ├── 2a  Fundamental Frame Model on REAL data     (ε-prediction L2)           [§2.4.1]
   │   └── 2b  Synthetic (Unreal-Engine) finetune       (precise GT pose/depth)      [§2.4.2]
   │
Stage III Post-Training   real-time few-step generator  (DMD → 2-step, t_mid=200)    [§2.5]
```

Each stage is described below with its **purpose**, **objective**, and the **exact shipped
constants** where the repo actually carries them. Training-side details the paper leaves
unspecified (LR, #GPUs, step counts, dataset size) are marked **TBD** — do not invent them.

---

## Stage 1 — Pre-Training (image-generation prior) `[§2.3 p4]`

**Purpose:** establish a high-fidelity, compute-efficient image-generation prior as the DiT
backbone. The paper frames this as backbone **selection**, not new pretraining:

> "We select PixArt-Σ [8] as our foundation model … an efficient Diffusion Transformer (DiT)
> [24] for text-to-image generation that achieves quality competitive with state-of-the-art
> models at significantly lower computational cost." `[§2.3 p4]`

**Objective:** PixArt-Σ's own text-to-image objective (inherited; not restated). The only
architectural fingerprint that survives into the released code is the **XL_2** DiT config:

| Field | Value | Source |
|---|---|---|
| `depth` | 28 | `PixArtWorldFM.py:482-483` (`PixArtWorldFM_XL_2`) |
| `hidden_size` | 1152 | `PixArtWorldFM.py:482-483` |
| `patch_size` | 2 | `PixArtWorldFM.py:482-483` |
| `num_heads` | 16 | `PixArtWorldFM.py:482-483` |
| `in_channels` | 4 (latent) | lineage PixArt-Σ |
| `learn_sigma` | True | `iddpm.py:15`; `worldfm_infer.py:215` |

📝 **Status: paper-only.** Only the derived architecture is present (`PixArtWorldFM.py:1-7`);
there is no pretraining entrypoint. Whether any continued pretraining/finetuning of PixArt-Σ
occurred on top of the public weights is **unspecified** (open question, see `worldfm.card.yaml`
`flags_to_verify`). Parameter count is **TBD** — do not fabricate a B-count.

---

## Stage 2 — Middle-Training (controllable frame model + spatial memory) `[§2.4 p4–7]`

**Purpose:** transform the single-image generator into a **camera-conditioned frame model**
that generates each target view *independently* yet stays multi-view consistent via hybrid
spatial memory. Two sub-stages: a fundamental frame model on real data (2a), then a synthetic
Unreal-Engine finetune (2b).

### 2a — Fundamental Frame Model on real data `[§2.4.1 p4–7]`

**Objective — latent ε-prediction L2** `[§2.2 p3–4]`:

```
L = E_{z_tgt, ε∼N(0,I), t} [ ‖ ε − ε_θ(z_t, t, C) ‖² ]          # training loss (§2.2)
z_t = α_t · z_tgt + σ_t · ε                                      # forward noising (§2.2)
C  = { x_ref, π_ref, π_tgt, x̂_tgt }                              # full condition set
```

`x̂_tgt` is the point-cloud rendering at the target viewpoint (cond1 / explicit 3D anchor);
`x_ref` is the reference frame (cond2 / implicit memory). Verified ε (not x0/velocity) is the
target: `iddpm.py:11,15,18` uses `noise_schedule="linear"`, `learn_sigma=True`,
`diffusion_steps=1000` with `LossType.MSE` / `ModelMeanType.EPSILON`.

**Architectural modifications added in this stage** (all verified in-code, see `docs/03`/`docs/05`):

- **Self-attention-only tri-condition injection.** Input = width-concat `[z_t | cond1 | cond2]`;
  cross-attention disabled. → `PixArtWorldFMMS.py:449` (concat), `:707-713` (split, keep target),
  `worldfm_infer.py:97` (`disable_cross_attn=True`).
- **PRoPE camera-pose encoding (adopted).** Pᵀ on queries, P⁻¹ on keys/values + 2D RoPE.
  ⚠️ Implemented (`prope.py:52-140`) but **not activated on shipped inference** —
  `worldfm_infer.py` never passes `use_prope`/`prope_viewmats`. See flags below.
- **Hybrid spatial memory.** Explicit point-cloud anchor (`point_renderer.py:89-191`) + implicit
  nearest-reference-frame memory (`depth_selector.py:169-319`).

**Training data** `[§2.4.1 p6]` — per real clip: sample **16 frames**; MapAnything `[17]`
estimates pose+depth; **4** frames build the global point cloud, **12** are training targets;
each target's reference = temporally closest of the 4; random shuffling/masking simulates
real-world disorder. 📝 **paper-only** — no dataset/dataloader released; size/#clips **TBD**.

### 2b — Synthetic (Unreal Engine) finetuning `[§2.4.2 p7]`

**Purpose:** real-data point clouds inherit depth/pose errors from feedforward reconstruction,
which leak inter-view inconsistencies into cond1. UE synthetic data provides **precise
ground-truth** camera pose + depth to correct this.

**Method:** select semantically-valid start poses; synthesize trajectories by stochastic motion
sampling *or* pre-defined motion templates; enforce collision avoidance; build training pairs
**analogously to real data** (4-reference / 12-target split).

> "We finetune the fundamental frame model on this synthetic dataset for **a limited number of
> steps**. The controlled exposure … is intentional — excessive finetuning … would compromise the
> model's ability to generate realistic appearances on natural images." `[§2.4.2 p7]`

📝 **Status: paper-only.** Step count is "limited" but **unstated**; no finetune code or UE data
released.

### Training strategy (stabilization, Stage II) `[§2.4.1 p6–7]`

Three regularizers described in the paper. **All paper-only** (no training code):

- **Noise-schedule biasing.** Increase sampling probability of **high-noise** timesteps so the
  model learns coarse spatial layout before fine texture `[§2.4.1 p6]`.
- **Progressive condition injection.** Early training supplies **only the reference frame**
  (implicit memory), forcing the model to learn implicit cues and avoid overfitting the much
  stronger explicit point-cloud anchor; the anchor is **introduced gradually** `[§2.4.1 p6–7]`.
- **Random anchor masking.** In later training the explicit anchor is **randomly masked** with
  some probability to prevent over-dependence on the 3D prior `[§2.4.1 p7]`.

> ⚠️ The inherited math library does carry `training_losses` / `training_losses_diffusers`
> methods (`gaussian_diffusion.py:744,857`) — these are OpenAI/PixArt **boilerplate**, not a
> released training loop. They are **not wired** to any optimizer, dataloader, or entrypoint.

---

## Stage 3 — Post-Training distillation (real-time few-step generator) `[§2.5 p7]`

**Purpose:** distill the multi-step Stage-II teacher into a **few-step** student for real-time /
interactive rendering on consumer GPUs, "with minimal loss in spatial consistency and visual
fidelity" `[§2.5 p7]`.

**Mechanism — Distribution Matching Distillation (DMD) `[42]`.** Maintain a **frozen real-score**
copy of the base model and a **dynamically-updated fake-score** model trained on generator
outputs; the **difference of their denoising predictions** is the gradient signal that updates
the generator (approximate-KL minimization). A **complementary regression loss** on pre-computed
noise-image pairs from the base model's deterministic sampler stabilizes training and preserves
mode coverage. Extends Variational Score Distillation (VSD). Readably:

```
∇_θ D_KL(p_θ,τ ‖ p_data,τ) ≈ − E[ ( s_real(ẑ_τ,τ) − s_fake(ẑ_τ,τ) ) · ∂ẑ_τ/∂θ ]   # DMD gradient (§2.5)
L_total = L_DMD + λ · L_regression          # regression on base-model deterministic pairs (§2.5)
```

📝 **The DMD *training procedure* is paper-only.** What the repo ships is the **already-distilled
student inference** + a multi-step teacher sampler. Two empirical findings drive the released
schedule `[§2.5 p7; §3 p12–13]`:

- **2-step > 1-step.** A second step is a dedicated refinement pass — single-step from pure noise
  recovers coarse structure but struggles with fine detail.
- **t_mid = 200 is critical** (on a 1000-step schedule). Step 1 denoises T→200 (establishes coarse
  structure); step 2 denoises 200→0 (refines from a relatively clean state). Too-large t_mid
  degenerates step 2 into hard single-step denoising.

```
step 1:  ẑ_mid ← denoise(z_T,   t=999 → t_mid=200)     # coarse structure   [§2.5; worldfm_infer.py:464-473]
step 2:  x̂_0  ← denoise(ẑ_mid, t=200  → 0)             # fine refinement   [§2.5; worldfm_infer.py:474-484]
```

### Distilled student paths actually in the code

| Path | Schedule | cfg_scale | Code |
|---|---|---|---|
| **1-step student** (when `step=1`) | one forward at t=999, x0-prediction | 0.0 | `worldfm_infer.py:455-463` |
| **2-step student** (when `step=2`, the shipped default) | t=999 → t_mid=200 → 0 | 0.0 | `worldfm_infer.py:464-484` |
| **Multi-step teacher** (`step∉{1,2}`) | DPM-Solver++, order 2, time_uniform, multistep | 4.5 | `worldfm_infer.py:497-567`, `dpm_solver.py:30` |

Checkpoints: `weights/worldfm_1-step.pth`, `weights/worldfm_2-step.pth` (`worldfm/download.py`,
`download_ckpts.py`). The 1000-step teacher schedule is the inherited IDDPM linear schedule
(`iddpm.py:11,18`; loaded at `worldfm_infer.py:215`).

> ⚠️ **Paper↔code note (teacher sampler):** the paper calls the teacher only a *"multi-step
> deterministic sampler"* `[§2.5 p7]` — it **does not name "DPM-Solver++"** and gives **no
> teacher step count**. The "DPM-Solver++, order-2, ~14 steps" framing is **inferred from the
> code** (the multi-step path's factory is `algorithm_type="dpmsolver++"`, `worldfm_infer.py:505`,
> with an *"e.g. 14 steps"* comment at `:505`) — accurate for the code's fallback path, but **not
> a paper claim**.

---

## Exact shipped hyperparameters — `default.yaml` (the only model config)

The repo ships **one** model config — `default.yaml` — and it is an **inference** config. There
are **no training-loop hparams** (LR, batch size, optimizer, total steps) anywhere. The values
that bear on the distilled schedule are transcribed verbatim below.

**`worldfm:` block — `default.yaml:48-64`** (Step-4 WorldFM inference):

| Field | Value | Meaning / note |
|---|---|---|
| `step` | **2** | distilled student steps; `1` or `2` → DMD path, else multi-step teacher (`default.yaml:50`) |
| `model_path` | `weights/worldfm_${worldfm.step}-step.pth` | selects 1- vs 2-step checkpoint (`default.yaml:51`) |
| `vae_path` | `weights/vae` | AutoencoderKL (`default.yaml:52`) |
| `image_size` | **512** | baseline eval resolution; ~25 FPS claim is at 512×512 (`default.yaml:53`) |
| `version` | `sigma` | PixArt-Σ variant → `PixArtWorldFMMS_XL_2` (`default.yaml:54`; `worldfm_infer.py:131-139`) |
| `cfg_scale` | **4.5** | ⚠️ applies **only to the multi-step teacher**; see discrepancy note below (`default.yaml:55`) |
| `compile_model` / `compile_mode` | true / `reduce-overhead` | torch.compile on denoiser (`default.yaml:56-57`; `worldfm_infer.py:164-175`) |
| `vae_channels_last` | true | VAE memory layout for speed (`default.yaml:58`) |
| `vae_deterministic` | true | encode uses `latent_dist.mode()` (`default.yaml:59`) |
| `compile_vae` / `compile_vae_mode` | true / `reduce-overhead` | torch.compile on VAE E/D (`default.yaml:60-61`; `worldfm_infer.py:191-206`) |
| `disable_vae_slicing` / `disable_vae_tiling` | true / true | slicing/tiling off for speed (`default.yaml:62-63`) |

**Distillation-schedule constants — `worldfm_infer.py`** (these are the closest thing to a
`DmdConfig` in this release):

| Field | Value | Code |
|---|---|---|
| `mid_t` | **200** | `worldfm_infer.py:99` (dataclass), `run_pipeline.py:517` |
| `_ts_999` | 999 | `worldfm_infer.py:217` |
| `_ts_mid` | `int(mid_t)` = 200 | `worldfm_infer.py:218` |
| `_a_999 / _s_999` | √α₉₉₉ / √(1−α₉₉₉) | `worldfm_infer.py:219-220` |
| `_a_mid / _s_mid` | √α₂₀₀ / √(1−α₂₀₀) | `worldfm_infer.py:221-222` |
| underlying schedule | IDDPM linear, **1000** steps, `learn_sigma=True` | `worldfm_infer.py:215`; `iddpm.py:11,15,18` |
| teacher sampler | DPM-Solver++, order 2, `time_uniform`, `multistep` | `worldfm_infer.py:564-565`; `dpm_solver.py:30` |
| VAE `scaling_factor` | 0.13025 | `worldfm_infer.py:214` |

**Engineering optimizations enabling the FPS claim** `[§3 p12]` — torch.compile on denoiser+VAE,
condition/cond2 VAE-latent caching, KV-cache management:
`worldfm_infer.py:164-211`, `:356-381`, `:410-438`; `run_pipeline.py:769-774`.

> ⚠️ **`WorldFM.yaml` is NOT a model config.** It is a conda environment lockfile
> (`name: WorldFM`, pinned `pytorch=2.5.0=cuda12.4`, `WorldFM.yaml:1,86-87`). It carries no
> training or model hyperparameters; ignore it for hparam transcription.

---

> ⚠️ **Note on `cfg_scale` (paper↔code discrepancy):** `default.yaml:55` sets
> `worldfm.cfg_scale: 4.5`, but that value flows **only** to the multi-step teacher path
> (`run_pipeline.py:549,568` → `infer_from_render_u8_multistep`). The shipped 1-/2-step DMD path
> **hardcodes `cfg_scale=0.0`** (`WorldFMInprocessConfig` at `worldfm_infer.py:100`, applied at
> `run_pipeline.py:517`). So the released default (`step=2`) runs at **cfg_scale=0.0**, not 4.5.
>
> ⚠️ **Note on PRoPE:** PRoPE is the paper's *adopted* camera encoding `[§2.4.1 p5–6]` and is
> implemented (`prope.py:52-140`), yet the shipped inference never passes camera matrices, so the
> released 1-/2-step checkpoints run with **plain self-attention and no explicit camera-pose
> modulation**. Whether the released weights were *trained* with PRoPE (and the modulation is merely
> bypassed at inference) is **unclear** — flagged in `worldfm.card.yaml` `flags_to_verify`.
>
> **Note on training reproducibility:** every training-side quantity — LR, optimizer, batch size,
> total/Stage-II/UE-finetune step counts, dataset size, #clips, #GPUs, compute — is **unstated**,
> and no training code is released. The 3-stage pipeline as described here reproduces the *paper*,
> not the repo. Cross-reference the full mechanism↔status table in
> [`04-paper-code-crosswalk.md`](04-paper-code-crosswalk.md); data detail lives in
> [`07-training-data.md`](07-training-data.md); the distilled decode loop in
> [`08-runtime-and-decode.md`](08-runtime-and-decode.md).

---

## Released checkpoints — licensing & load mechanics

**Files in `weights/` (published at `inspatio/worldfm` on Hugging Face):**

| File | Size | What |
|---|---|---|
| `worldfm_2-step.pth` | ~2.46 GB (2,458,577,701 B) | 2-step DMD student (the default) |
| `worldfm_1-step.pth` | ~2.46 GB (2,458,577,701 B) | 1-step DMD student (fast) |
| `vae/diffusion_pytorch_model.safetensors` | ~320 MB | `AutoencoderKL` VAE |

- Both `.pth` files share an **identical byte count but different MD5s** — genuinely distinct
  weights; both instantiate the **same `PixArtWorldFMMS_XL_2`** architecture (selection keys off
  `version`+`image_size`, never `step`). Parameter count is **unstated** (~0.6 B estimated from
  the 2.46 GB fp32 file).
- **Load mechanics** (`modules/worldfm_infer.py:149-176`): `find_model` → `torch.load(map_location=CPU)`
  (fp32-on-CPU transient ~2.46 GB); **`del state_dict["pos_embed"]` + `load_state_dict(strict=False)`**
  because the runtime re-derives positional embeddings for the actual latent grid via
  `warm_pos_embed_cache(width_multiplier=3)` (the 3× for tri-condition); then `.eval().to(fp16)`.
  The compiled entry point is `forward_with_dpmsolver`, whose output is `chunk(2, dim=1)[0]`
  (keeps the 4-channel ε, drops the variance half from `pred_sigma`).
- **Licensing:** the WorldFM library + the two students + VAE are **Apache-2.0** — but a full
  pipeline is **non-commercial**: FLUX.1-Fill-dev / NF4 (FLUX Dev Non-Commercial), HunyuanWorld-1.0
  (Tencent, non-commercial + geo-restricted EU/UK/Korea), ZIM (CC BY-NC). MoGe (MIT) and
  Real-ESRGAN (BSD-3) are permissive.
- **Withheld:** the teacher (Stage-II fundamental model), the DMD fake-score/critic, all
  Stage-I/II training weights, optimizer states, and all training code/data.
