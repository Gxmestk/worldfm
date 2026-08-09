# 03 — Methodology Reference (from the Full Technical Report)

> Authoritative methodology, distilled from `paper/2603.11911v3.pdf`
> (*InSpatio-WorldFM: An Open-Source Real-Time Generative Frame Model for Spatial
> Intelligence*, 17 pp, arXiv v3, 6 May 2026). Single paper (no intro/full split).
> This is the *how*: formulation, the progressive three-stage training, data, and
> evaluation. Page refs `[pN]` are to that report; section refs use the paper's own
> `[§2.x]`. Cross-refs to code use `path:line` (verified in this repo).
> Machine-readable card → [`worldfm.card.yaml`](worldfm.card.yaml). Verified 2026-07-29.

## Model at a glance

- **Backbone:** **PixArt-Σ** Diffusion Transformer (DiT) `[§2.3, p4]`, selected (not
  newly pretrained) for its quality/efficiency balance. Code ships the
  **PixArt-Sigma `XL_2`** config (`worldfm/diffusion/model/nets/PixArtWorldFM.py:482-483`).
  Parameter count **not stated** in the paper.
- **Task:** **conditional novel-view synthesis (NVS)** — from **one** reference image
  `x_ref` (with pose `π_ref`) and a **target** pose `π_tgt`, generate a geometrically
  consistent target-view frame `[§2.2, p3]`.
- **Paradigm:** **frame-based** latent diffusion — **each frame generated independently**,
  rejecting the window/sequential video-world-model paradigm `[§1, p2; §2.4.1, p4]`.
  Multi-view consistency is recovered **purely through conditioning**, not temporal
  recurrence.
- **Output:** **512×512** RGB frames, one per target pose `[§3, p12]`.
- **Speed:** **~25 FPS** at 512×512 on a single **H-series GPU**; **10 FPS** on an
  **RTX 4090** `[§3, p12]`, via **2-step DMD** distillation + engineering (compile, latent caching).
- **Release:** **inference-only** (Apache-2.0 library). All training (Stages I–III), data
  curation, and training strategies are **paper-only**; the internal panorama model is
  withheld (`README.md:52-53`).

## 2.2 Formulation — *the* mechanism to understand

InSpatio-WorldFM is a **conditional generative frame model** that denoises in the **VAE
latent space** `[§2.2, p3–4]`. With `E`/`D` the AutoencoderKL encoder/decoder, the model
learns a **noise-predicting** (ε-prediction) conditional denoiser that reverses the forward
process on the target latent `z_tgt = E(x_tgt)`. The training objective `[§2.2, p3]`:

```
L = E_{z_tgt, eps~N(0,I), t} [ || eps − eps_θ(z_t, t, C) ||² ]     # (1)

z_t = α_t · z_tgt + σ_t · eps                                       # (2)  noised latent
C   = { x_ref, π_ref, π_tgt, x̂_tgt }                                # (3)  full condition set
```

Here `x̂_tgt` is the **point-cloud rendering at the target viewpoint** from 3D foundation
models `[34,36,35,21]`, serving as an **explicit 3D spatial anchor** `[p4]`. Two things to
notice: (i) there is **no temporal index** — each target is a single frame; (ii) consistency
is entirely in the condition set `C`. The loss is the standard latent-ε-prediction L2
`[§2.2]`; the codebase carries the inherited `training_losses`/`training_losses_diffusers`
methods (`worldfm/diffusion/model/gaussian_diffusion.py:744,857`) as **library boilerplate
that is not wired to any released training loop** (📝 paper-only as a training mechanism).

### Conditioning — hybrid spatial memory, width-concatenated into self-attention `[§2.4.1, p4–6; Fig.3, p5]`

> ⭐ Key architectural point: the backbone is a **self-attention-only DiT**. The condition
> set is **not** a cross-attention bank. The transformer input is a **width-concatenation
> of three latent maps** — `[z_t | cond1 | cond2]` — that share one patch-embed + sinusoidal
> pos-embed, run full self-attention, then **only the target slice of the output is kept**.
> Cross-attention is disabled. This is the single most important design choice and it is the
> active shipped path.

```
input  = concat_along_WIDTH( z_t (target) , cond1 (explicit anchor) , cond2 (reference) )
output = TransformerDiT(input) ;  retain only the target-width slice  →  eps_θ
```

| Stream | Role | Detail |
|---|---|---|
| **`z_t` (target)** | what is denoised | noised target-view latent at timestep `t`; only this slice is retained after the transformer. Code: width-concat → `worldfm/diffusion/model/nets/PixArtWorldFMMS.py:449`; target-only slice → `PixArtWorldFMMS.py:707-713`. |
| **`cond1` — explicit 3D anchor** | coarse global geometry | **point-cloud rendering `x̂_tgt`** = global 3D cloud projected/splatted to the target camera. Built offline from panorama depth. Code: pure-PyTorch splatting `modules/point_renderer.py:89-191`; depth→PLY `modules/pano_postprocess.py:123-149`; wired `run_pipeline.py:415-428,451`. |
| **`cond2` — implicit memory** | fine appearance + plausible hallucination in unobserved regions | the **nearest reference frame** (selected by depth-consistency + view-angle + distance weighting at inference). Code: `modules/depth_selector.py:169-319` (`select_best_condition_index`), `:184-185` (`dist_min_m=1.0`, `dist_max_m=5.0` defaults); cond2 indexing+cache `modules/worldfm_infer.py:414-438`. |

Self-attention was chosen over cross-attention empirically for higher quality `[§2.4.1, p4]`.
A `cond2`-as-cross-attention KV branch **exists** in source
(`PixArtWorldFMMS.py:382-428`) but is **disabled** at inference
(`use_cond2_cross_attn=False`, `modules/worldfm_infer.py:452`).

### Camera pose encoding — PRoPE (adopted) vs Plücker vs pure-parametric `[§2.4.1, p5–6]`

The paper explores three strategies for injecting `π_ref`/`π_tgt` and **adopts PRoPE**
(Projected Relative Positional Encoding, "Cameras-as-RPE" `[20]`):

```
PRoPE:  Q ← P_i^T · Q ,   K ← P_i^{-1} · K ,   V ← P_i^{-1} · V   (+ 2D RoPE)    # (4)
```

applied per view, so attention **natively reasons about cross-view geometry** `[p5–6]`. It
beats **Plücker-ray** `[15,1,2]` (additive 6-D `(o×d, d)` features, no attention modulation)
and **pure-parametric** `[3]` (MLP on raw R/T) — chosen for **fastest convergence and most
stable control** `[p5–6]`.

> ⚠️ **Paper↔code note:** PRoPE and Plücker are **implemented but NOT activated** on the
> shipped inference path. PRoPE: `worldfm/diffusion/model/nets/prope.py:52-140`, applied in
> `PixArtWorldFM_blocks.py:158-231`, gated **default-off**
> (`PixArtWorldFMMS.py:85-89,488,521-534`). Plücker: `plucker.py:8-72`, injection
> `PixArtWorldFMMS.py:639-666`, `use_plucker` default `False` (`:480`). The inference service
> `modules/worldfm_infer.py` **never passes** `use_prope`/`prope_viewmats`/`use_plucker`, so
> the released 1-/2-step checkpoints run with **plain self-attention and no explicit camera
> modulation**. It is unclear whether the released weights were trained with PRoPE and the
> modulation is merely bypassed, or camera conditioning is effectively absent in the public
> model. Pure-parametric has **no code at all** (paper-only). Full evidence →
> [`04-paper-code-crosswalk.md`](04-paper-code-crosswalk.md).

### Offline / online two-phase runtime `[§2.1, p3; Fig.2, p3]`

The condition set is built in **two phases**. **Offline:** a single image → 360° panorama
→ depth → global point cloud. **Online:** the frame model renders target views in real time,
re-rendering `x̂_tgt` for each target camera and re-selecting `cond2`.

1. **Offline panorama** (`step1_panogen`): image→360° `[§2.1, p3]`. Code uses
   **HunyuanWorld-1.0** as an **open-source substitute** (`modules/panogen.py:71-199`,
   `run_pipeline.py:280-315`); the internal panorama generator is **withheld**
   (`README.md:52-53`).
2. **Offline 3D anchor build** (`step2_moge_pipeline`): panorama split into **42**
   Fibonacci-sphere perspective views (**FOV 45°**), **MoGe-2** infers per-view depth,
   merged and unprojected to a global point cloud. Code: `modules/moge_pano.py:20-24`
   (`FOV_DEG=45.0`, `NUM_VIEWS=42`), `:57-66` (`_get_panorama_cameras`);
   `run_pipeline.py:354-379`.
3. **Online per-frame synthesis** (`step3_*` / `step4_*`): for each target pose, render
   `x̂_tgt` (cond1), pick the nearest reference (cond2), run the distilled DiT. Code:
   `run_pipeline.py:406-451` (render+select), `:534-567` (`infer_from_render_u8` /
   `..._multistep`).

### Roll-out step (per target frame) `[§2.4.1]`

(1) read target pose `π_tgt`; (2) render explicit anchor `x̂_tgt` → VAE-encode → `cond1`,
select nearest reference → `cond2`; (3) sample `z_t` from noise, run `eps_θ` for `step`
denoising steps; (4) VAE-decode to `x_tgt` at 512×512. **One denoising call per target pose;
no window/multi-frame batching** (`modules/worldfm_infer.py:386-494`).

---

## Training — three stages `[§1, p2; §2.2, p4; Fig.3]`

The paper describes an explicit **progressive three-stage pipeline** that evolves a
foundation image generator → a controllable frame model with spatial memory → a real-time
few-step generator `[§1, p2]`.

### Stage 1 — Pre-Training (PixArt-Σ backbone) `[§2.3, p4]`

**Backbone *selection*, not new pretraining.** PixArt-Σ `[8]` is chosen for high-fidelity
generation **and** computational efficiency (the latter directly bears on real-time
deployability) `[p4]`. No continued pretraining of PixArt-Σ is described; whether any
occurred is unspecified. 📝 **paper-only** — only the derived architecture is in the repo
(`worldfm/diffusion/model/nets/PixArtWorldFM.py:1-7,482-483`); no training entrypoint.

### Stage 2 — Middle-Training: controllable frame model + hybrid memory `[§2.4, p4–7]`

Transforms the image generator into a **camera-conditioned frame model with hybrid spatial
memory**, via architectural modifications (tri-condition concat, PRoPE, explicit+implicit
memory) on data built from real video + synthetic environments. Two phases:

**2a — Fundamental Frame Model on real data `[§2.4.1, p4–7]`.** Adds the tri-condition
concat + PRoPE + hybrid spatial memory; trained under ε-prediction L2 [Eq. 1]. Three
**training strategies** `[§2.4.1 Training Strategy, p6–7]` stabilize what temporal recurrence
would otherwise enforce:
- **Noise-schedule biasing:** increase the sampling probability of **high-noise timesteps**
  so the model learns coarse spatial layout before fine texture `[p6]`.
- **Progressive condition injection:** in **early training supply only the reference frame
  (implicit memory)** — this prevents the model overfitting to the much stronger explicit
  point-cloud anchor and neglecting the implicit pathway; the explicit anchor is introduced
  **gradually** `[p6–7]`.
- **Random anchor masking:** in later training, mask the explicit anchor with some
  probability to prevent over-dependence on the 3D prior `[p7]`.

**2b — Synthetic-data finetuning with Unreal Engine `[§2.4.2, p7]`.** Feedforward
reconstruction depth/pose contains errors → inter-view inconsistencies in the point-cloud
anchors. To correct this, the authors generate **UE** trajectories with **precise
ground-truth camera pose and depth** (initial pose from semantically valid regions;
stochastic or template motion; **collision avoidance** for viewpoint validity) and finetune
the fundamental frame model **for a limited number of steps** — deliberately small, so
natural-appearance priors are not lost (`"even a small amount … yields a significant
improvement"`, `[p7]`). Training pairs mirror the real pipeline (4 reference + 12 targets).

> ⚠️ **Stage 2 is paper-only.** No `train.py`, dataset/dataloader, optimizer, or training
> loop is shipped; the three training strategies and the UE finetune are described in prose
> only. Deep dive (mechanisms + hparams) → [`06-training-distillation.md`](06-training-distillation.md).

### Stage 3 — Post-Training: Distribution Matching Distillation (DMD) `[§2.5, p7]`

Distills the multi-step Stage-II teacher into a **few-step generator** with minimal loss of
spatial consistency / fidelity. DMD `[42]` trains a few-step student to match the teacher's
output distribution by minimizing an **approximate KL** between the real distribution (base
model) and the synthetic distribution (generator), using **two diffusion models** — a
**frozen** copy of the base model estimating the **real score**, and a **dynamically-updated**
model trained on generator outputs estimating the **fake score** — whose **denoising-prediction
difference is the gradient signal**; a complementary **regression loss on pre-computed
noise-image pairs** stabilizes training and preserves mode coverage. DMD extends Variational
Score Distillation (VSD) `[§2.5, p7]`. Two empirical findings:

```
2-step > 1-step : step1 T → t_mid establishes coarse structure;
                  step2  t_mid → 0 is a dedicated refinement pass.        # (5)

t_mid = 200     : on a 1000-step schedule, the best balance.               # (6)
```

Single-step denoising recovers coarse geometry but struggles with fine detail in one pass;
`t_mid` too large re-introduces that difficulty in step 2 `[§2.5, p7–8; §3, p12]`.

> ⚠️ **DMD training is paper-only; only the distilled *inference* is released.** Code:
> 1-step at `modules/worldfm_infer.py:455-463`; **2-step (preferred)** at `:464-484`
> (`mid_t=200`, `:99`; schedule coefficients `:217-222`); the multi-step DPM-Solver++
> **teacher** sampler at `:497-567` (`DPMS` factory `worldfm/diffusion/dpm_solver.py:6-36`).
> The 1000-step schedule is `IDDPM("1000", ...)` (`worldfm_infer.py:215`). Distilled
> checkpoints: `weights/worldfm_1-step.pth`, `weights/worldfm_2-step.pth`
> (`default.yaml:50-51`). Deep dive → [`06-training-distillation.md`](06-training-distillation.md);
> runtime/decode internals → [`08-runtime-and-decode.md`](08-runtime-and-decode.md).

---

## 2 — Training data `[§2.4.1, p6; §2.4.2, p7]`

The paper does **not** state total clip count, total frame count, model params, training
compute, #GPUs, or step counts (including the "limited number of steps" of the UE finetune).
What it does specify:

| Source | Type | Role |
|---|---|---|
| internet videos + **DL3DV** `[22]` + **RealEstate10K** `[51]` | real, multi-view | fundamental frame model training `[p6]` |
| authors' own captured videos | real | fundamental frame model training `[p6]` |
| **Unreal Engine** `[12]` | synthetic, **precise GT pose + depth** | synthetic finetune (Stage 2b) `[p7]` |

- **Pose/depth recovery:** a feedforward reconstruction model (**MapAnything** `[17]`)
  estimates per-frame pose + depth on real clips `[p6]`.
- **Curation (real):** per clip randomly sample **16 frames** → **4** form the **reference
  group** (build the global point cloud) → **12** are **training targets**; each target's
  reference is the **temporally closest** of the 4; the point-cloud render = global cloud
  projected onto the target camera. **Random shuffling and masking** simulate the disorder
  and discreteness of real-world operation `[p6]`.
- **Curation (synthetic):** mirrors the **4-ref / 12-target** split with stochastic/template
  trajectories + collision avoidance `[§2.4.2, p7]`.
- **Caption:** **none.** The model is pose/image-conditioned; there is no text-caption path
  at inference (cross-attention disabled, `modules/worldfm_infer.py:97`; caption embeddings
  zeroed, `:233-240`), despite the PixArt-Σ text-to-image lineage.

Deep dive → [`07-training-data.md`](07-training-data.md).

---

## 3 — Evaluation `[§3, p12–13]`

> ⚠️ **Paper↔code note:** evaluation is **qualitative only**. The paper reports **no
> FID/PSNR/LPIPS/multi-view-consistency metrics and no benchmark tables** `[§3, p12–13]`.
> Claims of "strong multi-view consistency" and "minimal perceptual difference" after
> distillation are asserted visually (Figs. 4–8), **not numerically substantiated**.

- **Protocol:** each example = **1 reference image + 10 frames rendered from different
  camera viewpoints** `[§3, p12, Fig.4]`. Assesses the **fundamental (teacher) frame model**
  and the **distilled** variant across diverse scene types `[p12]`.
- **Baselines / comparisons (none tabulated):** **RTFM** `[37]` and **StarGen** `[45]` appear
  **only as related-work / design context** (`[§1 p2]`, `[§2.4.1 p6]`) — they are **not**
  experimental baselines; there is **no head-to-head comparison** (no table, no metric vs any
  other system). The **only** comparison is the authors' **own Stage-II teacher vs Stage-III
  distilled student**, by visual inspection `[§3, p12–13]`; the **internal ablation** of the
  three camera-pose encodings (Plücker vs PRoPE vs pure-parametric) → PRoPE chosen `[§2.4.1, p5–6]`
  (asserted "fastest convergence / most stable," no numbers).
- **Notable absence — no-GT consistency metrics are computable but not reported.** FID/KID
  (distributional realism) and the multi-view-consistency metrics (**MEt3R** CVPR'25,
  **Flow Warping Score**, **regrounding/re-projection**) require **no ground-truth target view**
  — they are computable on the model's own outputs — yet none is reported. For a system claiming
  to be a "world model," cross-view consistency is the defining property and is never measured.

**Results numbers (everything the paper actually quantifies):**

| Quantity | Value | Source |
|---|---|---|
| Inference FPS, H-series GPU | **~25 FPS** @ 512×512 | `[§3, p12]` (SKU not pinned, e.g. not stated as H100) |
| Inference FPS, RTX 4090 | **10 FPS** | `[§3, p12]` (low GPU memory footprint) |
| Baseline eval resolution | **512×512** | `[§3, p12]` |
| Distillation steps (preferred) | **2-step** | `[§2.5, p7; §3, p12]` |
| Distillation steps (compared) | 1-step | `[§2.5, p7]` |
| Intermediate timestep `t_mid` | **200** (on a 1000-step schedule) | `[§2.5, p8; §3, p12]` |
| Forward-noise schedule length | **1000** | implied `[§3, p12]`; code `IDDPM("1000")` (`worldfm_infer.py:215`) |
| Frames per eval example | 1 reference + 10 target views | `[§3, p12]` |
| Engineering levers | KV-cache management + efficient VAE latent caching | `[§3, p12]`; code `modules/worldfm_infer.py:164-211` (`torch.compile`, `reduce-overhead`), `:356-381,410-438` (cond2 latent cache) |
| Quantitative metrics (FID/PSNR/…) | **none reported** | `[§3]` — qualitative only |

**Takeaways:** real-time speed is the demonstrated win, and it is bought by **2-step DMD
with `t_mid=200`** plus heavy engineering; "2-step > 1-step" and "`t_mid=200` is best" are
the two concrete empirical findings. The FPS figures depend on **engineering levers**
(compile, caching), not on model size alone, and are **not separately quantified** against
the teacher. Multi-view consistency and post-distillation fidelity are **visual claims
only**.

### Standard NVS metrics — what a rigorous eval reports (and WorldFM omits)

WorldFM reports **none** of these. A rigorous generative-NVS evaluation (the GenVS / CAT3D /
Stable Virtual Camera standard) reports four classes of metric:

- **Distributional realism (no ground truth needed):** **FID**, **KID**.
- **Fidelity to a held-out view:** **PSNR**, **SSIM**, **LPIPS**, **DISTS** (need a real target photo).
- **Multi-view / 3D consistency (no ground truth needed):** **MEt3R** (CVPR'25, DUSt3R+DINO),
  **Flow Warping Score** (RAFT warp residual), **regrounding / re-projection** (GenVS).
- **Video:** **FVD**.

> ⚠️ **Why this is the *sharpest* critique (the asymmetry).** Fidelity metrics (PSNR/SSIM/LPIPS)
> need a real target photo — and for a *novel* view (an angle never photographed) there is none, so
> an author can fairly say "PSNR is uncomputable here." That excuse is legitimate, and it's why
> generative-NVS papers often skip fidelity metrics. **MEt3R / Flow-Warping Score / regrounding need
> no such target** — they measure the model's outputs *against each other* (cross-view agreement,
> frame-to-frame flicker, loop-closure drift) and are free to compute on any output. So a "world
> model" that claims cross-view consistency yet reports **none** of them has no excuse: omitting
> them isn't "we couldn't afford the benchmark," it's "we chose not to test the very thing we
> claim." That no-excuse asymmetry — not the loudest critique, but the hardest to defend against —
> is what makes the no-GT omission the real evaluation gap.

### Peer methods (related-work positioning — NOT experimental baselines)

None of these is run on a shared benchmark; the only actual comparison is own teacher vs student.

| Method | Ref | Paradigm | Open? | Note |
|---|---|---|---|---|
| **RTFM** (World Labs) | `[37]` | real-time **frame** model (implicit KV-cache memory) | closed | the primary foil; WorldFM = open-source + explicit 3D anchors |
| **StarGen** | `[45]` | video-diffusion + keyframe feature-warping | open (CVPR'25) | memory-design foil |
| **GEN3C** (NVIDIA) | `[26]` | 3D-cache video generation | open (CVPR'25) | closest to WorldFM's point-cloud anchor, but video-based |
| **Cosmos** (NVIDIA) | `[6]` | video world foundation model | open | cited as a precursor |
| **Matrix-Game** (Skywork) | `[48]` | action-conditioned video | open | video-based peer |
| **Long-term spatial memory** | `[38]` | video w/ spatial-memory module | closed | the "long-term spatial consistency" foil |
| **Worldplay** | `[29]` | real-time interactive video | — | a real-time competitor |

All except RTFM are **video-diffusion** (sequential frames, window latency, drift). WorldFM and
RTFM are the only **frame-based** ones; WorldFM's pitch = RTFM-style real-time, but **open-source
+ explicit 3D grounding**.

## Limitations & future work `[§4.1, p13]`

### Stated limitations (paper, §4.1 p13)

- **Dynamic content:** both the frame model and the multi-view-consistency training data
  contain **limited dynamic content** → generating dynamic scenes with high quality/stability
  is hard `[p13]`. WorldFM is fundamentally a **static-scene** world model.
- **Limited motion boundary:** historical memory relies on multi-view/panoramic observations
  whose generation models are high-compute/memory and **offline-only**, introducing a **motion
  boundary** at the online handoff — the camera is trapped inside the region the offline
  panorama reconstructed `[p13]`.
- **Interactive visual stability:** removing inter-frame temporal constraints → noticeable
  **frame jitter during interaction** `[p13]`.

### Additional limitations (verified, not paper-stated)

- **Qualitative-only evaluation** — no FID/PSNR/LPIPS/consistency metrics at all; "strong
  multi-view consistency" is asserted, not measured (see §3, and the no-GT critique above).
- **PRoPE inactive in release** — the paper's chosen camera encoding is implemented but off on
  the shipped inference path; unclear whether the released weights were trained with it.
- **Inference-only release** — all training (Stages 1–3), data, and the internal panorama model
  are withheld.
- **Non-commercial pipeline** — Apache-2.0 code/weights, but FLUX.1-Fill-dev + HunyuanWorld-1.0
  + ZIM make end-to-end use non-commercial (HunyuanWorld also geo-barred in EU/UK/Korea).
- **Practical free-fly degradation** — leaving the reconstructed region → cond1 goes blank →
  hallucinated garbage; cond2 snaps between 42 discrete views; the 16 GB-fit config (NF4 FLUX,
  reduced MoGe depth) coarsens the point cloud vs. the paper.

### Future work

The paper has **no standalone "Future Work" section** — it folds directions into the
limitations/conclusion (= "address the three stated limitations"):

- **Dynamic scenes** — expand training data + frame model to handle dynamic content with
  quality/stability (the biggest stated gap).
- **Dissolve the motion boundary** — move panorama/multi-view generation **online** (or far
  lighter) so the camera isn't confined to the offline-reconstructed bubble.
- **Temporal constraints** — reintroduce inter-frame temporal modeling to kill the jitter
  *without* sacrificing real-time speed (the core trade-off).
- **(Broader, not paper-stated)** quantitative — especially no-GT consistency — evaluation;
  permissive substitutes for the NC/geo-restricted components for any real deployment.

## Lineage (who begat which piece)

PixArt-Σ `[8]` / DiT `[24]` (backbone) · Latent Diffusion / Stable Diffusion (LDM) `[27]`
(VAE latent space) · **PRoPE / Cameras-as-RPE** `[20]` (adopted camera encoding; Plücker
`[15,1,2]` compared) · Distribution Matching Distillation (DMD) `[42]` (2-step distillation;
extends VSD) · VGGT `[34]` / DUSt3R `[36]` / MoGe `[35]` / MapAnything `[17]` / MegaDepth
`[21]` (feedforward 3D reconstruction for anchors/pose) · Stable Virtual Camera `[49]` /
Cat3D `[14]` (multi-view diffusion providers) · HunyuanWorld-1.0 `[31]` (open-source panorama
substitute) · RTFM `[37]` (World Labs, closed real-time frame model) · StarGen `[45]`
(keyframe feature-warping memory).

> _Deeper dives: formulation math → [`05-formulation.md`](05-formulation.md); training &
> distillation hparams → [`06-training-distillation.md`](06-training-distillation.md);
> training data → [`07-training-data.md`](07-training-data.md); runtime/decode →
> [`08-runtime-and-decode.md`](08-runtime-and-decode.md); full paper↔code evidence →
> [`04-paper-code-crosswalk.md`](04-paper-code-crosswalk.md); diagrams →
> [`09-diagrams.md`](09-diagrams.md)._
