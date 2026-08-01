# 05 — Formulation Deep Dive

> From `paper/2603.11911v3.pdf` §2.2 (Formulation, p3–4), §2.4.1 (Fundamental Frame Model,
> p4–6), and §2.5 (Post-Training, p7). The precise mechanism: how one target view is
> generated, what enters the DiT, how the hybrid spatial memory is formed, and how the model
> is distilled to a 2-step real-time sampler. Equations rendered readably and cross-checked
> against the rendered PDF (pypdf garbles math); every `code:` ref is verified in this repo.
> Sibling deep-dives: [`06-training-distillation.md`](06-training-distillation.md) (Stages I–III
> + DMD objective), [`08-runtime-and-decode.md`](08-runtime-and-decode.md) (sampler internals),
> [`04-paper-code-crosswalk.md`](04-paper-code-crosswalk.md) (full status table). Verified 2026-07-29.

## 1. The conditional frame-model objective `[§2.2, Eq. (training loss)]`

InSpatio-WorldFM is a **latent diffusion** model that synthesizes **one target view at a time**
from a single reference image `x_ref` under a user-defined target camera pose `π_tgt` [§2.2 p3].
There is no temporal window and no inter-frame recurrence — each target is an independent
sample. Let `E`/`D` be a pretrained `AutoencoderKL` VAE and `z = E(x)` the latent. The model is
a conditional denoiser `ε_θ` trained to reverse the forward process in latent space by predicting
the noise added to the target latent `z_tgt`:

```
L = E_{z_tgt, ε ~ N(0,I), t} [ ‖ ε − ε_θ(z_t, t, C) ‖² ]            # (1)
```

```
z_t = α_t · z_tgt + σ_t · ε,    ε ~ N(0,I)                          # (2)
```

`(α_t, σ_t)` is the predefined noise schedule. The training target is verified to be **epsilon**
(`learn_sigma=True, pred_sigma=True`, 1000-step linear `IDDPM` schedule), not `x0` or velocity.

```
C = { x_ref, π_ref, π_tgt, x̂_tgt }                                  # (3)
```

The condition set `C` carries the reference image + its pose, the target pose, and `x̂_tgt` — the
**point-cloud rendering at the target viewpoint** from a 3D foundation model, which serves as an
explicit spatial anchor [§2.2 p4]. Two of these conditions (`x_ref`, `x̂_tgt`) enter as **image
latents**; the two poses (`π_ref`, `π_tgt`) enter through camera-pose encoding (§3).

> 📝 **Paper-only as a *training* mechanism.** Eq. (1) is the Stage-II objective, but **no
> training loop, optimizer, or dataset is released**. The inherited `training_losses` helpers at
> `worldfm/diffusion/model/gaussian_diffusion.py:744,857` are library boilerplate and are **not
> wired to any shipped training entrypoint**. Only the *inference* of the already-distilled model
> is in code — see §5.

## 2. The tri-condition width-concat (self-attention only) `[§2.4.1]`  ⭐

> The single most important architectural fact. The backbone is **one self-attention DiT**.
> Conditions are **not** a cross-attention bank. For each target view the transformer input is
> built by **spatially concatenating three latent maps along the width** — the noised target, the
> explicit 3D anchor, and the reference frame — running **full self-attention**, then **slicing
> off everything but the target** after the final layer.

```
X_in = concat_width(  z_t (target) | cond1 = ẑ_tgt (anchor) | cond2 = ẑ_ref (reference) )   # (4)
```

All three components share one patch-embedding layer and sinusoidal positional embeddings
[§2.4.1 p4–5]; after the transformer, the output is split along the width and **only the target
slice is retained** as the prediction. Empirically the paper reports self-attention injection beats
cross-attention alternatives on quality [§2.4.1 p4], so cross-attention is disabled on the shipped
path.

| Stream | Source | Role | Code |
|---|---|---|---|
| `z_t` target | noised target latent (`z ~ N(0,I)` at inference) | what gets denoised → `x_tgt` | `worldfm_infer.py:441` |
| `cond1` anchor | VAE latent of point-cloud render `x̂_tgt` at `π_tgt` | explicit coarse 3D geometry prior | `worldfm_infer.py:425` (`z_c1`), `:451` |
| `cond2` reference | VAE latent of nearest reference frame `x_ref` | implicit appearance memory | `worldfm_infer.py:410-416` (`z_c2`), `:451` |

→ Concat: `worldfm/diffusion/model/nets/PixArtWorldFMMS.py:449` (`x = torch.cat([x, cond1, cond2], dim=3)`).
→ Target-only retention: `PixArtWorldFMMS.py:707-713` (split, keep `x[..., :chunk_width]`).
→ Self-attn enforced: `modules/worldfm_infer.py:97` (`disable_cross_attn=True`), `:452` (`use_cond2_cross_attn=False`).

> ⚠️ **Paper↔code note (text path).** The backbone is PixArt-Σ, a *text-to-image* DiT [§2.3 p4],
> but the caption/text cross-attention is fully disabled at inference — `disable_cross_attn=True`
> and `caption_embs = torch.zeros(...)` at `worldfm_infer.py:233,534`. The released model is
> **pose/image-conditioned only**; there is no text-prompt path. A `cond2` cross-attention branch
> (`PixArtWorldFMMS.py:382-428`) exists in source but is dead on the shipped path — verify before
> touching.

### Why "concat + self-attention, not cross-attention" matters

Because target, anchor, and reference attend **to each other** within one self-attention, the model
can cross-reference coarse geometry against reference appearance freely while denoising — stronger
than independent cross-attention KV banks. The cost is a wider joint sequence, but each target is
still **one independent forward pass**: no window batching, no temporal recurrence. This is the
concrete machinery behind the "frame-based, real-time" claim, and it is *why* multi-view consistency
must be recovered purely through conditioning (the trade-off explored in [`01-big-picture.md`](
01-big-picture.md)).

## 3. Camera-pose encoding: PRoPE `[§2.4.1]`

Encoding camera geometry into the transformer is what turns an image generator into a
*controllable* frame model. The paper explores three strategies for injecting pose [§2.4.1 p5–6]:

- **Plücker-ray embedding** [15,1,2]. Per patch, 6-D Plücker coordinates in world coords, projected
  by a 2-layer MLP and **added** to the patch embeddings — additive, does not modulate attention.
- **PRoPE (Projected Relative Positional Encoding)** [20] — **adopted**. Camera projection matrices
  `P_i` are applied directly to the attention tensors, plus 2D rotary embeddings for intra-image
  structure, so attention reasons natively about cross-view geometry.
- **Pure-parametric injection** [3] — raw `R`/`T` through a learned MLP, added to hidden states.

```
PRoPE:   Q' = P_i^⊤ Q,   K' = P_i^{-1} K,   V' = P_i^{-1} V   (+ 2D RoPE)        # (5)
```
```
Plücker:  p = (o × d, d) ∈ R^6  --MLP_2-->  add to patch embeddings               # (6)
```

PRoPE is chosen for "fastest convergence and the most stable camera control" [§2.4.1 p6]; the other
two are the ablation baselines.

> ⚠️ **Paper↔code note (camera modulation is INACTIVE on shipped inference).** All three strategies
> except pure-parametric are implemented — PRoPE at `worldfm/diffusion/model/nets/prope.py:52-140`
> (applied in `PixArtWorldFM_blocks.py:158-231`, precomputed `PixArtWorldFMMS.py:521-634`, gated
> `use_prope` default-off at `PixArtWorldFMMS.py:85-89,488`); Plücker at
> `worldfm/diffusion/model/nets/plucker.py:8-72` (injected `PixArtWorldFMMS.py:639-666`,
> `use_plucker` default-off `:480`). But **`modules/worldfm_infer.py` never passes `use_prope` /
> `prope_viewmats` / `use_plucker`**, so the released 1-/2-step checkpoints run with **plain
> self-attention and no explicit camera-pose modulation**. Open: were the released weights trained
> with PRoPE and the modulation merely bypassed at inference, or is camera conditioning effectively
> absent in the public model? Verify. (`pure-parametric` has **no code at all** — paper-only.)

## 4. Hybrid spatial memory: explicit anchor + implicit memory `[§2.4.1]`

The concat of §2 is not arbitrary — it instantiates a **hybrid spatial memory** [§2.4.1 p6,
Fig.3 p5]. RTFM [37] uses posed frames as primitive memory; StarGen [45] warps keyframe features.
WorldFM combines two complementary channels:

- **Explicit anchor (`cond1` = `x̂_tgt`).** A global 3D point cloud (built offline from panorama
  depth) is projected/splatted to the target camera to give **coarse global geometry** for that
  view. It anchors the generation in 3D but is bounded by feedforward-reconstruction error in the
  point cloud.
- **Implicit memory (`cond2` = `x_ref`).** The reference frame is selected as the **nearest
  condition view** (depth-consistency + view-angle + distance weighting) and injected through
  self-attention, supplying **fine appearance** and letting the model plausibly **hallucinate
  unobserved regions**.

Explicit holds coarse structure; implicit preserves fine detail. Both reach the target **only
through self-attention** (§2), never through a separate bank — which is exactly what Eq. (4)'s
width-concat realizes.

| Memory | Built | Rendered/selected for target `π_tgt` | Code |
|---|---|---|---|
| explicit `cond1` | offline: panorama → MoGe depth → global point cloud | splat global cloud into target camera → `x̂_tgt` | `modules/point_renderer.py:89` (`render`); build `modules/pano_postprocess.py:123-149`; wire `run_pipeline.py:415-428` |
| implicit `cond2` | offline: 42 condition views w/ depth + pose | pick nearest view by depth/angle/distance | `modules/depth_selector.py:169-319` (`select_best_condition_index`), `:128` (`build_condition_db_in_memory`) |

## 5. Real-time sampling objective: 2-step DMD `[§2.5]`

Stage III distills the multi-step teacher into a few-step generator via **Distribution Matching
Distillation** [42] (the *training* procedure is paper-only — see [`06-training-distillation.md`](
06-training-distillation.md)). What is **shipped** is the already-distilled **2-step sampler**,
which is the default (`step=2`, `mid_t=200`):

```
Step 1  (T → t_mid):   ε₁ = ε_θ(z, t=999, C),   z ~ N(0,I)
                       x̂₀ = α₉₉₉ · z − σ₉₉₉ · ε₁
                       z_mid = α_mid · x̂₀ + σ_mid · ε',   ε' ~ N(0,I)        # (7)
Step 2  (t_mid → 0):   ε₂ = ε_θ(z_mid, t_mid, C)
                       x̂ = α_mid · z_mid − σ_mid · ε₂                     # (8)
```

The first step predicts `x̂₀` from pure noise and **re-noises to `t_mid`**, establishing coarse
spatial structure; the second step refines from that relatively clean state. `t_mid = 200` (on the
1000-step schedule) is reported as the best balance — large enough that step 1 does the bulk of the
denoising, small enough that step 2 is an effective refinement pass [§2.5 p7; §3 p12]. The paper
finds **2-step > 1-step**: a single forward from pure noise recovers coarse geometry but struggles
with fine detail, so the second step is a dedicated refinement phase.

> Code: 1-step branch `worldfm_infer.py:455-463`; 2-step branch `:464-484`; schedule coefficients
> `_a_999/_s_999/_a_mid/_s_mid` precomputed from the 1000-step `IDDPM` alphas at `:215-222`
> (`mid_t=200` at `:99`, `_ts_999` at `:217`). A multi-step DPM-Solver++ teacher path
> (`worldfm_infer.py:497-567`, `worldfm/diffusion/dpm_solver.py:6-36`) runs only when `step ∉ {1,2}`
> — see [`08-runtime-and-decode.md`](08-runtime-and-decode.md).
>
> ⚠️ **Paper↔code note (cfg).** `default.yaml` sets `cfg_scale: 4.5`, but the shipped 1-/2-step DMD
> path uses `cfg_scale: 0.0` (`worldfm_infer.py:100`); 4.5 applies **only** to the multi-step
> teacher. The released default (`step=2`) therefore runs **without classifier-free guidance**.

## 6. Where to read this in code (start here)

1. `modules/worldfm_infer.py:386-494` — `infer_from_render_u8`, **one denoising call per target
   pose**: encode `cond1`/`cond2`, build `z ~ N(0,I)`, run the 1-/2-step sampler, VAE-decode.
2. `worldfm/diffusion/model/nets/PixArtWorldFMMS.py:449` — the tri-condition width-concat (Eq. 4);
   `:707-713` — target-only output retention.
3. `modules/worldfm_infer.py:97,452` — self-attention enforced (`disable_cross_attn=True`,
   `use_cond2_cross_attn=False`); `:233,534` — caption embeddings zeroed (text path off).
4. `modules/point_renderer.py:89` — explicit-anchor point-cloud splat to the target camera.
5. `modules/depth_selector.py:169-319` — implicit-memory nearest-view selection (`cond2`).
6. `worldfm/diffusion/model/nets/prope.py:52-140` and `plucker.py:8-72` — PRoPE / Plücker
   encoders (implemented but **inactive** on shipped inference — see §3 ⚠️).
7. `modules/worldfm_infer.py:215-222,455-484` — 1000-step schedule coefficients and the 2-step
   DMD sampler (Eqs. 7–8).
