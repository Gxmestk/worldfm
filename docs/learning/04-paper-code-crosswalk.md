# 04 — Paper ↔ Equation ↔ Code Crosswalk

> For each mechanism: **what it is · paper section + equation · where it lives in this
> repo · whether the released code actually contains it.** Use this as the map when
> reading any paper claim against the implementation. Grounded in `paper/2603.11911v3.pdf`
> (single paper, arXiv 2603.11911) and the released inference repo. Page refs `[pN]` are to
> that PDF; code refs are `path:line` relative to the repo root and were verified directly.
> Verified 2026-07-29.
>
> This is an **inference-only** release. Training-side items are marked ⚠️ (no `train.py`,
> dataset, optimizer, or training loop ships — see [`06-training-distillation.md`](06-training-distillation.md)).

Legend: ✅ active in shipped inference code · ⚙️ code/schema present, training loop not released · 📝 paper-only (not in repo) · ⚠️ implemented in source but **not activated** on the shipped inference path (the load-bearing WorldFM distinction).

## Architecture & formulation

| Mechanism | Paper | Code (file:line) | Status |
|---|---|---|---|
| **Independent per-frame generation** — one denoising call per target pose, no window/multi-frame batching | §2.4.1 [p4], §1 [p2] | `modules/worldfm_infer.py:386-494` (`infer_from_render_u8`, bs=1); per-frame loop `run_pipeline.py:565` | ✅ |
| **Self-attention-only tri-condition** `[z_t \| cond1 \| cond2]` width-concat (NOT a cross-attn bank); only the target slice of the output is retained | §2.4.1 [p4-5], Fig.3 [p5] | `worldfm/diffusion/model/nets/PixArtWorldFMMS.py:449` (cat `dim=3`), `:707-713` (split → target `chunk_width`); mask-stripe variants at `:438-447` | ✅ |
| **Cross-attn disabled** (caption/text path off); conditions enter via self-attn tokens | §2.4.1 [p4] (self-attn > cross-attn) | `modules/worldfm_infer.py:97` (`disable_cross_attn=True`), `run_pipeline.py:514`; caption embs zeroed `worldfm_infer.py:233-241` | ✅ |
| cond2 **cross-attention alternative branch** (cond2 → cross-attn KV tokens) | §2.4.1 [p4] (cross-attn rejected) | `PixArtWorldFMMS.py:382-428`, `:92-147`; disabled `use_cond2_cross_attn=False` at `worldfm_infer.py:452,546,554` | ⚠️ |
| **KV-compress PixArt-Sigma DiT backbone** (adaLN-Zero, qk-norm); `kv_compress_layer` empty by default → no compression active | Lineage PixArt-Sigma [8]/DiT [24], §2.3 [p4] | `PixArtWorldFM_blocks.py:86-194` (`AttentionKVCompress.downsample_2d`); XL_2 config `PixArtWorldFM.py:482-483` (depth 28 / hidden 1152 / patch 2 / heads 16) | ✅ |
| **VAE encode/decode** (AutoencoderKL, deterministic `mode()`; `vae_scale=0.13025`; channels_last, slicing/tiling off) | §2.2 [p3-4] (latent diffusion, VAE E/D) | `modules/worldfm_infer.py:178-271` (`_vae_encode_eager`/`_vae_decode_eager`), `:214` (`vae_scale`) | ✅ |

## Challenge 1 — Interaction (camera control)

| Mechanism | Paper | Code (file:line) | Status |
|---|---|---|---|
| **PRoPE camera-pose encoding** (Cameras-as-RPE, adopted) — `P_iᵀ` on Q, `P_i⁻¹` on K/V + 2-D RoPE | §2.4.1 [p5-6] | `prope.py:52-140` (`prepare_prope_apply_fns`); apply in attention `PixArtWorldFM_blocks.py:158-231`; precompute `PixArtWorldFMMS.py:521-634`; gated `kwargs.get("use_prope", False)` at `:488`, block-fwd `:85-89` | ⚠️ |
| **Plücker-ray camera embedding** (compared alternative) — additive 6-D `(o×d, d)` via MLP | §2.4.1 [p5] | `plucker.py:8-72` (`compute_plucker_rays`); injection `PixArtWorldFMMS.py:639-666`, proj MLP `:246-250`; `use_plucker` default False at `:480` | ⚠️ |
| **Pure-parametric camera injection** (3rd strategy, MLP on R/T) | §2.4.1 [p5-6] | n/a — no implementation anywhere in the repo | 📝 |
| Target-pose + intrinsics source (user-supplied) | §2.2 [p3] | `demo/meta.json` (`K` fx=fy=320, `c2w` pose list, 512×512); wired `run_pipeline.py:448-451` | ✅ |

## Challenge 2 — Consistency (hybrid spatial memory)

| Mechanism | Paper | Code (file:line) | Status |
|---|---|---|---|
| **Explicit 3D anchor** — global point-cloud splat to target view (`x̂_tgt`, cond1); pure PyTorch (no EGL/OpenGL) | §2.4.1 [p6], §2.2 [p4] | `modules/point_renderer.py:89-191` (`TorchPointCloudRenderer.render`); depth→PLY `modules/pano_postprocess.py:123-149` (`compute_ply_arrays`); wired `run_pipeline.py:415,451` | ✅ |
| **Implicit memory** — nearest reference frame (cond2) by depth-consistency + view-angle + distance weight | §2.4.1 [p6] | `modules/depth_selector.py:169-319` (`select_best_condition_index`), `:128` (`build_condition_db_in_memory`); cond2 indexing `worldfm_infer.py:414-417`; thresholds `default.yaml:42-46` (dist 1–20 m) | ✅ |
| **Offline anchor build** — panorama depth via MoGe (42 Fibonacci-sphere views, fov 45°, merged → global cloud) | §2.1 [p3], §2.4.1 [p6] | `run_pipeline.py:354-379` (MoGe infer+merge); `modules/moge_pano.py:20-24` (`NUM_VIEWS=42`, `FOV_DEG=45`), `:57-66` (`_get_panorama_cameras`); cfg `default.yaml:23-31` | ✅ |
| **Offline panorama generation** (image → 360) | §2.1 [p3], Fig.2 [p3] | `modules/panogen.py:71-199` (`Image2PanoramaDemo`, HunyuanWorld-1.0); `run_pipeline.py:280-315` | ✅ † |

† SUBSTITUTE: uses **HunyuanWorld-1.0**; the internal panorama model is withheld (`README.md:52-53`), so "consistent scene generation" reproducibility depends on the substitute.

## Challenge 3 — Stability (training-side regularization)  ⚠️ not in repo

| Mechanism | Paper | Code (file:line) | Status |
|---|---|---|---|
| Latent-diffusion **ε-prediction** loss `L = E[‖ε − ε_θ(z_t,t,C)‖²]` | §2.2 [p3, Eq.] | `worldfm/diffusion/model/gaussian_diffusion.py:744,857` (inherited `training_losses*` boilerplate, **not wired** to any released loop) | ⚙️ |
| **Noise-schedule biasing** toward high-noise `t` | §2.4.1 [p6] | n/a | 📝 |
| **Progressive condition injection** (implicit memory first, anchor later) | §2.4.1 [p6-7] | n/a | 📝 |
| **Random anchor masking** regularization | §2.4.1 [p7] | n/a | 📝 |
| **Synthetic-data finetune** (Unreal Engine precise GT pose/depth) | §2.4.2 [p7] | n/a | 📝 |
| Multi-view data curation (4-ref / 12-target split, MapAnything pose+depth) | §2.4.1 [p6] | n/a — no dataset/dataloader released; size/#clips unstated | 📝 |

## Challenge 4 — Runtime (distillation)

| Mechanism | Paper | Code (file:line) | Status |
|---|---|---|---|
| **DMD 1-step** denoising (single forward at `t=999`, one-shot x₀ pred) | §2.5 [p7], §3 [p12] | `worldfm_infer.py:455-463` (`_ts_999=999` at `:217`); ckpt `weights/worldfm_1-step.pth` | ✅ |
| **DMD 2-step** `t=999 → t_mid=200 → 0` (PREFERRED; 2-step > 1-step) | §2.5 [p7], §3 [p12] | `worldfm_infer.py:99` (`mid_t=200`), `:464-484`, coeffs `:217-222`; default `step=2` `default.yaml:50`; wired `run_pipeline.py:515-516` | ✅ |
| **Multi-step DPM-Solver++ teacher** (CFG, order 2, `time_uniform`, `multistep`) | §2.5 [p7] (teacher model) | `worldfm_infer.py:497-567` (`infer_from_render_u8_multistep`); factory `worldfm/diffusion/dpm_solver.py:6-36` (`algorithm_type="dpmsolver++"`) | ✅ |
| **Engineering**: `torch.compile` (reduce-overhead) on denoiser+VAE; cond2 + VAE latent caching | §3 [p12] (KV-cache + VAE latent caching) | `worldfm_infer.py:164-211` (compile), `:356-381` (`set_cond2_candidates_from_arrays` cache), `:410-438` (`z_c2` reuse); `run_pipeline.py:769-774` (`condition_latent_cache` event) | ✅ |

## Training stages  ⚠️ loop not released

| Stage | Paper | Code presence |
|---|---|---|
| **Stage I** Pre-Training (PixArt-Sigma backbone **selection**) | §2.3 [p4] | 📝 architecture only (`PixArtWorldFM.py:482-483`); no training entrypoint |
| **Stage II** Middle-Training (frame model + hybrid memory + UE finetune) | §2.4 [p4-7] | 📝 modules exist (tri-condition ✅, PRoPE ⚠️); train loop/dataset/optimizer not released |
| **Stage III** Post-Training DMD distillation (real + fake score, approx-KL + regression) | §2.5 [p7, Eq.] | 📝 training not released; only the **already-distilled** 1/2-step inference (`worldfm_infer.py:455-484`) |
| Checkpoint loading (1/2-step + VAE from HF) | §2.5 / §3 release | ✅ `worldfm/download.py` (`find_model`/`download_model`), `download_ckpts.py` → `huggingface.co/inspatio/worldfm` |

## Paper↔code discrepancies to verify (load-bearing)

> Full machine-readable list in [`worldfm.card.yaml`](worldfm.card.yaml) → `flags_to_verify`. The four that
> change how you read the paper:

- ⚠️ **PRoPE is the paper's ADOPTED camera encoding, but the shipped 1/2-step checkpoints run with plain self-attention** — `worldfm_infer.py` never passes `use_prope`/`prope_viewmats`/`prope_Ks`. Unclear whether the released weights were trained with PRoPE and the modulation is merely bypassed, or camera conditioning is effectively absent. Plücker (the compared alternative) is likewise implemented but never activated (`use_plucker` unset).
- ⚠️ **`cfg_scale` mismatch**: `default.yaml:55` sets `worldfm.cfg_scale=4.5`, but `run_pipeline.py:517` hardcodes `cfg_scale=0.0` for the shipped 1/2-step DMD path (which does not run an unconditional branch at all); 4.5 applies **only** to the multi-step DPM teacher (`worldfm_infer.py:497-567`).
- ⚠️ **Text/caption conditioning fully disabled** at inference despite the PixArt-Sigma text-to-image lineage (`disable_cross_attn=True`, caption embeddings zeroed). Released model is pose/image-conditioned only.
- ⚠️ **Evaluation is qualitative only** — no FID/PSNR/LPIPS/multi-view-consistency metrics or benchmark tables anywhere [§3 p12-13]; "minimal perceptual difference" after distillation is asserted, not numerically substantiated.

> Other unstated facts (do not fabricate): model parameter count, dataset size/#clips/total frames, training
> compute/#GPUs/step counts, and the exact "H-series GPU" SKU behind the ~25 FPS / 10 FPS claims [§3 p12].
> MoGe must be checked out at pinned commit `7807b5de` (`setup.sh`); submodules are empty until
> `git submodule update --init --recursive` runs.
