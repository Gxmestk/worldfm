# 01 — Big Picture: InSpatio-WorldFM

> Grounded in `paper/2603.11911v3.pdf` (17 pp, "InSpatio-WorldFM: An Open-Source
> Real-Time Generative Frame Model for Spatial Intelligence", InSpatio Team, 6 May
> 2026) + this repo's code. There is **one paper** (no intro/full split). Page refs
> `[pN]` and section refs `[§x]` are to this paper; code refs are `path:line` in
> this repo (an **inference-only** release — training code is not shipped).
> Verified 2026-07-29.

## TL;DR

InSpatio-WorldFM is an **open-source, real-time, frame-based world model for novel-view
synthesis (NVS)** — a **PixArt-Σ DiT** that you drive with **one reference image and a
target camera pose**, and which emits **one spatially consistent target view per call**
through a **2-step distilled denoiser**. Its contribution is *not* a single new module
but a **specific posture**: it **rejects the video/window world-model paradigm
entirely**, generating each frame **independently** with no inter-frame temporal
recurrence, and recovers global 3D consistency **purely through conditioning** — an
**explicit point-cloud anchor + an implicit reference-frame memory**, both reaching the
denoiser only via **self-attention**. It is the open-source counter to World Labs' closed
**RTFM** and to video-based world models, organized around six framing axes for each of
which it states *the taxonomy, its pick, the rationale, and the trade-off*.

- **Backbone:** PixArt-Σ latent DiT (XL_2 config). **Output:** 512×512 RGB, **one frame
  per target pose**; **2-step DMD denoising** (`t=999 → t_mid=200 → 0`) by default.
- **Two conditioning channels, both self-attention:** (1) **cond1** — a point-cloud
  render of the target view (**explicit 3D anchor**); (2) **cond2** — the nearest
  reference frame (**implicit memory**), width-concatenated as `[z_t | cond1 | cond2]`.
- **Two-phase runtime:** an **offline** stage builds the global point cloud (panorama →
  MoGe depth → unproject); the **online** frame model renders + denoises in real time.
- **No temporal recurrence:** stability is bought by *conditioning*, not by a recurrent
  state — so multi-view coherence holds but **frame jitter** is a stated limitation.
- **Speed:** ~**25 FPS @ 512² on an H-series GPU**, **10 FPS on RTX 4090** [p12], via DMD
  distillation + `torch.compile` + VAE/condition latent caching.

## The framing that makes this paper teachable

Most world-model papers bury their choices. This one **organizes the field around six
framing axes** and, for each, gives a **taxonomy of approaches → InSpatio-WorldFM's pick
→ the rationale and the trade-off**. That structure is itself the big picture:

| Framing axis | One-line definition | Failure mode it attacks |
|---|---|---|
| **Generation paradigm** | Frame-based vs video/window temporal recurrence | Window-level latency; accumulating spatial error |
| **Real-time / latency** | Offline heavy multi-step vs real-time consumer-GPU | Non-interactive inference; can't run locally |
| **Spatial consistency** | How 3D coherence is enforced across views | Geometry drift; revisit looks wrong |
| **Training-stage evolution** | Single-stage vs progressive curriculum | Unstable control; overfit to strong anchor |
| **Offline vs online** | One-pass vs two-phase anchor-build + synthesis | Motion boundary at the handoff |
| **NVS** | Feedforward / multi-view diffusion / single-ref real-time | Can't do free, interactive exploration |

> **Key distinction the paper insists on** [§1, p2]: *video-based* world models inherit
> strong motion priors but suffer **(a) interactive latency** (window-level attention/
> decoding — "each generation step must process all frames in the window") **and (b)
> accumulating spatial errors** (short-term temporal continuity is optimized, not
> long-term spatial consistency). InSpatio-WorldFM's answer is to **remove the window
> entirely** and re-derive consistency from **explicit 3D structure baked into each
> frame's conditions**. The cost — spelled out in [§4.1, p13] — is **noticeable frame
> jitter**, because nothing ties consecutive frames together.

## The core loop (two-phase: offline anchors → online per-frame synthesis)

```
# ── OFFLINE: build global 3D anchor + reference appearances ─────────────
panorama  = HunyuanWorld.image_to_panorama(input_image)     # modules/panogen.py:71; run_pipeline.py:299
depth     = MoGe.split_infer_merge(panorama, 42 fib-sphere views, fov 45)  # moge_pano.py:20-24,57-66; run_pipeline.py:354-379
xyz, rgb  = unproject_panorama_depth(depth)                 # pano_postprocess.py:123-149  (compute_ply_arrays)
cloud     = TorchPointCloudRenderer(xyz, rgb)               # modules/point_renderer.py:26-29
cond_db   = build_condition_db_in_memory(views)             # modules/depth_selector.py:128

# ── ONLINE: ONE independent frame per target pose ───────────────────────
for pose in target_poses:                                   # run_pipeline.py:543-567  (per-frame loop)
    x_hat_tgt = cloud.render(K_tgt, c2w_tgt)                # point_renderer.py:89  (cond1 = explicit anchor)
    ci        = select_best_condition_index(depth, K_tgt,..)# depth_selector.py:169  (cond2 = implicit memory)
    z         = randn_latent()                              # worldfm_infer.py:441
    # 2-step DMD denoise: t=999 -> t_mid=200 -> 0           # worldfm_infer.py:464-484
    x0_a   = denoise(z,        t=999, [z_t|x_hat_tgt|cond2])# tri-condition WIDTH concat, PixArtWorldFMMS.py:449
    z_mid  = renoise(x0_a,     t=200)                       # worldfm_infer.py:474  (a_mid/s_mid, ts_mid :218-222)
    x0_b   = denoise(z_mid,    t=200, [z_t|x_hat_tgt|cond2])# dedicated refinement pass
    x_tgt  = VAE.decode(x0_b)                               # worldfm_infer.py:178-271  (vae_scale 0.13025 :214)
    emit(x_tgt)                                             # run_pipeline.py:813,822  (per-frame PNG | mp4@30fps)
```

The DiT's joint self-attention sees **one frame's worth of tokens**, with the three
conditions sitting **side by side in width** (latent 64² × 3 → patch grid 32×96 with
`patch_size 2`, i.e. **≈3k tokens**; `width_multiplier=3` at
`modules/worldfm_infer.py:161`). After the transformer, **only the target slice is
retained** (`PixArtWorldFMMS.py:707-713`). There is **no multi-frame batching in the
model** — the per-call cost is the unit of interaction, which is the whole point.

## The framing axes → design (the heart)

### 1) Generation paradigm: frame-based vs video-based — §2.4.1, §1

The paper taxonomizes world models by **how frames relate**:
- **Video / window-based temporal world models** [16,38,26,11,48,29] — sequential frames
  generated within a temporal window; bidirectional attention + full-window decoding.
  *Strong motion/appearance priors, but window-level dependency caps real-time
  interaction and lets spatial errors accumulate* [§1, p2].
- **Frame-based** — each frame generated **independently**; World Labs' **RTFM** [37] is
  the only named peer, and it is closed.

**InSpatio-WorldFM's choice: frame-based, independent per-frame generation.** It
"directly incorporates spatial structure into the generation of individual frames" via
"a minimal frame-based architecture … generating each frame independently" [§2.4.1, p4;
§1, p2]. **Trade-off:** eliminates window-level inference overhead and enables true
real-time interaction, **but removes inter-frame temporal constraints** — multi-view
consistency must be recovered **purely through conditioning** (no temporal recurrence),
which produces **noticeable frame jitter** during interaction [§4.1, p13].
→ code: one denoising call per target pose, `modules/worldfm_infer.py:386-494`
(`infer_from_render_u8`); per-frame loop `run_pipeline.py:543-567`.

### 2) Real-time / interactive latency — §1, §3

- **Offline heavy multi-step generation** — high-fidelity but seconds-per-frame.
- **Real-time few-step, consumer-GPU deployable** — the engineering goal.

**InSpatio-WorldFM's choice: true real-time, interaction-friendly, consumer-GPU
deployable.** Window-level dependency "fundamentally limits the ability of such models to
support truly real-time interaction" [§1, p2]; the frame paradigm enables "low-latency
frame synthesis." Quantified [§3, p12]: **~25 FPS @ 512² on an H-series GPU**, **10 FPS on
RTX 4090**. **Trade-off:** real-time speed is bought by **DMD few-step distillation
(2-step)** plus heavy engineering (`torch.compile`, VAE/condition latent caching). The
"minimal perceptual difference" after distillation is **asserted qualitatively, not
numerically substantiated** [§3, p12-13] — there are **no FID/PSNR/LPIPS/consistency
metrics** anywhere. ⚠️ The "H-series GPU" is **not pinned to a SKU** (e.g. H100); treat
the FPS figure as approximate. → code: 2-step DMD `modules/worldfm_infer.py:464-484`;
compile + caching `modules/worldfm_infer.py:164-211,356-381,410-438`.

### 3) Spatial / 3D consistency mechanism — §2.4.1 (Hybrid Spatial Memory), Fig.3

Organized by the **indexing principle** of spatial memory:
- **Posed-frame primitive memory** — **RTFM** [37] uses posed frames as primitives.
- **Keyframe feature-warping** — **StarGen** [45] warps features from posed keyframes.
- **Hybrid explicit-3D-anchor + implicit-neural-memory** — both a rendered geometric
  proxy *and* a learned appearance memory.

**InSpatio-WorldFM's choice: hybrid spatial memory = explicit 3D anchor + implicit neural
memory.** For each target view, a **point-cloud render** (`x̂_tgt`, **cond1**) holds coarse
global geometry, while the **reference frame** (**cond2**) supplies fine appearance and
"hallucinate[s] plausible content in unobserved regions" [§2.4.1, p6; §2.2, p4]. The two
interact **solely through self-attention** [Fig.3, p5]. **Trade-off:** two conditioning
channels add bandwidth/complexity, and the explicit anchor's quality is **bounded by
feedforward-reconstruction error** in the point cloud — which is exactly why an **Unreal-
Engine synthetic finetune** is needed to correct pose/depth errors [§2.4.2, p7]; implicit
memory can also hallucinate where unobserved. → code: point render `modules/point_renderer.py:89-191`;
nearest-reference selection `modules/depth_selector.py:169-319,128`; cond2 indexing+cache
`modules/worldfm_infer.py:414-417,410-438`.

### 4) Training-stage evolution (progressive) — §2.2, §2.3–2.5

- **Single-stage finetune** — one pass from a pretrained backbone.
- **Progressive multi-stage curriculum** — staged objectives and condition exposure.

**InSpatio-WorldFM's choice: a three-stage progressive pipeline** — image prior
(Stage I) → controllable frame model + spatial memory (Stage II) → real-time few-step
generator (Stage III) [§1, p2; §2.2, p4]. The Stage-II curriculum — **noise-schedule
biasing** toward high-noise `t`, **progressive condition injection** (implicit memory
first, explicit anchor later to avoid overfitting the strong anchor), and **random anchor
masking** — stabilizes geometric learning [§2.4.1, p6-7]. **Trade-off:** the curriculum
adds scheduling complexity, and **it is entirely training-side — NOT reproducible from
this (inference-only) repo**. 📝 No `train.py`, dataset/dataloader, or optimizer is
shipped (the inherited `gaussian_diffusion.py:744,857` `training_losses` boilerplate is
not wired to any released loop). → deeper dive: `docs/06`.

### 5) Offline vs online operation — §2.1 (Overview), Fig.2

- **Fully-online single-pass** — everything computed at query time.
- **Two-phase offline-anchor-build + online-real-time-synthesis.**

**InSpatio-WorldFM's choice: two-phase.** Offline, a **multi-view-consistent /
panorama model** generates observations that provide 3D anchors + reference appearances,
and a **reconstruction model** supplies depth [§2.1, p3; Fig.2, p3]. Online, the efficient
frame model does real-time synthesis and "updates scene content at keyframes." **Trade-off:
a MOTION BOUNDARY at the online handoff** — the offline panorama/multi-view provider is
high-compute and memory-heavy, restricted to offline, so the explorable volume is bounded
[§4.1, p13]. ⚠️ **Release caveat:** the **internal panorama generative model is withheld**
(README:52-53); the shipped path substitutes **HunyuanWorld-1.0**, and reproducibility of
"consistent scene generation" depends on that substitute. → code: panorama
`modules/panogen.py:71`; MoGe depth→cloud `run_pipeline.py:354-379`, `modules/moge_pano.py:20-24,57-66`.

### 6) Multi-view / novel-view synthesis (NVS) — §2.2 (Formulation)

- **Feedforward stereo/NVS** — fast but limited range.
- **Multi-view diffusion (offline)** — high consistency but offline/heavy [49,14].
- **Conditional single-reference real-time NVS** — one reference, free camera motion.

**InSpatio-WorldFM's choice: conditional NVS from a single reference image under user
camera motion.** Given `x_ref` with pose `π_ref`, generate target-view `x_tgt`
geometrically consistent with `x_ref` for a target pose `π_tgt` [§2.2, p3]. Multi-view-
consistent training data is a **named contribution** [§1, p2]. **Trade-off:** single-
reference NVS is cheap and interactive but has a **limited motion boundary** and "cannot
generate dynamic content with high quality/stability" [§4.1, p13]. → formulation deep
dive: `docs/05`.

## What this paper is — and isn't

- **Is:** a *real-time NVS system* paper + a *position statement* against video-based
  world models. It tells you *what* InSpatio-WorldFM does and *why*, organized by six
  framing axes, and positions ~50 prior works across the taxonomies (camera encoding,
  spatial memory, distillation, 3D reconstruction).
- **Isn't** a quantitative evaluation: there are **no benchmark tables, no FID/PSNR/
  LPIPS/multi-view-consistency metrics** [§3, p12-13]; evaluation is **qualitative only**
  (Figs.4-8, each "1 reference + 10 frames from different viewpoints"). "Strong
  multi-view consistency" and "minimal perceptual difference" after distillation are
  **asserted, not measured**.
- **Isn't** a training release: the repo is **inference-only**. All three training stages,
  data curation, and the training strategies (noise biasing, progressive injection,
  anchor masking, UE finetune, the DMD two-score objective) are **paper-only**. Only the
  *already-distilled* 1-/2-step inference is functional.
- Treat the paper's §2 as the **design rationale**; get the *formulation, training recipe,
  and engineering internals* from [`03-methodology-reference.md`](03-methodology-reference.md),
  [`06`](06-training-distillation.md), and [`08`](08-runtime-and-decode.md).

## Inputs / outputs / data

- **Inputs:** one **reference image** `x_ref` with pose `π_ref` (cond2 / implicit memory);
  a **target pose** `π_tgt = (K, E)`; an **offline-built global point cloud** rendered to
  the target view as `x̂_tgt` (cond1 / explicit anchor). ⚠️ **No text at inference** —
  caption cross-attention is disabled (`disable_cross_attn=True`,
  `modules/worldfm_infer.py:97`), caption embeddings zeroed (`:233-240`), despite the
  PixArt-Σ text-to-image lineage.
- **Output:** target-view RGB `x_tgt` at **512×512**, one frame independently per target
  pose; written as per-frame PNG (`--save_mode image`) or stitched MP4 (default 30 fps)
  [p12; `run_pipeline.py:813,822,655`]. No separate super-resolution/refinement stage.
- **Data (stated):** real sources — internet video, **DL3DV** [22], **RealEstate10K**
  [51], authors' own captures; synthetic — **Unreal Engine** [12] with precise GT
  pose+depth [§2.4.1, p6; §2.4.2, p7]. Per clip, 16 frames → **4 reference** (build the
  global point cloud) + **12 targets**; pose/depth via **MapAnything** [17]. ⚠️ Dataset
  **size, #clips, total frames, model params, and training compute are NOT stated**.

---

## InSpatio-WorldFM — Methodology Card (normalized, for cross-paper comparison)

> This card is the **unit of comparison** leader agents will fill for every paper.
> Schema defined in [`02-comparison-framework.md`](02-comparison-framework.md).

| Axis | InSpatio-WorldFM |
|---|---|
| **Identity** | InSpatio-WorldFM (InSpatio Team), arXiv 2603.11911 v3, 2026-05-06. Open-source, **Apache-2.0** (library code only; HunyuanWorld-1.0 & MoGe submodules separately licensed). |
| **Base backbone** | **PixArt-Σ** latent DiT, XL_2 config (depth 28, hidden 1152, patch 2, 16 heads) [§2.3, p4]. |
| **Setting / Task** | Real-time **conditional NVS** — one reference image + target pose(s) → target-view frame(s); interactive/free-exploration. |
| **Paradigm** | **Frame-based latent diffusion**, independent per-frame (rejects video/window temporal recurrence). |
| **Conditioning inputs** | reference image `x_ref` (cond2), point-cloud render `x̂_tgt` (cond1), target pose `π_tgt`. **No text** at inference. |
| **Control mechanism** *(Challenge 1)* | **PRoPE** (Cameras-as-RPE) camera encoding [§2.4.1, p5-6] — ⚠️ **adopted in the paper but NOT activated on shipped inference**; compared vs **Plücker-ray** [15,1,2] and **pure-parametric** [3]. |
| **Memory mechanism** *(Challenge 2)* | **Hybrid:** explicit point-cloud anchor (cond1) + implicit nearest-reference self-attention (cond2), width-concatenated `[z_t\|cond1\|cond2]`. |
| **Stability / anti-drift** *(Challenge 3)* | **No temporal recurrence** — consistency via *conditioning*, not recurrence; training-side regularization (noise biasing, progressive injection, anchor masking) + UE synthetic finetune — **all 📝 paper-only**. |
| **Runtime** *(Challenge 4)* | **DMD 2-step distillation** (`t_mid=200`) + `torch.compile` + VAE/condition latent caching. |
| **Resolution / fps / latency** | **512×512**, ~**25 FPS (H-series)** / **10 FPS (RTX 4090)** [p12]; default `step=2`. |
| **Distinguishing idea** | Generate each frame **independently** yet stay globally 3D-consistent by conditioning every frame on **both** an explicit point-cloud anchor **and** an implicit reference memory reached **only via self-attention** — then distill to 2 steps. |
| **Stated limitations** | Weak **dynamic content**; **limited motion boundary** at the offline→online handoff; **frame jitter** (no temporal constraints); **qualitative-only** evaluation [§4.1, p13]. |
| **Lineage** | PixArt-Σ / DiT (backbone), LDM (latent diffusion), DMD/VSD (distillation), PRoPE & Plücker (camera), RTFM/StarGen (memory), VGGT/DUSt3R/MoGe/MapAnything (3D recon), HunyuanWorld-1.0 (panorama substitute). |

> **Deeper methodology** (formulation, 3-stage training, data, distillation, runtime
> internals) → [`03-methodology-reference.md`](03-methodology-reference.md). A
> **machine-readable** version of this card → [`worldfm.card.yaml`](worldfm.card.yaml).

## Code map (condensed — full evidence in `docs/04`)

| Mechanism | Where |
|---|---|
| Independent per-frame inference (1 denoise call/target pose) | `modules/worldfm_infer.py:386-494`; loop `run_pipeline.py:543-567` |
| Tri-condition width-concat `[z_t\|cond1\|cond2]` + target-slice retention | `worldfm/diffusion/model/nets/PixArtWorldFMMS.py:449,707-713` |
| Self-attention-only injection (cross-attn disabled) | `modules/worldfm_infer.py:97`; `run_pipeline.py:515` |
| Explicit 3D anchor: pure-PyTorch point-cloud splat render | `modules/point_renderer.py:89-191`; PLY `modules/pano_postprocess.py:123-149` |
| Implicit memory: nearest reference (depth + view-angle + distance) | `modules/depth_selector.py:169-319,128`; cond2 cache `modules/worldfm_infer.py:414-417,410-438` |
| Offline anchor build: MoGe panorama depth (42 fib-sphere views) | `run_pipeline.py:354-379`; `modules/moge_pano.py:20-24,57-66` |
| Offline panorama generation (HunyuanWorld-1.0 substitute) | `modules/panogen.py:71`; `run_pipeline.py:299` |
| DMD 1-step (`t=999`) and **2-step** (`999→200→0`, preferred) | `modules/worldfm_infer.py:455-463,464-484`; `mid_t=200` `:99` |
| Multi-step DPM-Solver++ teacher (CFG=4.5) | `modules/worldfm_infer.py:497-567`; `worldfm/diffusion/dpm_solver.py:6-36` |
| VAE encode/decode (AutoencoderKL, `vae_scale 0.13025`, deterministic) | `modules/worldfm_infer.py:178-271,214` |
| PRoPE camera encoding (adopted; ⚠️ inactive at inference) | `worldfm/diffusion/model/nets/prope.py:52-140`; gate `PixArtWorldFMMS.py:85-89,488` |
| Plücker-ray camera encoding (compared; ⚠️ inactive) | `worldfm/diffusion/model/nets/plucker.py:8-72`; `PixArtWorldFMMS.py:480,639-666` |
| Engineering: `torch.compile` + condition/VAE latent caching | `modules/worldfm_infer.py:164-211,356-381`; `run_pipeline.py:771` |
| Config (default `step=2`, `cfg_scale=4.5`) | `default.yaml:50,55` |
| Checkpoint download (1/2-step + VAE) | `worldfm/download.py`; `download_ckpts.py` |

## Open questions — status

The full paper (`paper/2603.11911v3.pdf`) + code **resolve most** of the framing questions:

- ✅ **Distillation schedule** — 2-step with `t_mid=200` on a 1000-step linear schedule;
  2-step beats 1-step; consistent across paper [§2.5/§3, p7,p12] and code (`mid_t=200`
  `modules/worldfm_infer.py:99`; `iddpm.py` 1000-step).
- ✅ **Condition injection design** — self-attention-only, width-concat; cross-attn
  rejected; matches paper's adopted design [§2.4.1, p4] and code (`disable_cross_attn=True`
  `:97`; `use_cond2_cross_attn=False` `:452`).
- ✅ **Runtime composition** — offline anchors (panorama+MoGe) + online frame model; the
  internal panorama model is withheld, HunyuanWorld-1.0 substitutes (README:52-53).
- ✅ **Two-phase / motion boundary** — confirmed as a stated limitation [§4.1, p13].

**Still open / to verify in code:**
- ⚠️ **PRoPE active at inference?** Paper's *adopted* camera encoding is **implemented**
  (`prope.py:52-140`) but `worldfm_infer.py` **never passes** `use_prope`/`prope_viewmats`/
  `Ks`, so the shipped 1/2-step checkpoints run with **plain self-attention and no explicit
  camera modulation**. Unclear whether released weights were trained with PRoPE and the
  path is merely bypassed, or camera conditioning is effectively absent in the public
  model. (Plücker likewise implemented but inactive; pure-parametric has **no code**.)
- ⚠️ **`cfg_scale` config vs runtime** — `default.yaml:55` sets `cfg_scale: 4.5`, but the
  shipped 1/2-step DMD path runs at **`cfg_scale: 0.0`** (`worldfm_infer.py:100`,
  `run_pipeline.py:517`); 4.5 applies **only** to the multi-step DPM teacher.
- ⚠️ **Model parameter count** — not stated (do **not** fabricate a B-count); architecture
  is PixArt-Σ XL_2.
- ⚠️ **Quantitative metrics** — none (FID/PSNR/LPIPS/consistency); evaluation is
  qualitative only [§3, p12-13].
- ⚠️ **Dataset size / training compute / `H-series` SKU** — all unstated.
- ⚠️ **MoGe submodule** — must be checked out at pinned commit `7807b5de` (`setup.sh`);
  submodules are empty in this checkout and must be initialized before the pipeline runs.
