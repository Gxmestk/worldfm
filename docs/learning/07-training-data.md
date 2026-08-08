# 07 — Training Data Deep Dive

> Grounded in §2.4.1 (Training Data, p6), §2.4.2 (Finetuning with Synthetic Data, p7), and
> §2.1 (offline anchor/reference providers, p3). Page refs `[pN]` are to the single paper
> (arXiv 2603.11911v3); section refs `[§x]` use the paper's own numbering; code refs are
> `path:line` against this repo. Status emojis: ✅ in released inference code · 📝 paper-only.
> Verified 2026-07-29.
>
> ⚠️ **Paper↔code note (release scope):** this is an **inference-only release.** No dataset,
> dataloader, sampler, or training entrypoint is shipped — every training-side data mechanism
> in this doc is 📝 paper-only. What *is* in the repo is the **offline anchor-build pipeline**
> (`run_pipeline.py` Step 1–3), which is the inference-time realization of the §2.1 offline
> stage, not the §2.4.1 training corpus. The two are described separately below.

## Data sources `[§2.4.1 p6, §2.4.2 p7]` — real + synthetic, no counts released

The corpus is heterogeneous along two axes: **visual domain** (internet walk-throughs vs
scene-level traversals vs synthetic) and **supervision fidelity** (estimated pose/depth vs
ground-truth). Real captures anchor photorealism and appearance; the synthetic Unreal-Engine
render contributes **precise GT geometry** to repair feedforward-reconstruction error. This is
the named contribution of §1 — *multi-view consistent training data curation* — and it is how
WorldFM learns stable cross-view spatial relationships without temporal recurrence.

| Source | Type | Pose / depth GT | Role |
|---|---|---|---|
| Internet videos | real, open-domain | MapAnything `[17]` (estimated) | broad appearance + camera-motion coverage |
| **DL3DV** `[22]` | real, scene-level traversal (**10,510 videos / 51.2 M frames**, 4K@60fps; COLMAP poses) | MapAnything `[17]` | long contiguous multi-view traversals |
| **RealEstate10K** `[51]` | real, indoor walkthrough (**~10 K YouTube videos → ~80 K clips / ~10 M frames**; SLAM/BA pseudo-GT poses) | MapAnything `[17]` | dense indoor camera motion |
| Own captured videos † | real | MapAnything `[17]` | internal coverage / style |
| **Unreal Engine** `[12]` † | synthetic (precise GT) | **GT camera pose + depth** | corrects feedforward depth/pose errors `[§2.4.2 p7]` |

† internally curated / **not released.** `[Table 1]`-style clip counts do **not** exist in this
paper — dataset size, #clips, total frames, #GPUs, and training compute are all unstated (TBD).

- **Pose recovery:** for every real clip, per-frame intrinsics + poses + depth are recovered
  with **MapAnything** `[17]`, a feedforward metric 3D reconstruction model `[§2.4.1 p6]`.
- **Training record:** `{16 sampled frames, per-frame pose + depth, 4-ref / 12-target split}`
  (see below). No hierarchical caption is attached — WorldFM is **pose/image-conditioned only**.
- **Dynamic-content limit:** both the frame model and this corpus *contain limited dynamic
  content*, which is why WorldFM struggles to generate dynamic scenes `[§4.1 p13]`.

## Per-clip sampling & reference pairing `[§2.4.1 p6]` — the 4-ref / 12-target split

WorldFM's "curation" is **not** a gate pipeline (contrast AlayaWorld's six gate classes). It is a
**multi-view-consistent sample construction**: from each real clip, build one training sample
that pairs every target view with a global 3D anchor and a nearby reference. The recipe, verbatim
from `[§2.4.1 p6]`:

1. **Sample 16 frames** per real video clip (random).
2. **MapAnything** `[17]` estimates per-frame camera pose + depth across all 16.
3. **4 frames → reference group** → unprojected into a single **global point cloud**.
4. **12 frames → training targets**; each target's *reference* is the **temporally closest**
   frame of the -frame reference group.
5. The **point-cloud render at the target viewpoint** (cond1, `x̂_tgt`) is the global cloud
   projected onto the target camera plane `[§2.2 p4]`.
6. **Random shuffling + masking** of the reference/anchor simulate the disorder and discreteness
   of real-world observation order `[§2.4.1 p6]`.

> ⭐ **Why this matters:** steps 3–5 are exactly the hybrid spatial memory the *inference* model
> consumes — explicit 3D anchor (cond1) + nearest reference (cond2). The training corpus is
> constructed to mirror the runtime conditioning, so the model never sees a distribution mismatch
> between train and inference. The `📝` training-side constructs below each have an `✅`
> inference-time code twin:

| Training construct (paper, 📝) | Inference-time realization (code, ✅) |
|---|---|
| 4-frame global point cloud `[§2.4.1 p6]` | `modules/pano_postprocess.py:123-149` (`compute_ply_arrays`, panorama depth → PLY) |
| temporally closest reference selection | `modules/depth_selector.py:169-319` (`select_best_condition_index`, depth-consistency + view-angle + distance weight; `dist_min_m`/`dist_max_m` `default.yaml:43-44`) |
| point-cloud render at target view (cond1) | `modules/point_renderer.py:89-191` (`TorchPointCloudRenderer.render`) |

> ⚠️ **Paper↔code note (selector distance range):** `select_best_condition_index`'s *function*
> default is `dist_min_m=1.0, dist_max_m=5.0` (`depth_selector.py:184-185`), but the shipped
> pipeline overrides this to `1.0 / 20.0` via `default.yaml:43-44`. The wider 20 m ceiling
> reflects inference over a full 360° panorama's 42 condition views, not the 4-frame training
> reference group. — verify intent.

## Synthetic-data finetune (Unreal Engine) `[§2.4.2 p7]` — precise GT pose/depth

The fundamental frame model trained on real data is geometrically *reasonable*, but MapAnything's
feedforward depth/pose estimates *inevitably contain errors* that leak inter-view inconsistency
into the point-cloud renders, destabilizing viewpoint transitions and content persistence
`[§2.4.2 p7]`. The UE finetune exists to correct exactly this.

- **Ground truth:** Unreal Engine `[12]` supplies **precise GT camera poses and depth** — no
  feedforward error.
- **Trajectory synthesis:** start from a semantically valid region; generate motion by **stochastic
  sampling or pre-defined motion templates**; enforce **collision avoidance** for viewpoint validity.
- **Same 4/12 split:** training pairs are constructed analogously to the real pipeline — 4
  reference + 12 targets `[§2.4.2 p7]`.
- **Dose control:** finetune for **"a limited number of steps"** (count not stated — TBD).
  Excessive synthetic exposure would erode natural-appearance priors; empirically even a small
  amount of UE finetuning yields large gains in transition stability `[§2.4.2 p7]`.

📝 **Paper-only.** No UE dataset, trajectory generator, or finetune loop is released.

## The offline anchor-build pipeline (in-code) `[§2.1 p3, Fig.2]`

This is the **synthetic-data pipeline as evidenced in `modules/` + `run_pipeline.py`** — the
offline stage of the two-phase runtime that turns **one** input image into the 3D anchor +
reference appearances the online frame model consumes `[§2.1 p3]`. It is the *inference-time
realization* of the offline providers referenced in §2.1 (multi-view diffusion / reconstruction
/ panorama), not a training-corpus builder.

| Stage | Paper | Code |
|---|---|---|
| **Step 1** panorama (image → 360°) | `[§2.1 p3]` `[31,13,46]` | `modules/panogen.py:71-199` (`Image2PanoramaDemo`, HunyuanWorld-1.0 = FLUX.1-Fill-dev + LoRA); `run_pipeline.py:280-315` (`step1_panogen`) |
| **Step 2** panorama depth (MoGe) | `[§2.1 p3]` `[34,36,35]` | `modules/moge_pano.py:20-24` (`NUM_VIEWS=42`, `FOV_DEG=45.0`), `:57-66` (`_get_panorama_cameras`, Fibonacci sphere); `run_pipeline.py:354-384` (split → infer → merge → `/100.0`) |
| **Step 3** global point cloud (depth → PLY) | `[§2.4.1 p6]` | `modules/pano_postprocess.py:123-149` (`compute_ply_arrays`); `run_pipeline.py:392` (`postprocess_panorama`) |
| **Step 4** condition views + transforms | `[§2.4.1 p6]` | `modules/pano_postprocess.py:285-341` (`generate_conditions`, 42 yaw/pitch views → `transforms_condition.json`) |

Config anchors: MoGe checkpoint `Ruicheng/moge-2-vitl-normal` (`default.yaml:24`),
`resolution_level=30` (`:25`) — ⚠️ **shipped repo default; our 16 GB-fit commit reduces it to `9`**
(see [`repo-mods.md`](../repo-mods.md)); `depth_scale=100.0` default in `postprocess_panorama`
(`pano_postprocess.py:350`); condition intrinsics `cond_fx=cond_fy=320`, `cond_size=504`
(`pano_postprocess.py:353-355`).

> ⚠️ **Paper↔code note (withheld internal model):** *"For consistent scene generation, we employ
> an internal generative model that is not included in the open-source release"* (`README.md:52`).
> The shipped Step-1 path therefore uses **HunyuanWorld-1.0 as an open-source substitute**
> (`README.md:53`, `modules/panogen.py:1-8`). The paper's claim of *"consistent scene generation"*
> depends on the substitute; the README asserts the substitution *"does not impact the core
> spatial reasoning framework"* (`README.md:53`). — verify parity empirically.

> ⚠️ **Paper↔code note (Real-ESRGAN / ZIM are NOT a WorldFM stage):** Real-ESRGAN and ZIM are
> built as submodules (`README.md:31,45,48`) because **`hy3dworld` uses them internally**
> (`run_pipeline.py:258-259`,266) — they are dependencies of HunyuanWorld-1.0's panorama pipeline,
> **not** a separate super-resolution / refinement post-process in WorldFM. There is no SR/refinement
> stage after the frame model; outputs are written at `512×512` as-is. Do not mistake them for an
> AlayaWorld-style optional quality stage.

## Caption annotation — none `[§2.2, §2.4.1]`

WorldFM has **no text-caption supervision path.** Despite the PixArt-Σ text-to-image backbone
`[§2.3 p4]`, the released model is **pose/image-conditioned only**:

- caption cross-attention is **disabled** at inference (`disable_cross_attn=True`,
  `modules/worldfm_infer.py:97`);
- caption embeddings are **zeroed** (`torch.zeros(...)`, `modules/worldfm_infer.py:233,534`).

So the two-level caption schema you would expect from a T2I-derived world model (cf. AlayaWorld's
video-level + segment-level tracks) **does not apply here** — there is no `full_prompt`, no
`camera_path` enum, no `<camera>` dropout. Camera control is carried by the pose inputs
(`π_ref`, `π_tgt`) and the PRoPE/Plucker encoders, not by text (see `05-formulation.md`).

## What is NOT released (data + curation)

- **No dataset / dataloader / sampler** — the 4-ref/12-target construction, MapAnything pose
  recovery, and random shuffling/masking are all 📝 paper-only.
- **No UE synthetic dataset or trajectory generator** — §2.4.2's GT-pose/depth finetune is 📝.
- **No training compute numbers** — #clips, total frames, #GPUs, step counts, finetune dose all
  unstated (TBD).
- **Internal panorama generative model withheld** — HunyuanWorld-1.0 is the substitute
  (`README.md:52-53`); reproducibility of "consistent scene generation" rests on that substitute.
- **MapAnything is training-only** `[§2.4.1 p6]`; the released *inference* anchor-build uses
  **MoGe-2-vitl-normal** instead (`default.yaml:24`). These are different feedforward
  reconstructors — verify the train/inference anchor-quality gap.

---

*Training-side mechanics (noise-schedule biasing, progressive condition injection, random anchor
masking, UE finetune dose) live in `06-training-distillation.md`; the paper↔code status of every
mechanism is tabulated in `04-paper-code-crosswalk.md`; the offline→online two-phase runtime is
detailed in `08-runtime-and-decode.md`; the conditioning math (`x̂_tgt`, `C = {x_ref, π_ref, π_tgt,
x̂_tgt}`) is in `05-formulation.md`.*
