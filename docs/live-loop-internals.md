# Live-loop internals — how the cached anchors & per-frame path actually work

> Companion to [`live-loop.md`](./live-loop.md) (the **ops recipe**) and
> [`learning/08-runtime-and-decode.md`](./learning/08-runtime-and-decode.md) (the **decode**
> internals). This doc is the *why it works* layer discovered while building `live_server.py`: what
> the cached offline anchors really are, why an arbitrary camera pose can be served at runtime, and
> the per-frame stochasticity limitation. All claims verified against source `path:line` on
> 2026-07-29 (repo commit `d51ada2`).

## 0. Atomic scope

| You want | Go to |
|---|---|
| How to **run** the live loop | [`live-loop.md`](./live-loop.md) |
| The **decode** internals (cond1/cond2 VAE, DMD steps, `torch.compile`) | [`learning/08-runtime-and-decode.md`](./learning/08-runtime-and-decode.md) |
| The 4-step pipeline + per-frame function signatures | [`learning/08`](./learning/08-runtime-and-decode.md) §"two-phase pipeline" |
| This doc | the cached-anchor **cache structure**, **arbitrary-pose** pose-generality, **stochasticity** |

## 1. The cached offline anchors — what `--reuse_intermediates` actually loads

The offline phase (steps 1–3) turns one perspective image into a panorama, a 3D point cloud, and a
fixed library of 42 "condition" views, cached under `outputs/<name>/intermediates/`. The online phase
(`--reuse_intermediates`, and the live server) reloads them via `_load_postprocess_result`
(`run_pipeline.py:225-250`). **Only three artifacts are ever reloaded; the rest are write-only
forensic files with dead loaders.**

| File in `intermediates/` | Reloaded online? | Why / note |
|---|---|---|
| `postprocess_arrays.npz` | ✅ **required** | `savez_compressed` of `pano_bgr`, `depth`, `ply_xyz (N,3)`, `ply_rgb (N,3) uint8` (`run_pipeline.py:214-222`). ~148 MB on disk → ~340 MB decompressed (8.39 M points for `mario`). **Only `ply_xyz`/`ply_rgb` are used online**; `pano_bgr`/`depth` are loaded then unused. |
| `transforms_condition.json` | ✅ **required** | 42 condition camera poses (`K` + `c2w` per view) — parsed by `modules/transforms_io.py`. ~34 KB. |
| `conditions/*.png` (42) | ✅ **required** | the 42 condition RGB views (504×504). `FileNotFoundError` if **any** is missing (`run_pipeline.py:240`). Needed to build the cond2 latent cache. |
| `pointcloud.ply` | ❌ forensic | binary PLY; its reader `ply_io.load_ply_xyz_rgb` (`ply_io.py:102`) has **zero callers** — the online path gets xyz/rgb from the npz. Safe to delete to slim the cache. |
| `moge_depth_raw.npy`, `moge_mask.png` | ❌ forensic | write-only; `load_depth_npy` (`pano_postprocess.py:61`) is dead code. |
| `panorama.png` (inside `intermediates/`) | ❌ forensic | the 4096×2048 resized pano used to build the PLY. Distinct from the root `outputs/<name>/panorama.png` (1024×512 FLUX step-1 cache). Neither is needed online. |

> **The "condition DB" is not a file.** The ~42-entry `ConditionDB` (`modules/depth_selector.py:25`:
> `P_views`, `depth_views`, `C_views`) is **rebuilt in memory every run** by re-rendering the PLY at
> each of the 42 condition poses (`build_condition_db_in_memory` `:127-161`). It is reconstructed
> from the PLY + transforms, not deserialized. (Its pixels are even ignored — `depth_views` come from
> the renderer, not the condition PNGs.)

**Implication for the live server:** it only reads the npz + JSON + 42 PNGs once at preload
(`live_server.py` `preload()`); the heavy FLUX (step 1) and MoGe (step 2) stages never run, so the
process stays inside the 16 GB budget. You can safely delete the four forensic files to shrink the
cache from ~330 MB to ~180 MB without affecting the live loop.

## 2. Why arbitrary-pose live looping works (the renderer is pose-general)

The decisive finding: **the per-frame path accepts any camera pose, not just the precomputed
trajectory.** The shipped demo feeds `meta.json`'s `c2w` list, but nothing requires membership in
that list. Per frame, WorldFM consumes two image conditions, produced as follows:

- **cond1 — rendered fresh for the requested pose.** `step3_render_one` calls
  `TorchPointCloudRenderer.render_torch(K_3x3, c2w_4x4, c2w_is_camera_to_world=True)`
  (`run_pipeline.py:458`; `modules/point_renderer.py:194-208`). This is a pure-PyTorch point-cloud
  splatter: it projects the cached PLY (`w2c = inv(c2w)`, `z = X[:,2]` depth, standard pinhole)
  **for any 3×3 `K` + 4×4 `c2w`**, with **no trajectory-membership check** (`point_renderer.py:88-109`).
  So cond1 is a fresh splat of the geometry from wherever the user puts the camera.
- **cond2 — selected, not rendered.** `select_best_condition_index` (`modules/depth_selector.py:169-319`)
  scores the 42 fixed panorama views by depth-consistency + view-angle + distance-weight and returns
  the argmax index (`:310`). cond2 is then fetched from the pre-encoded latent cache by that index
  (`worldfm_infer.py:414-417`) — it is **not** re-rendered for the new pose.

**Pose convention (important for any client):** OpenCV camera-to-world `c2w` — camera `+X` = right,
`+Y` = down, `+Z` = forward (look direction). `K` is the standard 3×3 intrinsics (demo
`fx=fy=320, cx=cy=256`). `axis_flip` defaults to none (`point_renderer.py:80-86`). The `live/viewer.html`
client builds this `c2w` from an OpenCV look-at and self-checks it reconstructs the scene's `c2w0`.

**Graceful degradation (no crashes):** for a pose whose rendered depth has no valid samples (e.g.
looking outside the reconstructed volume), `select_best_condition_index` returns `(0, 0, 0)` — index 0,
**no exception** (`depth_selector.py:234-235`). cond1 simply goes sparse/blank and cond2 snaps to view
0, so the output fades rather than crashes. `max_view_angle_deg=180` in `default.yaml:41`, so angle
gating is off. → **the cached anchors are fully sufficient for free-fly live looping; stay near the
reconstructed volume for good cond1 coverage.**

This is why the live loop is a **wrapper, not a rewrite**: the code already separates one-time setup
(`run_pipeline.py:766-782`) from a pose-general per-frame body (`:788-804`). `live_server.py` calls
`step3_render_one` + `step4_infer_one` with user-supplied `K`/`c2w` and reuses the resident
`renderer`/`cond_db`/`svc`.

## 3. Per-frame stochasticity (the motion-flicker limitation)

The denoiser noise `z = torch.randn(...)` is drawn **fresh every call with no seed** in the inference
path (`modules/worldfm_infer.py:441`; only step-1 panorama generation seeds, `panogen seed=42`).
Consequences:

- **Same pose → different image each call.** The hallucinated detail boils/flickers even when you
  hold still — UNLESS you seed.
- **`live_server.py` mitigates with per-pose seeding:** `torch.manual_seed(hash(rounded c2w, K))`
  before each `step4_infer_one` (`_pose_seed`). A held view is then stable (seed is constant).
  Disable with `--no_seed` for "alive" per-frame noise.
- **Motion flicker remains.** Seeding per-pose gives *spatial* stability (same pose ⇒ same image)
  but **not temporal coherence** — as the camera moves, the seed changes every frame, so the detail
  still shimmers across frames. This is the visible artifact during a fly-through. True temporal
  coherence would require **carrying the noise latent across frames** (feed the previous frame's `z`
  / output forward), which the released inference path does not do — a future improvement, not a bug.

This is a property of the **frame-based paradigm** (each view generated independently — see
[`learning/08`](./learning/08-runtime-and-decode.md) "why there is no streaming-decode constraint"):
real-time is structurally achievable, at the cost of inter-frame coherence.

## 4. Minor gotchas

- **`PIL.Image.fromarray(arr, mode="RGB")` deprecation** — used in `set_cond2_from_array`/
  `set_cond2_candidates_from_arrays` (`worldfm_infer.py:329,370`) and emits a `DeprecationWarning`
  ("removed in Pillow 13, 2026-10-15"). Harmless today; pin Pillow <13 or drop the `mode=` kwarg if
  you upgrade. The live server's own JPEG path uses `cv2.imencode`, so it is unaffected.
- **`torch.cuda.max_memory_allocated()` ≠ `nvidia-smi`.** The `/metrics` VRAM number is
  torch-allocated live tensors only (~1.8 GB); `nvidia-smi` shows the full process footprint
  (~4.8 GB incl. the caching allocator's reserved pool + CUDA context). Both are correct, different
  bases — don't compare them directly. The launcher's peak comes from `nvidia-smi`.
- **`cfg_scale` is ignored on the 1-/2-step DMD path** (hardcoded `0.0` at `step4_init`
  `run_pipeline.py:524`; only the multistep teacher uses it). Don't expose it as a live-loop knob
  unless you wire the multistep path. See [`learning/08`](./learning/08-runtime-and-decode.md) §cfg_scale.
- **Fixed shapes for CUDA graphs:** `image_size`/`render_size`/dtype must stay byte-identical to init
  or the `reduce-overhead` compiled graphs re-trace (a tens-of-seconds stall mid-traffic). The live
  server keeps them fixed at 512²/fp16.
- **`cond2 not set`:** the first request after a restart raises `RuntimeError` at
  `worldfm_infer.py:395-396` if `set_cond2_candidates_from_arrays` didn't run at preload — the server
  does it in `preload()`, so this only bites if you skip preload.

## Cross-links

- [`live-loop.md`](./live-loop.md) — run/ops recipe for `live_server.py` + `live/viewer.html`.
- [`learning/08-runtime-and-decode.md`](./learning/08-runtime-and-decode.md) — decode internals, DMD steps, `torch.compile`, measured FPS.
- [`repo-mods.md`](./repo-mods.md) — the 16 GB working-tree edits (NF4, MoGe resolution, merge caps).
- [`../../docs/oom-guardrailing.md`](../../docs/oom-guardrailing.md) — the OOM self-guard + memory-budget techniques.
- [`../../docs/realtime-model-serving.md`](../../docs/realtime-model-serving.md) — the general pattern for wrapping a GPU model in an interactive server (this live loop is the worked example).
