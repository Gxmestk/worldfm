# Running the WorldFM interactive live loop

> A practical **run/ops recipe** for the interactive WebSocket fly-through built around the WorldFM
> per-frame API. The shipped demo (`run_pipeline.py`) is a **batch renderer** (fixed trajectory →
> MP4); `live_server.py` turns the same per-frame path into a **real-time, pose-driven server**
> (camera pose in → generated frame out, ~3.7 FPS at step-1).
>
> **Target box:** the same 16 GiB RAM cgroup / RTX 4060 Ti 16 GB Incus container as
> [`run-guide.md`](./run-guide.md). Verified working **2026-07-29**: WS stream 3.67 FPS, peak VRAM
> 4766 MiB (nvidia-smi), MemAvailable ~9.8 GB free.

## 0. Atomic scope

This doc is **only** the live-loop run recipe. For everything else, follow the cross-links:

| You want | Go to |
|---|---|
| The 4-step pipeline, `step3_render_one`/`step4_infer_one` internals | [`learning/`](learning/) — esp. [`04-paper-code-crosswalk.md`](learning/04-paper-code-crosswalk.md), [`08-runtime-and-decode.md`](learning/08-runtime-and-decode.md) |
| How to run the **batch** demo | [`run-guide.md`](./run-guide.md) |
| The working-tree edits that make WorldFM fit 16 GB | [`repo-mods.md`](./repo-mods.md) |
| OOM / self-guard / quantization techniques | [`../../docs/oom-guardrailing.md`](../../docs/oom-guardrailing.md) |

## 1. What it is

`live_server.py` (repo root) is an **aiohttp** server that reuses the cached offline anchors in
`outputs/<name>/intermediates/` and the existing per-frame functions from `run_pipeline.py`
(`step3_init`, `step3_render_one`, `step4_init`, `step4_infer_one`) — it is a **wrapper, not a
rewrite**. No FLUX or MoGe is loaded (reuse-only), so the process stays well inside the 16 GB budget.

Architecture (one sentence): preload the resident objects once, then for each request call the
two-line per-frame body — all GPU work **serialized on one worker thread** (see §5).

| File | Role |
|---|---|
| `live_server.py` (repo root) | Server: preload, single inference worker, endpoints, warmup, metrics |
| `live/viewer.html` | In-browser orbit fly-through viewer (canvas + WS) |
| `scripts/serve.sh` | Launcher: `LD_LIBRARY_PATH` + `uv run` + PID self-guard, logs to `logs/` |

## 2. Prerequisites

- The uv venv + `LD_LIBRARY_PATH` from [`run-guide.md` §1–§2](./run-guide.md). (`serve.sh` sets
  the `LD_LIBRARY_PATH` itself.)
- **Cached offline anchors** already built: `outputs/mario/intermediates/`
  (`postprocess_arrays.npz`, `transforms_condition.json`, `conditions/*.png`). These come from a
  successful [`run-guide.md` Phase 1](./run-guide.md) / `--prepare_only` run; the demo set already
  exists on this box. No new dependencies — `aiohttp` and `cv2` are already pinned/installed.
- **Port 8123 free.** (Port 8000 is taken by the host's `kiro-gateway`; override with `PORT=`.)

## 3. Start it

```bash
bash scripts/serve.sh                 # foreground; Ctrl-C to stop
# env overrides: PORT=8123 HOST=0.0.0.0 STEP=1 RAM_LIMIT=2500 bash scripts/serve.sh
```

Boot is **~30–40 s**: it loads the cached anchors, builds the renderer + condition DB, loads the
WorldFM checkpoint + VAE (`torch.compile`), VAE-encodes the 42 cond2 candidates, then runs 3 warmup
frames (the first captures the compiled CUDA graphs, ~11 s). The server only starts listening once
warmup completes; watch progress in `logs/serve.log`. When you see
`[live] READY — viewer at http://0.0.0.0:8123/`, it is serving.

The launcher self-guards the 16 GB cgroup: it polls `/proc/meminfo` and, if `MemAvailable` drops
below `RAM_LIMIT` (default 2500 MiB), kills its **own process group by PID** (`kill -9 -"$PY"`,
**never** `pkill -f`) — same pattern as `runs.sh`.

## 4. Access & endpoints

The server binds `0.0.0.0:8123`. From the container: `http://localhost:8123/`. From the **host
browser**, this container's webapps are reachable directly by hostname — no `incus` proxy needed:

```
http://s-gtksoon1.lan.bsthun.in:8123/
```

| Endpoint | What |
|---|---|
| `GET /` | The viewer (`live/viewer.html`). **drag** orbit · **scroll** dolly · **R** reset. |
| `GET /health` | `{status, step, name, subscribers}` (`ready` once warmup done). |
| `GET /scene` | `{name, step, image_size, render_size, K, c2w0}` — the scene + home pose. |
| `GET /frame?c2w=<4×4 JSON>&K=<3×3 JSON>` | One-shot render → `image/jpeg` (512×512). Defaults to `c2w0`/`K`. |
| `WS /stream` | Bidirectional: client sends `{c2w, K}` JSON; server pushes binary JPEG frames + `{type:"stats", idx, ms, fps}` JSON. |
| `GET /metrics` | `{vram_alloc_mb, vram_peak_mb, mem_available_mb, fps, subscribers}`. |

**Pose convention:** OpenCV camera-to-world `c2w` (4×4, columns = right/down/forward; `+Z` = look
direction) + `K` (3×3; demo `fx=fy=320, cx=cy=256`). Arbitrary poses are accepted — cond1 is
re-rendered from the cached PLY for any pose; cond2 is *selected* (nearest of the 42 cached views).
Far off-volume poses degrade gracefully (sparse cond1, cond2 snaps to the closest view) rather than
crash.

## 5. How it works (and why it's built this way)

- **Reuse-only.** It never calls `setup_external_repos`/step1/step2 — MoGe and HunyuanWorld are not
  loaded, so only the renderer + condition DB + WorldFM service stay resident (~4.8 GB VRAM).
- **One inference worker thread.** `torch.compile(mode="reduce-overhead")` uses CUDA graphs that are
  **thread-affine**: the compile setup (`step4_init`) and *every* call — including the first lazy
  capture — must run on the same thread, or the compiled VAE/model forward asserts. So `preload()`
  *and* all `/frame`+`/stream` jobs run on one dedicated worker thread; the asyncio loop never
  touches torch. (This is the one non-obvious gotcha — see troubleshooting.)
- **Latest-pose-wins + broadcast.** Streaming keeps only the newest requested pose and broadcasts
  each rendered frame to every subscriber — smooth fly-through with bounded RAM (no frame pile-up).
  `/frame` is a one-shot job that takes priority over the stream each worker tick.
- **Per-pose seeding.** The denoiser draws fresh noise each call (the pipeline does not seed), which
  flickers. The server seeds from the pose (`torch.manual_seed(hash(c2w,K))`) so a held view is
  stable. Disable with `--no_seed` for "alive" per-frame noise.

## 6. Expected results

Measured 2026-07-29 (RTX 4060 Ti 16 GB, step-1, reuse-only, after warmup):

| Metric | Value |
|---|---|
| WS stream FPS | **3.67** (271 ms/frame incl. JPEG encode) |
| `/frame` steady | ~256 ms (3.9 FPS) |
| Peak VRAM (nvidia-smi) | **4766 MiB** (~4.8 GB) |
| `torch.cuda.max_memory_allocated` | ~1.8 GB (live tensors only; allocator reserves the rest) |
| MemAvailable (16 GB cgroup) | ~9.8 GB free |
| Boot to READY | ~35 s (preload + 3 warmup frames) |

Convention sanity: `/frame` at `c2w0` matches the cached `outputs/mario/step1_output_0000.png` with
color-histogram correlation **0.996** (diff is per-pose-seeded noise, not a convention error).

## 7. Troubleshooting

- **`address already in use` on bind** → another process holds the port. Port **8000 is the
  `kiro-gateway`** (do not kill it); the launcher defaults to **8123**. Pick another with `PORT=…`.
- **`AssertionError` during warmup, or "VAE compiled encode failed, fallback to eager"** → torch
  inference ran on a different thread than its compile setup. This is fixed in `live_server.py`
  (preload runs on the worker thread). If you refactor, keep all torch on one thread.
- **`No execution plans support the graph`** → missing `LD_LIBRARY_PATH` over the venv's
  `nvidia/*/lib` (see [`run-guide.md` §2](./run-guide.md)). `serve.sh` sets it; if you run
  `live_server.py` directly, export it first.
- **Slow first connection / "loading"** → warmup (~30–40 s). The server only listens after warmup;
  check `logs/serve.log` for `[live] READY`.
- **OOM / self-abort** → the launcher kills its own PID when RAM is low; reuse-only keeps the
  footprint bounded (~2–3 GB RAM steady, one-shot startup spike from the checkpoint load). Never
  `pkill -f` — see [`../../docs/oom-guardrailing.md`](../../docs/oom-guardrailing.md).
