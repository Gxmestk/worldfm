# Running the WorldFM demo on a memory-constrained box

> A practical, copy-pasteable **run/ops recipe** for the InSpatio-WorldFM inference repo
> (`github.com/inspatio/worldfm`, paper arXiv `2603.11911`). This is the *only* doc here about
> how to actually run the demo. It does **not** explain the architecture or reproduce the paper's
> theory — those live in [`learning/`](learning/) (start at
> [`learning/01-big-picture.md`](learning/01-big-picture.md)).
>
> **Target box:** 16 GiB RAM cgroup, no swap, single consumer GPU (here an RTX 4060 Ti 16 GB,
> Incus container `s-gtksoon1`). Every command below was run in exactly that environment and the
> measured results are in [§6](#6-expected-results).

## 0. Atomic scope

This doc is **only** the run/ops recipe. For everything else, follow the cross-links instead of
duplicating:

| You want | Go to |
|---|---|
| Architecture, what each of the 4 steps does, `run_pipeline.py` internals | [`learning/`](learning/) — esp. [`learning/01-big-picture.md`](learning/01-big-picture.md), [`learning/04-paper-code-crosswalk.md`](learning/04-paper-code-crosswalk.md) |
| Runtime/decode internals, `Profiler`, `torch.compile` + condition-cache | [`learning/08-runtime-and-decode.md`](learning/08-runtime-and-decode.md) |
| The full **measured** numbers (this run) | [`../outputs/mario/metrics.md`](../outputs/mario/metrics.md) and the raw [`../outputs/mario/metrics.json`](../outputs/mario/metrics.json) |
| OOM / self-guarding / quantization techniques (deep dive) | [`../../docs/oom-guardrailing.md`](../../docs/oom-guardrailing.md) |

---

## 1. Environment setup (uv)

Python 3.10, torch 2.5.0 on CUDA 12.4, in a uv venv. The repo ships a frozen pin list at
[`../requirements_uv.txt`](../requirements_uv.txt); the commands below are the ones that needed
hand-holding beyond a plain `uv pip install`.

```bash
# from the repo root (worldfm/)
uv venv --python 3.10 .venv
source .venv/bin/activate

# torch 2.5.0 +cu124 (pull from the pytorch index, not PyPI)
uv pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu124

# bulk of the app stack (diffusers 0.34, transformers 4.51, xformers 0.0.28.post2,
# triton 3.1.0, omegaconf, etc.) — frozen pins
uv pip install -r requirements_uv.txt
```

The tricky extras — each has one quirk:

```bash
# (a) mmcv 1.7.0 as PURE-PYTHON (skip the C++ custom-ops build, which is what bites everyone).
#     First pin an OLD setuptools (mmcv's setup.py breaks on setuptools>=80), then build in-place
#     with MMCV_WITH_OPS=0 so it never tries to compile CUDA extensions:
uv pip install 'setuptools<80'
MMCV_WITH_OPS=0 uv pip install mmcv==1.7.0 --no-build-isolation

# (b) bitsandbytes (NF4 quantization of FLUX transformer + T5 in Step 1)
uv pip install bitsandbytes

# (c) Real-ESRGAN (HunyuanWorld-1 super-res submodule) — editable, no build isolation
cd submodules/Real-ESRGAN
uv pip install basicsr-fixed facexlib gfpgan
uv pip install -e . --no-build-isolation
cd ../..

# (d) ZIM (HunyuanWorld-1 "anything" module) — editable
cd submodules/ZIM && uv pip install -e . && cd ../..

# (e) MoGe (depth from Step 2) — editable, pinned commit per setup.sh
cd submodules/MoGe
git checkout 7807b5de2bc0c1e80519f5f3d1f38a606f8f9925
uv pip install -e .
cd ../..
```

> `git submodule update --init --recursive` first if `submodules/` is empty.

---

## 2. The REQUIRED `LD_LIBRARY_PATH` (do not skip)

The NVIDIA runtime libs (cuDNN, nvrtc, cuBLAS…) are installed *inside the venv* by the
`nvidia-*-cu12` wheels, **not** in a system path. Without exporting them, cuDNN's JIT cannot find
`libnvrtc` / cuDNN at runtime and Step 1 (FLUX) blows up with:

```
No execution plans support the graph
```

Fix — export every `nvidia/*/lib` in the venv. This is the exact line the wrappers use:

```bash
NVIDIA_LIBS=$(find "$PWD/.venv/lib/python3.10/site-packages/nvidia" -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH}"
```

Put it in your activate hook or at the top of every run script. If you see
`No execution plans support the graph`, you forgot this — not a torch/cuDNN version bug.

---

## 3. Weights: HF gated model + WorldFM checkpoints

Two sets of weights, fetched separately.

**WorldFM student checkpoints + VAE** (public, repo `inspatio/worldfm`) — use the shipped helper,
which downloads `worldfm_1-step.pth`, `worldfm_2-step.pth`, and the VAE into `./weights/`:

```bash
python download_ckpts.py
```

**FLUX.1-Fill-dev** (Step 1 panorama generator) is a **gated** HuggingFace model. You must:

1. Request access at <https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev> and wait for
   approval.
2. Export a token with read scope: `export HF_TOKEN=hf_...` (or `huggingface-cli login`).

On this box we actually run the **NF4-quantized** variant (`diffusers/FLUX.1-Fill-dev-nf4`, see
[§4](#4-config-overrides-that-make-it-fit)) to fit 16 GB. That path is downloaded automatically on
first Step-1 run by `modules/panogen.py` — the `base_path` `black-forest-labs/FLUX.1-Fill-dev` is
the gated source whose license still governs use.

---

## 4. Config overrides that make it fit

The shipped [`../default.yaml`](../default.yaml) assumes a big GPU and OOMs on 16 GB. These three
overrides (already applied to this checkout — verified `path:line`) are what make the demo run:

| Stage | Override | Where | vs shipped |
|---|---|---|---|
| **panogen (Step 1)** | FLUX **NF4** (transformer + T5), canvas **512×1024**, `pipe.to("cuda")` (no CPU offload) | `modules/panogen.py:81` (`height,width=512,1024`), `:94` (`nf4_path`), `:122` (`self.pipe.to("cuda")`) | was 960×1920, full-precision FLUX |
| **MoGe (Step 2)** | `resolution_level=9`, `batch_size=1`, `merge_max=1024×512` | read in `run_pipeline.py:335` (`resolution_level`), `:338-339` (`merge_max_*`); defaults in `modules/moge_pano.py:23-24` | shipped `resolution_level=30` is **outside the valid 0–9 range** (~3.3× max tokens → OOM); `batch_size` was 4; `merge_max` was 4096×2048 (~11 GiB burst → cgroup OOM) |
| **worldfm (Step 4)** | render/infer @ 512×512, `cfg_scale=0.0` (shipped DMD path), `torch.compile(reduce-overhead)`, VAE tiling+slicing disabled | `run_pipeline.py` arg defaults / `modules/worldfm_infer.py` | unchanged from shipped inference path |

> The MoGe `resolution_level` correction (30 → 9) is the single highest-leverage fix: the shipped
> value is literally invalid and is the most common silent OOM source. If you re-clone, re-apply
> it.

All knobs are also CLI flags on `run_pipeline.py` (`--image_size`, `--step`, `--cfg_scale`,
`--moge_path`, `--render_size`, …) — see `python run_pipeline.py --help`.

---

## 5. The two-phase run

WorldFM is split into an **offline anchor-build** (Steps 1–3: panorama → MoGe depth/PLY →
condition DB) and the **online per-frame** synthesis (Step 4). Run them in two phases so the
expensive FLUX + MoGe work is done once and cached.

### Phase 1 — build & cache offline anchors

```bash
python run_pipeline.py --meta demo/meta.json --output_dir outputs --prepare_only
```

`--prepare_only` runs Steps 1–3 and exits before the WorldFM denoiser. It writes
`outputs/mario/panorama.png` and `outputs/mario/intermediates/` (PLY ≈ 8.4 M points, 42 rendered
conditions). **`panorama.png` caches**, so re-running Phase 1 skips the FLUX pass entirely; the
intermediates let Phase 2 skip all of Steps 1–3 via `--reuse_intermediates`.

### Phase 2 — online per-frame synthesis

```bash
# step 2 = 2-step DMD (default / best quality)
python run_pipeline.py --meta demo/meta.json --output_dir outputs \
    --step 2 --reuse_intermediates --profile_worldfm --save_mode image

# step 1 = 1-step DMD (fastest)
python run_pipeline.py --meta demo/meta.json --output_dir outputs \
    --step 1 --reuse_intermediates --profile_worldfm --save_mode image
```

- `--reuse_intermediates` — skip Steps 1–3, load the cached anchors (requires a successful Phase 1).
- `--profile_worldfm` — emit per-frame timing + peak VRAM to `outputs/mario/performance*.json`.
- `--save_mode image` — write PNGs (`step{1,2}_output_NNNN.png`) instead of just the mp4; drop it
  to get `step{1,2}.mp4`.

### Self-guarding wrappers (recommended on a 16 GB box)

Because a cgroup OOM kills the whole container with no `dmesg`, run via the two watchdog scripts
that poll `/proc/meminfo` and `kill -9` **their own** child PID (never a pattern) if available RAM
drops below `RAM_LIMIT` (default 2500 MiB):

```bash
bash scripts/prepare.sh   # Phase 1, logs -> logs/prepare.log
bash scripts/runs.sh      # Phase 2 step=2 then step=1, logs -> logs/runs.log
```

`runs.sh` additionally copies each step's `performance.json` → `performance_step{1,2}.json` and
renames `output_*.png` → `step{1,2}_output_*.png` before the next run clobbers them.

---

## 6. Expected results

Measured on the target box (RTX 4060 Ti 16 GB, torch 2.5.0+cu124, commit `d51ada2`, @512×512
per-frame WorldFM inference, steady-state = 3 `torch.compile` warmup frames excluded). Full data:
[`../outputs/mario/metrics.md`](../outputs/mario/metrics.md).

| run | steady-state FPS | avg inference (s/frame) | peak VRAM (MiB) |
|---|---|---|---|
| **step=2** (2-step DMD, default) | **2.63** | 0.380 | **4970** (~5 GB) |
| **step=1** (1-step DMD, fastest) | **3.90** | 0.256 | **4724** (~5 GB) |

vs. the paper `[§3 p12]`: ~10 FPS on an RTX 4090, ~25 FPS on an H-series. The 4060 Ti is ≈0.4× of a
4090, so ~4 FPS step1 here is the paper's ~10 FPS scaled down — consistent, not a regression.

**Outputs land in [`../outputs/mario/`](../outputs/mario/):**

| File | What |
|---|---|
| `panorama.png` | Step-1 generated panorama (caches Phase 1) |
| `step{1,2}_output_NNNN.png` | per-frame novel views (`--save_mode image`) |
| `step1.mp4`, `step2.mp4` | assembled videos (default `--save_mode video`) |
| `intermediates/` | PLY + 42 condition DB (enables `--reuse_intermediates`) |
| `performance*.json` | `--profile_worldfm` timing + VRAM |
| `metrics.json`, `metrics.md` | aggregated headline numbers |

---

## 7. Viewing on a headless VM

The box has no display. Pull artifacts out of the container, or serve them:

```bash
# Incus: copy a file/dir to the host (run from the HOST, not the container)
incus file pull s-gtksoon1/root/gtk-projects/world-model/worldfm/outputs/mario/step2.mp4 ./
incus file pull s-gtksoon1/root/gtk-projects/world-model/worldfm/outputs/mario/ . --recursive

# or, inside the container, serve the output dir over HTTP and fetch from the host browser
cd outputs/mario && python -m http.server 8000
```

---

## 8. Troubleshooting

For **any** out-of-memory symptom — cgroup RAM kill, CUDA OOM, the MoGe `resolution_level=30`
explosion, the `merge_max` 4096×2048 VRAM burst, or tuning `RAM_LIMIT` / NF4 / canvas size — see
the dedicated deep dive:

→ [`../../docs/oom-guardrailing.md`](../../docs/oom-guardrailing.md)

Two non-OOM gotchas are covered above and repeated here for searchability:

- **`No execution plans support the graph`** → missing `LD_LIBRARY_PATH`, see [§2](#2-the-required-ld_library_path-do-not-skip).
- **`mmcv` build fails / custom-ops compile error** → you skipped `MMCV_WITH_OPS=0` or
  `setuptools<80`, see [§1](#1-environment-setup-uv).
