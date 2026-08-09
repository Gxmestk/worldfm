# Handoff — WorldFM (read this first if picking up the project)

> **Project:** WorldFM / InSpatio-WorldFM (arXiv [2603.11911](https://arxiv.org/abs/2603.11911))
> **Repo:** `/root/gtk-projects/world-model/worldfm`
> **Host:** Incus container `s-gtksoon1` (16 GiB RAM cgroup, NO swap; RTX 4060 Ti 16 GB VRAM).
> **Last updated:** 2026-07-29.

## What's already done (don't redo)

- **Paper→docs study** — `worldfm/docs/learning/` (`01`–`10` + `worldfm.card.yaml` + `README`).
- **Demo runs end-to-end on this 16 GB box** — per-frame WorldFM @ 512² = **2.63 FPS (step2)**,
  **3.90 FPS (step1)**; peak VRAM ~5 GB. Cached anchors in `outputs/mario/intermediates/`.
- **Docs organized & deduped** —
  - Project: `worldfm/docs/` (`README`, `run-guide.md`, `repo-mods.md`, `live-loop.md` + `learning/`).
  - General AI: `../docs/` (`README`, `oom-guardrailing.md`, `quantization-reference.md`, `containerized-ai-environments.md`).
- **Interactive live-loop webserver — DONE (2026-07-29).** `live_server.py` + `live/viewer.html`
  wrap the per-frame API (step3/step4) into a WebSocket fly-through: reuse-only, step-1, ~3.7 FPS,
  peak ~4.8 GB VRAM. Launch `bash scripts/serve.sh` (**port 8123** — 8000 is the host's
  `kiro-gateway`). Recipe: `worldfm/docs/live-loop.md`.

## Read first

- **Your auto-loaded memory (`MEMORY.md`):** `uv-for-python-envs`, `ask-before-model-oom`,
  `worldfm-demo-run-config`, `container-host-transfer-constraints`, `paper-to-docs-reference-pattern`.
- `worldfm/docs/run-guide.md` — how to run the demo.
- `worldfm/docs/repo-mods.md` — the **uncommitted** working-tree code/config changes + the OOM each fixed.
- `../docs/oom-guardrailing.md` — the OOM playbook (self-guard, `memwarn.sh`, post-mortem, worked example).

## Hard rules (from the user — non-negotiable)

1. **ALWAYS use uv** (`uv venv` / `uv pip` / `uv run`) — never raw pip/conda/venv.
2. **Before loading ANY model that would exceed the 16 GB RAM cgroup / 16 GB VRAM: STOP and ASK the
   user**, and offer a quantized alternative. The VM OOM-crashed once.
3. The binding limit is the **16 GB memory cgroup (no swap)** — enforced by the kernel OOM-killer,
   *not* physical RAM or VRAM. Use the self-guarding wrapper (**kill by PID; NEVER `pkill -f`** — it
   matches the monitor's own cmdline and self-kills).
4. Keep all project files **in this repo** — run/dev scripts in **`scripts/`**, run logs in
   **`logs/`** (persistent) — **NOT `/tmp`** (cleared between sessions) or `/root/wfm`.
5. `export LD_LIBRARY_PATH` over `.venv/lib/python3.10/site-packages/nvidia/*/lib` or cuDNN/nvrtc JIT
   fails ("No execution plans support the graph").

## Interactive live-loop webserver — DONE (2026-07-29)

The previously-deferred next task is **complete and smoke-tested.** `live_server.py` +
`live/viewer.html` wrap the per-frame API (`step3_render_one` + `step4_infer_one` in
`run_pipeline.py`) into a WebSocket fly-through: reuse-only (no FLUX/MoGe), step-1, ~3.7 FPS,
peak ~4.8 GB VRAM. Pose convention verified (vs cached `step1_output_0000.png`, color-histCorrel 0.996).

```bash
bash scripts/serve.sh      # port 8123 (8000 is the host's kiro-gateway); host browser: http://s-gtksoon1.lan.bsthun.in:8123/
```

Recipe + troubleshooting: `worldfm/docs/live-loop.md`; internals (cache structure, arbitrary-pose
rendering, stochasticity): `worldfm/docs/live-loop-internals.md`. **Gotcha baked in:** `torch.compile`
(`reduce-overhead`) is thread-affine — preload *and* every inference call run on one worker thread,
or the compiled VAE/model forward asserts.

## Resume / run

```bash
cd /root/gtk-projects/world-model/worldfm && source .venv/bin/activate   # uv venv
bash scripts/prepare.sh   # builds+caches offline anchors (panorama.png cached → skips FLUX)
bash scripts/runs.sh      # step=2 then step=1, --profile_worldfm, self-guarded
bash scripts/serve.sh     # interactive live loop (WS fly-through), port 8123, self-guarded
```

```bash
bash scripts/serve-docs.sh   # mkdocs-material docs site, port 8126 (auto-rebuilds on docs/*.md edits)
```

> **Docs-site `uv run` gotcha (docs site ONLY):** `uv run mkdocs` / `uv run python -m mkdocs`
> **fail** in this container — `uv run` resolves to `/opt/venv/0/bin/python3`, which has no mkdocs
> (only the project's uv-managed `.venv` does, installed via `requirements-docs.txt`). Serve with
> `bash scripts/serve-docs.sh`, which calls `.venv/bin/python -m mkdocs serve --dev-addr 0.0.0.0:8126`
> directly. **Docs-site only** — `uv run python live_server.py` (`scripts/serve.sh`, port 8123)
> works fine; do not generalize the quirk to the live server.
