# Code & config modifications made to run on 16 GB

> **Atomic scope:** This doc is *only* the change record for the uncommitted working-tree edits made to get the WorldFM pipeline running in a 16 GB cgroup. It is not a guide and not a general technique reference.
>
> **Status:** These are **uncommitted working-tree changes**. Review with `git diff` (and `git status`) before committing. Nothing below is on `main`.
>
> **Cross-links:**
> - [`oom-guardrailing.md`](../../docs/oom-guardrailing.md) — the general OOM/OOM-guardrailing techniques these changes instantiate.
> - [`run-guide.md`](./run-guide.md) — how to actually run the pipeline once these edits are applied.

## Change record

| File | What changed | Why (the OOM it fixed) |
|---|---|---|
| `modules/panogen.py` — `Image2PanoramaDemo.__init__` | Load the NF4 transformer and NF4 `text_encoder_2` from `diffusers/FLUX.1-Fill-dev-nf4` via `from_pretrained(subfolder=...)`, then pass them into `Image2PanoramaPipeline.from_pretrained(base_path, transformer=…, text_encoder_2=…)`. Replaces `from_pretrained('black-forest-labs/FLUX.1-Fill-dev', bf16)`. | The original bf16 load materialized the full model (~34 GB) → cgroup OOM at load time. NF4 lets the two bf16-heavy components be swapped in without ever materializing the bf16 originals. |
| `modules/panogen.py` — `Image2PanoramaDemo.__init__` | Removed `fuse_lora()` and `unload_lora_weights()`. LoRA is kept as a non-fused adapter. | bnb NF4 does not fuse LoRA cleanly. Fusing/unloading under NF4 is unreliable, so the adapter is left in place. |
| `modules/panogen.py` — `Image2PanoramaDemo.__init__` | `enable_model_cpu_offload()` → `pipe.to('cuda')`. | CPU offload pins ~9 GB of weights in host RAM, which on its own pushes the cgroup over budget → OOM. With NF4 the model fits on the GPU, so no host-side pinning is needed. |
| `modules/panogen.py` — `Image2PanoramaDemo.__init__` | Canvas resolution `self.height, self.width` changed from `960 × 1920` → `512 × 1024`. | Reduces generation-time RAM and VRAM footprint of the panorama canvas. |
| `default.yaml` (moge) | `resolution_level` `30` → `9`. | `30` is outside the valid `0–9` range, which blew `max_tokens` up by ~3.3×, driving the O(n²) attention VRAM budget into OOM. `9` is the max valid level. |
| `default.yaml` (moge) | `batch_size` `4` → `1`. | Lower per-step memory pressure for the MoGe stage. |
| `default.yaml` (moge) | `merge_max_width`/`merge_max_height` `4096 × 2048` → `1024 × 512`. | The merge was a single ~11 GB allocation → cgroup OOM that landed between monitor polls. Smaller merge cap keeps the peak under budget. |
| `run_pipeline.py` — `step1_panogen` | After panorama generation: `del demo; gc.collect(); torch.cuda.empty_cache()`. | Releases FLUX before MoGe loads. bnb pipelines are not reliably freed by `gc` alone, so an explicit `del` + `empty_cache()` is required to actually return VRAM. |
| `run_pipeline.py` — `main` | `main` now passes `save_intermediates=…` through to `step1_panogen` so `panorama.png` is cached. | Caching the panorama enables process isolation on retry (the FLUX stage can be skipped/replayed instead of re-run). |

## Reviewing these changes

```bash
git status            # uncommitted working-tree files
git diff modules/panogen.py
git diff default.yaml
git diff run_pipeline.py
```

For the general techniques behind each edit (NF4 swap-in vs. bf16 load, why CPU offload is hostile to tight cgroups, attention-budget math for out-of-range `resolution_level`, bnb free-on-`del`), see [`oom-guardrailing.md`](../../docs/oom-guardrailing.md). For how to run the pipeline with these edits applied, see [`run-guide.md`](./run-guide.md).
