# 08 — Runtime & Decode Internals

> Grounded in the paper's runtime claim [§3, p12] and future-work levers [§4.2, p14], the shipped
> stage sequence + `Profiler` in `run_pipeline.py`, and the decode internals in
> `modules/worldfm_infer.py` — including the `torch.compile` + condition-cache optimization
> commit `d51ada2` (2026-05-06). Page refs [pN] are to the single paper `2603.11911v3`; code refs
> are `path:line`, every one verified against this repo. Verified 2026-07-29.
>
> ⚠️ **No shipped internal docs.** Unlike AlayaWorld (which ships a measured `record.md` latency
> profile and a `streaming_decoder_plan.md`), this repo ships **no** `record.md`, planning, or
> decode-design markdown — only `README.md` and this `docs/learning/` set. So "measured latency"
> below is the **paper's claim**, and the only in-repo measurement surface is the optional
> `Profiler` (which ships with **no recorded `performance.json`**). Treat all FPS numbers as
> paper-claimed, not reproduced here.

## The real-time framing — what the paper claims `[§3, p12]`

The frame-based paradigm makes real-time a **structural** property, not just an engineering win:
because each target view is generated independently, there is no window to wait on. The paper then
buys the actual throughput with two levers — few-step distillation (2-step DMD, see
[`06-training-distillation.md`](06-training-distillation.md)) and engineering optimizations:

- **~25 FPS** at 512×512 on **a single H-series GPU**, via "KV-cache management and efficient VAE
  latent caching" [§3, p12].
- **~10 FPS** on an **RTX 4090**, enabled by "its low GPU memory footprint" [§3, p12].
- "perceptual difference remains minimal in practice" after distillation — asserted
  **qualitatively**, not numerically substantiated [§3, p12–13] (no FID/PSNR/LPIPS).

> ⚠️ Paper↔code note: "H-series GPU" is **not pinned to a SKU** (e.g. H100). Treat the FPS as
> approximate and hardware-unspecified. There are no quantitative metrics anywhere in the paper
> [§3, p12–13]. — verify against your hardware.

### ✅ MEASURED on an RTX 4060 Ti (16 GB) — 2026-07-29, repo commit `d51ada2`

> The first recorded `performance.json` for this repo (it ships none). Produced via
> `--profile_worldfm`; full machine-readable data in `outputs/mario/metrics.{json,md}`.
> Steady-state excludes 3 `torch.compile` warmup frames.

| DMD step | steady-state inference FPS | end-to-end FPS | avg inference (s/frame) | peak VRAM |
|---|---|---|---|---|
| **step=2** (2-step, default) | **2.63** | 2.56 | 0.380 | 4.97 GB |
| **step=1** (1-step, fastest) | **3.90** | 3.75 | 0.256 | 4.72 GB |

- All at 512×512, `cfg_scale=0.0` (shipped DMD path), `torch.compile` (reduce-overhead).
- The 4060 Ti is ≈0.4× of an RTX 4090, so ~4 FPS (step=1) is consistent with the paper's ~10 FPS on a 4090 scaled down — **corroborates, does not contradict**, the real-time claim.
- **Caveat:** to fit the 16 GB box the panorama base was run **NF4** at a **512×1024** canvas and MoGe at `resolution_level=9` / merge `1024×512`. These affect 3D-anchor/visual quality, **not** the WorldFM per-frame FPS (which runs at `render_size=512` regardless).
- The slow, memory-heavy part is the **offline anchor build** (panorama→MoGe→point-cloud, minutes); only the **online per-frame** stage above is the "real-time" one.

## The two-phase pipeline in `run_pipeline.py` — the stage sequence

This is the real-time inference pipeline. It is explicitly **two-phase**: an *offline* anchor build
runs once per scene, then the *online* frame model runs once **per target pose** [§2.1, p3].

| Stage | Role | Where | Phase |
|---|---|---|---|
| `setup_external_repos` | Bind MoGe + HunyuanWorld submodules | `run_pipeline.py:253` | setup |
| `step1_panogen` | Image → 360° panorama (HunyuanWorld-1.0, open-source substitute) | `run_pipeline.py:280` | **offline** |
| `step2_moge_pipeline` | Panorama → depth → global PLY + condition views (MoGe-2, 42 views) | `run_pipeline.py:320` | **offline** |
| `step3_init` | Build point renderer + condition DB **once** | `run_pipeline.py:406` | online (once) |
| `step4_init` | Load distilled denoiser + VAE, `torch.compile`, **cache cond2 latents once** | `run_pipeline.py:482` | online (once) |
| `step3_render_one` | Render point cloud to target view (cond1) + pick nearest reference (cond2) | `run_pipeline.py:431` | online (per pose) |
| `step4_infer_one` | 2-step DMD decode of one frame | `run_pipeline.py:534` | online (per pose) |

The per-frame hot loop is `run_pipeline.py:780-819`: for each `c2w` pose it calls
`step3_render_one` → `step4_infer_one`, records a profiler frame, and either writes a per-frame PNG
(`--save_mode image`) or accumulates frames for an MP4 (`--save_mode video`, default 30 fps,
`run_pipeline.py:652-656,813-830`). **Chunk granularity = 1 frame** — the minimal frame-based unit.

> ⚠️ Paper↔code note: the **offline** anchor providers (panorama + MoGe depth) are high-compute and
> memory-heavy [§4.1, p13]; they are **not** real-time. The real-time claim covers only the online
> `step4_infer_one` path. The internal panorama model is withheld; the shipped substitute is
> HunyuanWorld-1.0 (`README.md:52-53`).

## Per-frame decode internals `[worldfm_infer.py:386-494]`

The single-frame hot path is `infer_from_render_u8`. Each call does, in order (with profiler hooks):

1. **cond1 prep + VAE encode** — the point-cloud render at the target view is VAE-encoded *per
   frame* (`z_c1`, `worldfm_infer.py:425`; `cond1_pre_ms`/`cond1_vae_ms` :405-428).
2. **cond2 latent reuse** — the nearest reference frame's latent is fetched **by index** from the
   pre-encoded cache (`z_c2`, :414-417); `cond2_vae_ms` fires only on the very first frame
   (:430-438). This is the "efficient VAE latent caching."
3. **DMD 2-step denoise** (default `step=2`, `default.yaml:50`):
   - **Step 1** at `t=999`: predict `eps1`, form `pred_x0` (:467-469).
   - **Re-noise** to the intermediate timestep: `noisy = a_mid·pred_x0 + s_mid·noise` (:474).
   - **Step 2** at `t_mid=200`: refine `eps2`, form the final `samples` (:478-480).
   - Coefficients are **precomputed** once: `mid_t=200` (:99), `ts_999` (:217), `a/s_999`
     (:219-220), `a/s_mid` (:221-222) on the linear 1000-step IDDPM schedule.
4. **VAE decode** of `samples` to RGB (:488; `vae_decode_ms` :487-491).

A `step==1` branch (:455-463) does single-shot x₀ prediction from pure noise (checkpoint
`weights/worldfm_1-step.pth`); the paper finds **2-step > 1-step** because the second pass is a
dedicated refinement from a cleaner state [§2.5, p7]. When `step∉{1,2}`, control falls through to
the **multi-step DPM-Solver++ teacher** sampler (`infer_from_render_u8_multistep` :497-567,
order=2, time_uniform, multistep, `cfg_scale=4.5` at :502) — the slower Stage-II path the released
repo can also run.

> ⚠️ Paper↔code note (cfg_scale): `default.yaml:55` sets `worldfm.cfg_scale: 4.5`, but the runtime
> `WorldFMInprocessConfig.cfg_scale` is **`0.0`** (`worldfm_infer.py:100`) for the shipped 1-/2-step
> DMD path. The 4.5 applies **only** to the multi-step teacher (:497-567). So the default `step=2`
> run executes with **no classifier-free guidance**, not 4.5. — verify intent.

## Why there is no streaming-decode constraint (the key contrast)

AlayaWorld's whole runtime doc is about a non-causal **video** VAE whose ~30-latent temporal
receptive field forces lossy chunked decode. **WorldFM has no such constraint, and that is the
point of the frame paradigm.** The shipped VAE is an image `AutoencoderKL` — a 2-D net — so each
frame's latent decodes **independently** with zero temporal receptive field: no future latents
needed, no chunk boundary, no overlap/tiling fusion, no lookahead lower bound. The decode is a pure
**throughput** cost, not a streaming-correctness cost. Real-time is therefore structurally
achievable, and the engineering only needs to make each isolated encode/decode cheap:

- **Deterministic mode** (`vae_deterministic=True`): uses `latent_dist.mode()`, no sampling noise
  (`worldfm_infer.py:257-265`).
- **channels-last** memory layout (`vae_channels_last=True`, :183-184, `_vae_input` :251-255).
- **Slicing/tiling disabled** for speed (`disable_vae_slicing/tiling`, :179-182;
  `default.yaml:62-63`).
- **`vae_scale = 0.13025`** (:214); `torch.compile` on the encode/decode wrappers (:191-211).

There is also **no text-decode overhead**: caption cross-attention is off
(`disable_cross_attn=True`, :97) and caption embeddings are zeroed (:233-241) — the model is
pose/image-conditioned only at inference.

## Engineering optimizations — `torch.compile` + condition/VAE latent caching `[commit d51ada2]`

This commit (2026-05-06) is what implements the paper's "KV-cache management and efficient VAE
latent caching" [§3, p12] behind the FPS claims:

| Optimization | What it does | Where |
|---|---|---|
| `torch.compile` (denoiser) | Compile `forward_with_dpmsolver`, `reduce-overhead`, `fullgraph=False`, `dynamic=False` | `worldfm_infer.py:164-175` |
| `torch.compile` (VAE) | Compile the encode/decode wrappers (same mode) | `worldfm_infer.py:191-211` |
| cond2 latent cache | VAE-encode all condition views **once**, reuse by index | `set_cond2_candidates_from_arrays` :356-381; reuse :414-417 |
| cond2 reuse on first frame | `cond2_vae_ms` fires once, then cached | `worldfm_infer.py:430-438` |
| Profiler event | Timed as `condition_latent_cache` | `run_pipeline.py:768-774` |

Defaults: `compile_model: true`, `compile_vae: true`, `compile_mode: reduce-overhead`
(`default.yaml:56-57,60-61`). Both compile paths fail **soft** — on any exception they log and fall
back to eager (:173-175, :207-211).

> ⚠️ Paper↔code note ("KV-cache"): the **VAE latent caching** is clearly implemented above, but an
> explicit **transformer attention KV-cache** is **not evident** in the shipped inference path —
> `forward_with_dpmsolver` runs fresh each step with no cross-step KV reuse. The paper's "KV-cache
> management" most plausibly refers to the condition/VAE latent caching (and `torch.compile`'s
> CUDA-graph buffer reuse under `reduce-overhead`), not an attention KV-cache. — verify.

## The performance profiler `[run_pipeline.py:102-207]`

Because no `record.md` ships, the only in-repo measurement surface is the optional `Profiler`,
enabled with `--profile_worldfm` and writing `performance.json`:

- **Per-frame record** (`record_frame` :127-168): `render_select_sec`, `inference_sec`,
  `frame_total_sec`, the selected condition index/hits/samples, and a `worldfm_ms` breakdown
  (`cond1_pre_ms`, `cond1_vae_ms`, `cond2_vae_ms`, `dmd_step1_ms`, `dmd_step2_ms`, `vae_decode_ms`,
  `total_ms`) sourced from `svc._last_profile` (`worldfm_infer.py:405-493`).
- **Summary** (`summary` :182-191): `all_frames` vs `steady_state` (skips `--perf_warmup_frames`),
  reporting `inference_fps` and `end_to_end_fps` (`_safe_rate` :91, :178-179).
- **Events**: staged timings like `condition_latent_cache` (:770-773).

⚠️ **No recorded `performance.json` is shipped** and **no measured numbers are published** — the
~25 / 10 FPS are paper-claimed. To get real numbers you must run `--profile_worldfm` on your own
H-series / 4090 hardware; the profiler is the harness for that, not a source of truth.

## Planned acceleration levers `[§4.2, p14]` — paper-only

The paper names future runtime improvements; **none are implemented** in the shipped inference path.
All are 📝 paper-only / future:

| Lever | Idea | Status |
|---|---|---|
| Linear attention | Replace dense self-attention to cut denoiser cost | 📝 paper-only [§4.2] |
| Efficient caching mechanisms | Further latent/KV-style caching | 📝 (partially realized via cond2 cache above) |
| VAE optimizations | Cheaper encode/decode | 📝 (channels_last + compile realized; more TBD) |
| Gaussian Splatting anchors | GS primitives as 3D anchors for fidelity/reflection | 📝 paper-only [§4.2] |

## How this maps to the paper's Runtime claim `[§3]`

- **Visual latency** → the 2-step DMD student (`dmd_step1_ms` + `dmd_step2_ms`) makes each
  `forward_with_dpmsolver` call cheap [§2.5, p7]; see [`06-training-distillation.md`](06-training-distillation.md).
- **Interaction latency** → per-frame: no window/chunk wait, by construction of the frame model
  [§2.4.1, p4]. One denoising call per target pose.
- **Engineering** → `torch.compile` (denoiser + VAE) + cond2/VAE latent caching underpin the
  FPS claims [§3, p12] — not separately quantified in the paper.
- **Chunk granularity** → **1 frame** (the minimal frame-based unit).

## Memory tie-in

The cond2 latent cache and per-frame VAE encode/decode are the runtime face of the tri-condition
design: cond1 (explicit 3D anchor) and cond2 (implicit reference memory) both enter as VAE latents
width-concatenated to `z_t` (see [`05-formulation.md`](05-formulation.md) and the comparison axes in
[`02-comparison-framework.md`](02-comparison-framework.md)). The unresolved frontier is not decode
correctness (the image VAE has no streaming problem) but raw throughput — which is exactly what the
planned linear-attention / GS-anchor levers [§4.2, p14] target.
