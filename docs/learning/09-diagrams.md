# 09 — Mechanism Diagrams (ASCII)

> Text diagrams of InSpatio-WorldFM's core mechanisms — render in any markdown/terminal
> and are fully agent-parseable. Companion to `05`–`08`; see those for the equations and
> code refs. Paper page refs `[pN]`/`[§x]` are to `paper/2603.11911v3.pdf`; code refs are
> `path:line` in this repo. Verified 2026-07-29.

## 1. System overview — two-phase offline/online, one tri-condition

```
          InSpatio-WorldFM  =  frame-based latent DiT  (per-frame NVS)
   ┌────────────────────────────────────────────────────────────────┐
   │ INPUTS: 1 ref image x_ref+π_ref   +   target pose(s) π_tgt     │
   └────────────────────────────────────────────────────────────────┘
      OFFLINE (high-compute, run once)        ONLINE (real-time, per π_tgt)
      image -> panorama (HunyuanWorld-1.0 *)    1. render point cloud -> cond1
            -> 42 views, MoGe-2 depth           2. pick nearest ref    -> cond2
            -> GLOBAL 3D POINT CLOUD            3. DMD 2-step denoise  -> x_tgt
      Challenge 2 — CONSISTENCY                Challenge 1 — INTERACTION
        HYBRID memory:                         pose π_tgt  (PRoPE †)
          explicit anchor (cond1)                -> no window, 1 frame / call
          + implicit reference (cond2)
      Challenge 4 — RUNTIME                    Challenge 3 — STABILITY
        DMD distil -> 2-step, t_mid=200         consistency via CONDITIONING
        ~25 FPS@512 (H-series), 10 FPS (4090)   (not temporal recurrence)
              └──►  tri-condition concat [z_t|cond1|cond2]  ◄──┘
      * internal pano model withheld; HunyuanWorld-1.0 substitute
      † PRoPE implemented but inactive on shipped inference (diagram 6)
```
Why it matters: the offline/online split is the whole shape of the system — heavy 3D-anchor
and reference providers are quarantined offline so the online path is one cheap frame model.
Code: offline build `run_pipeline.py:354-379`; per-target loop `run_pipeline.py:441,776`.

## 2. The tri-condition width-concat  `[z_t | cond1 | cond2]`  (condition assembly)  ⭐

```
 One self-attention DiT sequence (cond1 & cond2 are CLEAN σ=0; only target is noised):

  ┌────────────────┬──────────────────────┬──────────────────────┐
  │ noisy target   │ cond1  EXPLICIT      │ cond2  IMPLICIT      │
  │ z_t            │ anchor               │ memory               │
  │ α_t·z + σ_t·ε  │ point-cloud render   │ nearest ref frame    │
  │ (DENOISED)     │ x̂_tgt at π_tgt       │ x_ref (appearance)   │
  └────────────────┴──────────────────────┴──────────────────────┘
   shared patch-embed + sinusoidal pos-embed;  WIDTH-concatenated (dim=3)
   └──────── full self-attention over ALL tokens (cross-attn OFF) ────────┘
                              │
        after last block: SPLIT along width, retain ONLY the target slice
                              │
                   denoise ONLY z_t   ->   x_tgt
```
Why it matters: conditions enter via **self-attention** (concatenated tokens), *not*
cross-attention — empirically higher quality `[§2.4.1 p4]`. Conditions attend to each other
*and* the target in one pass, then only the target slice is kept. Code: concat at
`PixArtWorldFMMS.py:449`; split-and-retain at `PixArtWorldFMMS.py:707-713`; cross-attn off at
`worldfm_infer.py:97` (the cond2-cross-attn branch at `PixArtWorldFMMS.py:382-428` is disabled:
`use_cond2_cross_attn=False` at `worldfm_infer.py:452`).

## 3. Multi-view generation flow (per-frame roll-out)

```
   for each target pose  i = 1 .. N:              (NO window; one frame per call)
   ┌───────────────────────────────────────────────────────────────────────┐
   │ 1. render global point cloud into π_tgt_i  ->  x̂_tgt_i   (cond1)      │
   │ 2. select nearest condition view           ->  x_ref_i    (cond2)      │
   │      (depth-consistency + view-angle + distance; reuse latent by idx) │
   │ 3. VAE-encode cond1;  reuse cached cond2 latent z_c2                  │
   │ 4. sample x_tgt_i ~ pθ(· | z_t, cond1, cond2)       # 2 DMD steps     │
   │ 5. VAE-decode -> pixels;  write per-frame PNG / append to MP4         │
   │ 6. i <- i+1                  (NO history carry-over between frames)   │
   └───────────────────────────────────────────────────────────────────────┘
   independent per-frame  =>  constant compute/frame  =>  no error drift
   BUT no temporal constraint between frames  =>  frame jitter [§4.1 p13]
```
Why it matters: this is the "frame model" — there is **no multi-frame/window processing in the
model**; consistency is recovered purely through conditioning, never temporal recurrence
`[§2.4.1 p4]`. Code: per-target render+select `run_pipeline.py:441-471`; the 1-/2-step
denoising call at `worldfm_infer.py:386-494`.

## 4. Explicit 3D anchor pipeline — point-cloud rendering (cond1)

```
   OFFLINE ANCHOR BUILD   (run once per scene)
   image ─► panorama (HunyuanWorld-1.0 *)              360° appearance
            │  split into 42 Fibonacci-sphere views, fov 45°
            ▼
   MoGe-2-vitl depth per view                          feedforward metric depth
            │  merge panorama depth;  depth /100
            ▼
   GLOBAL POINT CLOUD  {xyz, rgb}                      persists across all target poses
   ────────────────────────────────────────────────────────────────────────────
   PER-TARGET RENDER   (online, each π_tgt)
   project + splat global cloud into target camera π_tgt
            │  z-buffer occlusion;  pure-PyTorch splatting (no EGL/OpenGL)
            ▼
   x̂_tgt  =  point-cloud rendering  (cond1)            coarse geometric prior
   * internal pano generative model withheld; HunyuanWorld-1.0 is the substitute
```
Why it matters: the anchor is the global-geometry half of hybrid memory — coarse but view-
invariant `[§2.4.1 p6]`, `[§2.2 p4]` (`x̂_tgt`). Its quality is bounded by feedforward-depth
error, which is exactly what the Unreal-Engine synthetic finetune corrects `[§2.4.2 p7]`.
Code: pano/MoGe constants `moge_pano.py:20-24`, cameras `:57-66`; depth build `run_pipeline.py:
354-379`; splatting render `point_renderer.py:89-191`; depth→PLY `pano_postprocess.py:123-149`.

## 5. Hybrid spatial memory — explicit anchor + implicit reference

```
                       ┌──── coarse global geometry (3D prior) ─────┐
   EXPLICIT anchor ──► │ point-cloud render x̂_tgt at target view    │──┐
   (cond1)             │ holds structure; bounded by recon error    │  │
                       └────────────────────────────────────────────┘  │
                                                                     ├──► self-attn
                       ┌──── fine appearance + hallucination ────────┐ │
   IMPLICIT memory ──► │ nearest reference frame x_ref (cond2)       │─┘
   (cond2)             │ retrieves / transfers visual content;       │
                       │ plausible fill in UNOBSERVED regions       │
                       └────────────────────────────────────────────┘
        └──► both feed the ONE self-attn stream:  [z_t | cond1 | cond2]
   Complementary: explicit anchors coarse geometry; implicit preserves
   fine appearance and hallucinates where the cloud has no points.
   Lineage: RTFM (posed-frame memory) · StarGen (keyframe warp) ·
            VGGT/DUSt3R/MoGe/MapAnything (feedforward 3D recon)
```
Why it matters: this is the consistency mechanism that makes independent frames cohere
`[§2.4.1 p6]`. cond2 is selected as the **nearest condition view** (depth-consistency +
view-angle + distance weighting). Code: `depth_selector.py:169-319` (`select_best_condition_index`),
condition DB `depth_selector.py:128`; cond2 indexing + latent cache `worldfm_infer.py:410-438`.

## 6. Camera / pose conditioning — three strategies, one adopted

```
   Three camera-encoding strategies [§2.4.1 p5-6]  —  inject π into the DiT:

   ┌──────────────────┬────────────────────────────┬───────────────────────┐
   │ strategy         │ how it encodes the camera  │ status                │
   ├──────────────────┼────────────────────────────┼───────────────────────┤
   │ PRoPE [20]       │ P_iᵀ on Q, P_i⁻¹ on K/V    │ ✅ adopted   ◄ best   │
   │ (Cameras-as-RPE) │ + 2D RoPE; modulates attn  │ ⚠️ inactive at infer  │
   ├──────────────────┼────────────────────────────┼───────────────────────┤
   │ Plücker ray      │ 6-D (o×d, d) per patch,     │ compared alternative  │
   │ [15,1,2]         │ MLP -> ADD to patch embed   │ ⚠️ inactive at infer  │
   ├──────────────────┼────────────────────────────┼───────────────────────┤
   │ pure-parametric  │ MLP on raw R/T -> ADD to    │ paper-only (no code)  │
   │ [3]              │ hidden rep                 │                       │
   └──────────────────┴────────────────────────────┴───────────────────────┘
   PRoPE chosen: fastest convergence, most stable control (modulates attention, vs
   Plücker's additive encoding). ⚠️ BOTH implemented branches are NOT activated on the
   shipped inference path (worldfm_infer.py never passes camera matrices / use_prope /
   use_plucker) — the released 1-/2-step checkpoints run with plain self-attention.
```
Why it matters: camera conditioning is the control channel that turns an image DiT into a
novel-view generator. Code (implemented): PRoPE `prope.py:52-140`, apply in
`PixArtWorldFM_blocks.py:158-231`, precompute `PixArtWorldFMMS.py:521-634`, gated default-off
`PixArtWorldFMMS.py:85-89,488`; Plücker `plucker.py:8-72`, injection `PixArtWorldFMMS.py:639-666`,
`use_plucker` default False `PixArtWorldFMMS.py:480`. See `04-paper-code-crosswalk.md` for the
⚠️ discrepancy and [`worldfm.card.yaml`](worldfm.card.yaml) `flags_to_verify`.

## 7. Three-stage progressive training graph

```
   PixArt-Σ DiT  (XL_2: d28/h1152/patch2/heads16)   ── backbone SELECTION
        │
        ▼
   STAGE I · Pre-Training                         [high-fidelity image prior]
      select an efficient text-to-image DiT   (NO new pretraining stated)
        │
        ▼
   STAGE II · Middle-Training                     [controllable frame model + memory]
      2a  Fundamental frame model on REAL data:
          tri-condition concat + PRoPE + hybrid spatial memory
          ε-prediction L2 loss;  16 frames/clip -> 4 ref (point cloud) + 12 targets
          strategies: noise-schedule biasing + progressive condition
          injection (implicit memory FIRST, anchor later) + random masking
      2b  Synthetic UE finetune:  precise GT pose/depth trajectories
          (collision avoidance; stochastic/template motion; LIMITED steps)
        │
        ▼
   STAGE III · Post-Training                    [multi-step teacher -> 2-step student]
      DMD: frozen real score  +  dynamically-updated fake score
           minimize approx-KL (denoising-prediction difference) + regression loss
      findings: 2-step > 1-step;   t_mid=200 (on a 1000-step schedule) is optimal
        │
        ▼
   2-step real-time generator   (full condition + memory stack retained)
   ⚠️  ALL three stages are PAPER-ONLY — the released repo is INFERENCE-ONLY
       (no train.py/dataset/optimizer; training_losses boilerplate is unwired)
```
Why it matters: the progressive curriculum (bias high-noise t, inject conditions gradually,
mask the anchor, then UE-finetune) is what stabilizes consistency that temporal recurrence
would otherwise enforce `[§2.4.1 p6-7]`, `[§2.4.2 p7]`. None of it is reproducible from the
release — only the already-distilled 1-/2-step inference ships. Deep dive:
`06-training-distillation.md`.

## 8. Runtime & distillation snapshot — 2-step DMD, qualitative-only eval

```
   DMD distillation (TRAINING, paper-only) — 2 scores, ~3 models [§2.5 p7]
   ┌──────────────────────────────────────────────────────────────────┐
   │ frozen base model   ──► s_real  (teacher score)                  │
   │ updated-on-fakes    ──► s_fake  (critic score)                   │
   │ student generator   ──► few-step ẑ   (+ regression on (ε,x) pairs)│
   │   ∇θ ∝ (s_real − s_fake) · ∂ẑ/∂θ      # approx-KL (extends VSD)  │
   └──────────────────────────────────────────────────────────────────┘
   ────────────────────────────────────────────────────────────────────
   SHIPPED INFERENCE (in-code): 2-step schedule, t_mid=200  [§3 p12]
      step 1:  z_999  ──predict ε──►  x̂_0  ──re-noise──►  z_200   (coarse structure)
      step 2:  z_200  ──predict ε──►  x̂_0                       (refine fine detail)
      1-step omits step 2;  2-step > 1-step (dedicated refinement pass)
   ────────────────────────────────────────────────────────────────────
   runtime @ 512×512  (KV-cache + VAE latent caching, torch.compile):
      ~25 FPS   H-series GPU   ◄ best           10 FPS   RTX 4090
   ⚠️  QUALITATIVE-ONLY eval — NO FID/PSNR/LPIPS/multi-view-consistency metrics;
       "minimal perceptual difference" after distillation is NOT numerically shown
```
Why it matters: real-time speed is bought by 2-step DMD distillation plus heavy engineering
(`torch.compile` reduce-overhead, cond2 + VAE latent caching), not by model size `[§3 p12]`.
The "2-step > 1-step" and "t_mid=200" findings are the paper's headline post-training result.
Code: 1-step `worldfm_infer.py:455-463`; 2-step `:464-484`; `mid_t=200` `:99`; `ts_999`/mid
coeffs `:217-222`; multi-step DPM-Solver++ teacher `:497-567`; torch.compile `:164-211`;
cond2 latent cache `:356-381`. Defaults: `default.yaml:50` step=2, `:55` cfg_scale=4.5
(4.5 applies only to the multi-step teacher; the shipped 2-step path runs cfg_scale=0.0 at
`worldfm_infer.py:100`). Internals: `08-runtime-and-decode.md`.

*Equations and full code walkthroughs live in [`05-formulation.md`](05-formulation.md) and
[`06-training-distillation.md`](06-training-distillation.md); the mechanism-by-mechanism
status table is [`04-paper-code-crosswalk.md`](04-paper-code-crosswalk.md); runtime internals
in [`08-runtime-and-decode.md`](08-runtime-and-decode.md); the machine-readable card at
[`worldfm.card.yaml`](worldfm.card.yaml).*
