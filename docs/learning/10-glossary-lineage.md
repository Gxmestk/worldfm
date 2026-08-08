# 10 — Glossary & Lineage

> The techniques InSpatio-WorldFM builds on, the baselines it is measured against, and the
> peer methods its comparison roster references. Each entry: **identity · key idea · → which
> InSpatio-WorldFM component**. Characterizations follow how the paper frames each
> (`paper/2603.11911v3.pdf`, §1 taxonomy + §2.4.1 memory/camera discussions + §3); read the
> originals for full detail. Bracketed numbers `[N]` are the paper's own citation numbers;
> arXiv IDs are drawn from its reference list (pp.14–17). Code refs are `path:line` in this
> inference-only repo.
> Verified 2026-07-29.

## A. Direct lineage — what InSpatio-WorldFM explicitly inherits

| Ancestor | Identity | Key idea (as WorldFM frames it) | → WorldFM component |
|---|---|---|---|
| **PixArt-Σ** | [8], ECCV'24 | Efficient DiT for text-to-image at 4K; "weak-to-strong" training; high quality at low compute — chosen for the quality/throughput balance needed for real-time deploy | **DiT backbone** (`worldfm/diffusion/model/nets/PixArtWorldFMMS.py`, XL_2 config depth 28) |
| **DiT (Scalable Diffusion Models w/ Transformers)** | [24], ICCV'23 | Patchify + transformer denoiser replaces U-Net; AdaLN-zero conditioning | **DiT block template** (`PixArtWorldFM_blocks.py:86-194` KV-compress attention) |
| **Latent Diffusion / Stable Diffusion (LDM)** | [27], CVPR'22 | Diffuse in a VAE latent space `z=E(x)`; ε-prediction loss `L=E‖ε−εθ(z_t,t,C)‖²` | **Latent framework + VAE E/D** (`modules/worldfm_infer.py:178-271`, `vae_scale 0.13025`) |
| **DMD (Distribution Matching Distillation)** | [42], CVPR'24 | Few-step student matches a multi-step teacher's output distribution via a frozen *real* score + a dynamic *fake* score; approximate-KL gradient + regression loss | **Stage III distillation** → the released 1-/2-step generator (`worldfm_infer.py:455-484`) |
| **Variational Score Distillation (VSD)** | named in text [§2.5], not in ref list | Score-distillation in distribution (no per-sample correspondence); DMD "extends" it | Conceptual parent of the **DMD objective** (training is paper-only) |
| **PRoPE / Cameras-as-RPE** | [20], arXiv:2507.10496, 2025 | Apply `P_iᵀ` to queries and `P_i⁻¹` to keys/values per view + 2D RoPE, so attention natively reasons over cross-view geometry | **Adopted camera-pose encoding** — ⚠️ implemented (`prope.py:52-140`) but **NOT activated** on shipped inference |
| **Plücker-ray camera embedding** | [15,1,2] (CameraCtrl-II ICCV'25; Cosmos / Cosmos-Transfer1, arXiv:2501.03575 / 2503.14492) | 6-D `(o×d, d)` per-patch ray coords, MLP→hidden, **added** to patch embeddings (additive, not attention-modulating) | **Compared alternative** — ⚠️ implemented (`plucker.py:8-72`) but inactive (`use_plucker` never set) |
| **Pure-parametric camera injection** | [3] (ReCamMaster, ICCV'25) | MLP on raw R/T parameters, added to hidden state; no explicit ray/projection structure | **3rd compared strategy** — paper-only (no code in repo) |
| **RTFM (Real-Time Frame Model)** | [37], World Labs blog, 2025 | Closed real-time **frame** model using **posed frames as primitive spatial memory** — the frame-paradigm foil WorldFM open-sources against | **Frame paradigm** + the memory foil (`§2.4.1 p.6`); WorldFM substitutes hybrid explicit+implicit memory |
| **StarGen** | [45], CVPR'25 | **Warps features extracted from posed keyframes** to condition a video-diffusion scene generator (spatiotemporal AR) | **Memory foil** — keyframe feature-warping; WorldFM uses point-cloud render + reference self-attention instead |
| **MoGe** | [35], 2024 | Monocular **metric** geometry (depth) estimation for open-domain images | **Offline panorama depth → point cloud** (shipped: `moge_pano.py`, `run_pipeline.py:354-379`, MoGe-2-vitl-normal) |
| **MapAnything** | [17], arXiv:2509.13414, 2025 | Universal feed-forward **metric 3D reconstruction** (per-frame pose + depth) | **Training-data pose/depth recovery** [§2.4.1 p.6] (paper-only; no dataloader shipped) |
| **VGGT / DUSt3R / MegaDepth** | [34] CVPR'25 / [36] CVPR'24 / [21] CVPR'18 | Feed-forward geometry: grounded transformer / stereo-style 3D / SfM-trained monocular depth | Cited **3D-foundation-model** alternatives behind the explicit anchor `x̂_tgt` [§2.2 p.4] |
| **Stable Virtual Camera / Cat3D** | [49] ICCV'25 / [14] arXiv:2405.10314, 2024 | **Multi-view diffusion** view synthesis / multi-view-to-3D creation — offline anchor/reference providers | Cited **offline multi-view-consistent** providers [§2.1 p.3] (not shipped; internal model withheld) |
| **HunyuanWorld-1.0** | [31], arXiv:2507.21809, 2025 | Immersive explorable 3D worlds from words/pixels; FLUX-backboned | **Shipped open-source panorama substitute** (`modules/panogen.py:71-199`) — replaces the withheld internal panorama model |

## B. Benchmarks & evaluation baselines

> ⚠️ **Paper↔code note:** §3 evaluation is **qualitative only** [§3 p.12–13, Figs.4–8] — there is
> **no benchmark suite, no quantitative table, no FID/PSNR/LPIPS/multi-view-consistency metrics**.
> The "baselines" below are framing foils and internal ablations, not head-to-head numbers. See
> `flags_to_verify` in [`worldfm.card.yaml`](worldfm.card.yaml).

| Name | Identity | Role |
|---|---|---|
| **(No external benchmark suite)** | n/a — §3 is qualitative Figs.4–8 only | The paper claims "strong multi-view consistency" and "minimal perceptual difference" post-distillation **without numerically substantiating** them |
| **RTFM** | [37], World Labs, 2025 (closed) | The **frame-model paradigm foil** — motivates WorldFM's open-source real-time frame model; no quantitative comparison |
| **StarGen** | [45], CVPR'25 | **Memory-design foil** (posed-frame primitive vs keyframe feature-warping vs WorldFM's hybrid); qualitative framing only |
| **Stage-II teacher (own, multi-step)** | this paper, §2.4 | **Internal ablation** vs the Stage-III distilled variant — used to argue "minimal perceptual difference" [§3 p.13] |
| **Camera-pose encoding ablation** | Plücker [15,1,2] vs **PRoPE [20]** vs pure-parametric [3] | **Internal ablation** [§2.4.1 p.5–6] → **PRoPE chosen** ("fastest convergence, most stable control"); ⚠️ note PRoPE is inactive on shipped inference (see `04-paper-code-crosswalk.md`) |

## C. Peer techniques referenced (comparison roster, grouped by framing axis)

**Camera / interaction (pose encoding & control):** **PRoPE / Cameras-as-RPE [20]** (adopted) ·
Plücker-ray [15,1,2] (CameraCtrl-II / Cosmos / Cosmos-Transfer1 — additive ray embedding,
compared) · pure-parametric / ReCamMaster [3] (MLP-on-R/T, compared) — the three strategies
ablated in §2.4.1. Genie 3 [10] (rule/constraint-conditioned world model).

**Memory / consistency (spatial memory & NVS providers):** RTFM [37] (posed-frame primitive
memory) · StarGen [45] (keyframe feature-warping) · GEN3C [26] (3D-informed world-consistent
video) — the three named memory designs WorldFM positions its **hybrid explicit-anchor +
implicit-reference** scheme against. Feed-forward reconstruction for the anchor: MapAnything
[17], VGGT [34], DUSt3R [36], MoGe [35], MegaDepth [21]. Offline multi-view/anchor providers:
Stable Virtual Camera [49], Cat3D [14]. Panorama providers (offline 360° appearance/geometry):
HunyuanWorld-1.0 [31], Diffusion360 [13], PanFusion [46]; image-to-3D-scene peers Layer-Pano3D
[39], WonderWorld [43], WonderJourney [44], LucidDreamer [9], SceneX [50].

**Generation paradigm (video-based world models — the rejected side):** Voyager [16], WorldPlay
[29], "Video world models w/ long-term spatial memory" [38], Matrix-Game [48], Cosmos [1] /
Cosmos-Transfer1 [2], HunyuanWorld-1.0 [31], HY-World / LingBot-World [32] — the window-based,
sequentially-generated systems WorldFM rejects [§1 p.2] for their interactive latency and
accumulating spatial errors.

**Runtime / distillation:** **DMD [42]** (2-step, t_mid=200 — the shipped real-time generator) ·
VSD (named score-distillation lineage) · inherited multi-step **DPM-Solver++** teacher sampler
(`worldfm_infer.py:497-567`) for the un-distilled path.

**Foundational:** PixArt-Σ [8] (selected DiT backbone) · DiT [24] · Latent Diffusion / Stable
Diffusion [27] · FLUX [18] (HunyuanWorld-1.0's backbone, i.e. the panorama substitute's
substrate) · Sora [6,7] ("video generation models as world simulators").

---

*When the leader agent compares InSpatio-WorldFM to any of these, the relevant axis mapping is in
[`02-comparison-framework.md`](02-comparison-framework.md) and WorldFM's own values are in
[`worldfm.card.yaml`](worldfm.card.yaml).*
