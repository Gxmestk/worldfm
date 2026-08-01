# 02 — Comparative Methodology Framework

> Goal: let **leader AI agents** compare InSpatio-WorldFM's methodology against
> other novel-view-synthesis / world-model / real-time-generative papers **on a
> shared schema**, so you see *how different method choices yield different
> outcomes* — not just who looks better in a demo.
>
> This doc defines (1) the **comparison schema** (the "methodology card"), (2) the
> **views** the comparison produces, (3) a **proposed roster** of papers to compare,
> and (4) the **leader-agent workflow spec** that produces it. InSpatio-WorldFM's own
> card lives in [`01-big-picture.md`](01-big-picture.md) (markdown) and
> [`worldfm.card.yaml`](worldfm.card.yaml) (machine-readable). Grounded in the single
> arXiv report `2603.11911` (page refs `[pN]`, section refs `[§x]`) and this repo's
> code (`path:line`). Verified 2026-07-29.

## Why a schema (and why this one)

Apples-to-apples comparison fails when each paper is summarized free-form. We pin a
**fixed set of axes** — drawn from InSpatio-WorldFM's own framing spine
(`frame vs video` paradigm → real-time latency → spatial-consistency mechanism →
camera control → training-stage evolution → offline/online split → NVS [§1 p.2,
§2.4.1 p.4-6]) plus standard world-model dimensions — so every paper is reduced to
the same vectors. Then "different methods → different outcomes" becomes a readable
matrix. The four challenge buckets below are the same four the rest of this doc set
uses (Challenge 1 interaction/control, 2 consistency/memory, 3 stability, 4 runtime).

## The methodology card (one per paper)

| Axis | What to capture |
|---|---|
| **Identity** | Name, group, date; open-source? weights? license? inference-only vs full release? |
| **Base backbone** | Architecture family (DiT, AR transformer, video DiT, U-Net) + specific model. |
| **Setting** | i2v / t2v / **NVS from a reference** / panorama; **interactive real-time vs offline**; single-view vs multi-view. |
| **Paradigm** | **frame-based (independent per-frame)** vs video/window-based temporal; diffusion vs autoregressive; latent vs pixel. |
| **Conditioning inputs** | reference image, target camera pose, depth/point cloud, text, action. |
| **Control mechanism** *(Challenge 1)* | none / additive ray embedding (Plücker) / **attention-modulating pose (PRoPE)** / pure-parametric MLP / cross-attention KV. |
| **Memory mechanism** *(Challenge 2)* | none / posed-frame primitive (RTFM) / keyframe feature-warping (StarGen) / **hybrid explicit-3D-anchor + implicit-reference** / temporal window recurrence. |
| **Stability / anti-drift** *(Challenge 3)* | none / temporal-window recurrence / **conditioning-only (no recurrence) + training regularization** / synthetic-GT finetune. |
| **Runtime** *(Challenge 4)* | multi-step teacher / **few-step distillation (DMD 2-step)** / 1-step; semantic-latency handling (per-frame vs chunk-wait). |
| **Resolution / fps / latency** | Reported FPS, GPU, resolution. |
| **Distinguishing idea** | One sentence: the thing this paper is *for*. |
| **Trade-offs / limits** | What it gives up; failure modes. |
| **Outcome evidence** | Headline results / demos — and whether they are quantified or qualitative-only. |

> Filling "Distinguishing idea" + "Trade-offs" + "Outcome evidence" per card is what
> turns a table into an *explanation* of method→outcome. For InSpatio-WorldFM the
> load-bearing distinguishing idea is: *generate each frame independently yet keep
> global 3D consistency purely through conditioning (explicit point-cloud anchor +
> implicit reference), then distill to 2-step for real-time* [§1 p.2]. The honest
> trade-off — and a row to fill carefully for every peer — is that evaluation here is
> **qualitative only** [§3 p.12-13]: no FID/PSNR/LPIPS/consistency numbers exist, so
> "outcome evidence" must distinguish *measured* gains from *asserted* ones.

## Comparison views (what the agents produce)

1. **By-challenge matrix** — for each of the 4 challenges, which mechanism each paper
   uses and *why it leads to its observed outcome* (e.g., "RTFM's posed-frame memory
   → cheap and real-time but coarse geometry; StarGen's keyframe feature-warping →
   strong keyframe fidelity but bound to an offline video-DiT; InSpatio-WorldFM's
   hybrid anchor+reference → recovers fine appearance that RTFM's primitives lose,
   at the cost of an offline panorama/depth provider").
2. **Head-to-head cards** — InSpatio-WorldFM vs each direct peer, axis-by-axis, with
   a "what you gain / what you lose" verdict (e.g., vs RTFM [37]: gain open-source
   weights + explicit-3D-anchor precision; lose closed-model benchmark parity, which
   is unverifiable either way since RTFM releases no code).
3. **Lineage / design-tree** — who begat whom (PixArt-Σ → its DiT backbone; RTFM →
   the posed-frame-memory idea it rejects-and-extends; VGGT/DUSt3R/MoGe/MapAnything
   → its point-cloud anchor build; PRoPE → its camera encoding; DMD → its 2-step
   speed). Makes "novel vs inherited" visible.
4. **Decision guide** — "if your priority is X (consumer-GPU real-time NVS /
   open-source reproducibility / large motion boundary / dynamic content / quantitative
   benchmarks), the method space looks like Y, and the best-bet papers are Z." This is
   the most useful output for *your own research*.

## Proposed roster

Grouped by role (citations are the report's `[N]`). Start with **Group A** for the
sharpest comparison; expand as needed. Full bibliographic identities live in
[`10-glossary-lineage.md`](10-glossary-lineage.md).

- **A. Direct peers — real-time / interactive NVS & world/frame models** (closest
  comparables):
  RTFM [37] (closed real-time frame model), StarGen [45] (keyframe feature-warping
  scene generation), GEN3C [26] (3D-informed world-consistent video), Voyager [16],
  WorldPlay [29], Video-world-models-with-long-term-spatial-memory [38], Cosmos [1] /
  Cosmos-Transfer1 [2], Matrix-Game [48], HunyuanWorld-1.0 [31] (also the shipped
  panorama substitute), HY-World / LingBot-World [32], Genie 3 [10], WonderWorld [43],
  WonderJourney [44], SceneX [50], Layer-Pano3D [39], LucidDreamer [9].
- **B. Technique progenitors** (explain *where* InSpatio-WorldFM's pieces come from):
  PRoPE / Cameras-as-RPE [20] (its adopted camera encoding), Plücker-ray /
  CameraCtrl-II [15] and pure-parametric / ReCamMaster [3] (the two compared-and-
  rejected camera encodings), VGGT [34] / DUSt3R [36] / MoGe [35] / MapAnything [17] /
  MegaDepth [21] (feedforward 3D reconstruction for the point-cloud anchor),
  Stable Virtual Camera [49] / Cat3D [14] (multi-view diffusion anchor/reference
  providers), Diffusion360 [13] / Text2360 [46] (panorama providers), DMD [42]
  (its 2-step distillation), VSD (its score-distillation lineage).
- **C. Backbones / diffusion foundation models** (the substrate):
  PixArt-Σ [8] (the selected DiT backbone), DiT [24], Latent Diffusion / Stable
  Diffusion [27], FLUX [18] (HunyuanWorld-1.0's backbone, the panorama substitute).

## Leader-agent workflow spec

A fan-out workflow (see the harness `Workflow` tool) that:

1. **Input:** InSpatio-WorldFM's filled card ([`01-big-picture.md`](01-big-picture.md)
   + [`worldfm.card.yaml`](worldfm.card.yaml)) + a chosen roster.
2. **Per paper (parallel):** one agent researches the paper (web; or your local PDF if
   you place it in `paper/`) and returns a **filled methodology card** (structured).
   Each card must mark which axes are *quantitatively* evidenced vs *qualitatively
   asserted* — a recurring gap in this subfield.
3. **Per challenge (parallel):** one agent builds the **by-challenge matrix** across
   all cards (control / consistency / stability / runtime).
4. **Synthesis (1 agent, xhigh):** produces the **head-to-head verdicts**, the
   **design-tree**, and the **decision guide**, grounded in the cards.
5. **Output:** a new `docs/learning/03-comparison-<roster>.md` + the structured cards
   archived under `docs/learning/cards/`.

The cards are deliberately structured so this runs deterministically and so you can
re-run it with a different roster (e.g., your own lab's papers) by dropping in new
PDFs and re-pointing the roster.

> **Paper↔code note (verify):** when filling InSpatio-WorldFM's *own* card, the
> synthesis notes a few axes whose "shipped" value differs from the paper's "adopted"
> value — the camera-encoding axis (PRoPE is **adopted** [§2.4.1 p.5-6] but **not
> activated** on the released inference path; `modules/worldfm_infer.py:97` never
> passes camera matrices) and the panorama axis (the internal multi-view/panorama
> model is **withheld**; README:52-53, with HunyuanWorld-1.0 as the open-source
> substitute). The full open-question list lives in
> [`04-paper-code-crosswalk.md`](04-paper-code-crosswalk.md) and the YAML's
> `flags_to_verify:`.

## How you can drive it

- **Use the proposed roster** (Groups A/B above) — fastest start; Group A is the
  sharpest because RTFM [37] and StarGen [45] are the named head-to-head targets.
- **Point at your own papers** — put them in `paper/` (as you did for
  `2603.11911v3`) and tell me the filenames; the agents will read them directly
  instead of web-searching.
- **Focus an axis** — e.g., "compare only the **memory** mechanisms" (RTFM posed-frame
  vs StarGen feature-warp vs InSpatio-WorldFM hybrid) or only **camera control**
  (PRoPE vs Plücker vs pure-parametric) for a deeper, narrower view.
- **Add a target outcome** — e.g., "I care about consumer-GPU (RTX 4090-class)
  real-time NVS" or "I need quantitative multi-view-consistency benchmarks" and the
  decision guide will rank methods against *your* priority (and flag, as above, that
  InSpatio-WorldFM currently offers neither measured metrics nor dynamic content).

---

*Tell me which roster / focus you want (see the question that follows) and I'll launch
the leader-agent comparison.*
