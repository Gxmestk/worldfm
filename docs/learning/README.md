# Learning InSpatio-WorldFM — Paper + Code Study Scaffold

> **Status:** the full ten-doc scaffold is written (big picture, comparison
> framework, methodology reference, crosswalk, four deep-dives, diagrams,
> glossary) plus a machine-readable methodology card.

## PROVENANCE

| | |
|---|---|
| **Current as of** | 2026-07-29 |
| **Paper source** | `paper/2603.11911v3.pdf` — *InSpatio-WorldFM: An Open-Source Real-Time Generative Frame Model for Spatial Intelligence*, InSpatio Team, arXiv v3, 6 May 2026 (17 pp). **Single paper** (no intro/full split). |
| **Code source** | Released **inference-only** repo (`github.com/inspatio/worldfm`). Apache-2.0 library code; the 1-/2-step distilled student weights are released. **All training code (Stages I–III) is withheld** — the load-bearing WorldFM distinction. |
| **Paper refs** | `[pN]` = PDF page; `[§x]` = paper's own section number. Code refs are `path:line` relative to the repo root, every one verified in this checkout. |
| **Machine-readable card** | [`worldfm.card.yaml`](worldfm.card.yaml) |

## Documents

| Doc | What it is |
|---|---|
| [**01-big-picture.md**](01-big-picture.md) | The **big picture**: InSpatio-WorldFM as a frame-based **PixArt-Σ DiT** for real-time novel-view synthesis, driven by one reference image + a target camera pose, emitting one spatially consistent view per call via a **2-step distilled denoiser**. Taxonomy framing, rollout loop, code map, and the methodology card. **Start here.** |
| [**02-comparison-framework.md**](02-comparison-framework.md) | The shared **comparison schema** (the "methodology card"), the comparison **views**, a **proposed roster** of peer NVS / world-model / real-time-generative papers, and the **leader-agent workflow spec** for multi-paper comparison. |
| [**03-methodology-reference.md**](03-methodology-reference.md) | **Authoritative methodology from the full report**: formulation (conditional frame-model objective, hybrid spatial memory), the **progressive three-stage training** (Pre-Training → Middle-Training → Post-Training), data, and evaluation. |
| [**04-paper-code-crosswalk.md**](04-paper-code-crosswalk.md) | **Mechanism → paper §/equation → `code:line`** map, with a status emoji per row. *The* map for reading any paper claim against the released implementation. |
| [**05-formulation.md**](05-formulation.md) | **Formulation deep dive**: the conditional latent-diffusion objective, exactly what enters the DiT, how the **hybrid spatial memory** (image-conditioning warp + nearby-view tokens) is assembled, and how the model is distilled to a **2-step real-time sampler**. |
| [**06-training-distillation.md**](06-training-distillation.md) | **Training & distillation deep dive**: Stages I–III + the **DMD/2-step distillation objective**, with an **honest status banner** — the entire 3-stage pipeline is paper-only; the shipped `default.yaml` carries *inference/distillation-schedule* constants, not training-loop hparams. |
| [**07-training-data.md**](07-training-data.md) | **Training data deep dive**: the real + synthetic data mixture (**no counts released**), synthetic-data finetuning, and the **offline anchor-build pipeline** (`run_pipeline.py` Step 1–3) — the inference-time realization of §2.1, *not* the training corpus. |
| [**08-runtime-and-decode.md**](08-runtime-and-decode.md) | **Runtime & decode internals**: the shipped stage sequence + `Profiler` in `run_pipeline.py`, decode internals in `modules/worldfm_infer.py`, and the **`torch.compile` + condition-cache** optimization (commit `d51ada2`). FPS/latency figures are **paper-claimed, not reproduced**. |
| [**09-diagrams.md**](09-diagrams.md) | **ASCII mechanism diagrams**: two-phase offline/online system overview, the tri-condition input, the hybrid-memory assembly, the roll-out, the distillation setup, and a results snapshot — render in any terminal/markdown. |
| [**10-glossary-lineage.md**](10-glossary-lineage.md) | **Lineage & glossary**: every ancestor InSpatio-WorldFM inherits from (PixArt-Σ, DMD, MoGe, …), the baselines it is measured against, and peer methods grouped by the §1 taxonomy. |
| [**worldfm.card.yaml**](worldfm.card.yaml) | InSpatio-WorldFM's methodology card in **machine-readable YAML** — the parseable reference for the leader/comparison agent (schema defined in `02`). |
| *(this README)* | Study scaffold: provenance, doc index, reading orders, conventions, open questions. |

## Suggested reading orders

**🏃 Quickstart (~45 min)** — *what is this and where's the proof?*
1. [**01** — big picture](01-big-picture.md) (TL;DR + taxonomy + rollout loop)
2. [**04** — crosswalk](04-paper-code-crosswalk.md) (what actually ships vs paper-only)
3. [**09** — diagrams](09-diagrams.md) (the ASCII system overview, one screen)
4. [**worldfm.card.yaml**](worldfm.card.yaml) (the card, in 30 lines)

**🔬 Deep-dive (paper study)** — *read the methodology end to end.*
1. [**01** — big picture](01-big-picture.md) → [**03** — methodology reference](03-methodology-reference.md)
2. [**05** — formulation](05-formulation.md) (objective → memory → 2-step sampler)
3. [**06** — training & distillation](06-training-distillation.md) (Stages I–III + DMD)
4. [**07** — training data](07-training-data.md) → [**08** — runtime & decode](08-runtime-and-decode.md)
5. [**10** — glossary & lineage](10-glossary-lineage.md) (place it among ancestors)

**🛠️ Implementer / reproducer** — *what can I actually run and modify?*
1. [**04** — crosswalk](04-paper-code-crosswalk.md) (status legend is your north star)
2. [**05** — formulation](05-formulation.md) § what-enters-the-DiT + memory assembly
3. [**08** — runtime & decode](08-runtime-and-decode.md) (stage sequence, `Profiler`, the `d51ada2` optimization)
4. [**06** — training](06-training-distillation.md) (read the **honest status banner** first: training is paper-only)
5. [**02** — comparison framework](02-comparison-framework.md) (if extending the card for your own method)

## Conventions

- **Estimates vs measured.** This is an **inference-only release**. **FPS and per-stage latency numbers are the paper's claim, not reproduced here** — the only in-repo measurement surface is the optional `Profiler` (`run_pipeline.py`), which ships with **no recorded `performance.json`**. Anywhere a doc says "measured," it means measured-by-the-paper; anything tagged with a code ref was read directly from this checkout. See [**08**](08-runtime-and-decode.md).
- **Current as of.** **2026-07-29** (every doc carries this in its header). The `d51ada2` perf commit is included; later repo changes are not reflected.
- **Status emoji legend** (used in [**04**](04-paper-code-crosswalk.md), [**06**](06-training-distillation.md), [**07**](07-training-data.md)):
  - ✅ **active in shipped inference code**
  - ⚙️ **code/schema present, but training loop not released**
  - 📝 **paper-only** (described in the paper, not in the repo at all)
  - ⚠️ **implemented in source but NOT activated on the shipped inference path** (the load-bearing WorldFM distinction)
- **Citation style.** Paper: `[pN]` = page, `[§x]` = paper section (numbering as printed), all to `paper/2603.11911v3.pdf`. Code: `path:line` relative to repo root, verified in this checkout. Bracketed `[N]` numbers in [**10**](10-glossary-lineage.md) are the paper's own citation indices.
- **Cross-links.** Every doc links to its siblings; the card ([`worldfm.card.yaml`](worldfm.card.yaml)) is referenced from [01](01-big-picture.md), [02](02-comparison-framework.md), and [03](03-methodology-reference.md).

## Pointer to the methodology card

The single machine-readable source of truth for InSpatio-WorldFM's methodology — for the leader/comparison agent or any downstream tooling — is
[`worldfm.card.yaml`](worldfm.card.yaml). Its schema (axes, views, roster format) is defined in
[`02-comparison-framework.md`](02-comparison-framework.md), and its values are justified across [03](03-methodology-reference.md), [05](05-formulation.md), [06](06-training-distillation.md), and [07](07-training-data.md).

---

*Everything above frames the released inference repo against the single arXiv report.
The recurring caveat: training is paper-only, latencies are paper-claimed — those are the
two places the docs mark explicitly so you don't over-trust the demo.*
