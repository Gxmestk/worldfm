# WorldFM docs — index

Map of the documentation written alongside this fork of `inspatio/worldfm`
(arXiv `2603.11911`). This index does not duplicate the docs — it is purely the
map. Follow the links for content.

> 👉 **Picking up this project?** Read [HANDOFF.md](HANDOFF.md) first — current status, hard rules
> (uv-always, 16 GB cgroup, ask-before-model-OOM), the next task, and resume commands.

## Purpose

Three questions, three kinds of answer:

- **Understand the model** → the [`learning/`](learning/) study scaffold (paper + code, ten docs).
- **Run the demo** → [`run-guide.md`](run-guide.md) (ops recipe).
- **What changed here & why** → [`repo-mods.md`](repo-mods.md) (working-tree change record).

## Documents

| Path | One-line description |
|---|---|
| [HANDOFF.md](HANDOFF.md) | **Read first if picking up the project** — status, hard rules, next task, resume commands. |
| [learning/](learning/) | Paper + code study scaffold — start at [learning/README.md](learning/README.md). |
| [learning/01-big-picture.md](learning/01-big-picture.md) | 30,000-ft overview of InSpatio-WorldFM: what it does, what is released, what is not. |
| [learning/02-comparison-framework.md](learning/02-comparison-framework.md) | Shared schema ("methodology card") for comparing this method against peer NVS / world-model papers. |
| [learning/03-methodology-reference.md](learning/03-methodology-reference.md) | Authoritative methodology distilled from the technical report: formulation, training, data, eval. |
| [learning/04-paper-code-crosswalk.md](learning/04-paper-code-crosswalk.md) | Per-mechanism map: claim → paper section/equation → repo `path:line` → present in released code? |
| [learning/05-formulation.md](learning/05-formulation.md) | Deep dive on the DiT forward, tri-conditioning, hybrid spatial memory, 2-step distillation. |
| [learning/06-training-distillation.md](learning/06-training-distillation.md) | Deep dive on the Pre- / Middle- / Post-Training stages (paper-only; training code is withheld). |
| [learning/07-training-data.md](learning/07-training-data.md) | Deep dive on training data, synthetic finetuning, and offline anchor/reference providers. |
| [learning/08-runtime-and-decode.md](learning/08-runtime-and-decode.md) | Runtime & decode internals, including the `torch.compile` + condition-cache optimization. |
| [learning/09-diagrams.md](learning/09-diagrams.md) | ASCII mechanism diagrams (agent-parseable, render anywhere). |
| [learning/10-glossary-lineage.md](learning/10-glossary-lineage.md) | Glossary, baselines, and the techniques InSpatio-WorldFM builds on. |
| [learning/worldfm.card.yaml](learning/worldfm.card.yaml) | Machine-readable methodology card (schema defined in 02). |
| [run-guide.md](run-guide.md) | Copy-pasteable run/ops recipe for the demo on a memory-constrained box. |
| [repo-mods.md](repo-mods.md) | Atomic record of uncommitted working-tree edits made to run in a 16 GB cgroup. |
| [live-loop.md](live-loop.md) | Run/ops recipe for the **interactive WebSocket fly-through** server (`live_server.py` + `live/viewer.html`). |
| [live-loop-internals.md](live-loop-internals.md) | *Why the live loop works:* cached-anchor cache structure, arbitrary-pose rendering, per-frame stochasticity. |

## Measured results (outside this folder)

Runtime numbers and output artifacts from an actual run live with the outputs,
not with the docs:

- [`../../outputs/mario/metrics.md`](../outputs/mario/metrics.md) — measured metrics for the `mario` scene (timings, memory, quality), generated from this checkout. Machine-readable companion: `../../outputs/mario/metrics.json`.

## Broader AI knowledge (outside this folder)

For general, non-WorldFM-specific AI / world-model concepts, see the shared
knowledge folder one level up:

- [`../../docs/`](../) — sibling knowledge docs at the workspace root (this `worldfm/docs/` is scoped to this repo only).

## Suggested reading orders

Pick the row that matches your goal; each is self-contained.

| Goal | Read in this order |
|---|---|
| **Understand the paper** | [01](learning/01-big-picture.md) → [03](learning/03-methodology-reference.md) → [05](learning/05-formulation.md) → [06](learning/06-training-distillation.md) → [04](learning/04-paper-code-crosswalk.md) (verify claims against code) |
| **Run it** | [run-guide.md](run-guide.md) → [repo-mods.md](repo-mods.md) (only if on a constrained box) → [live-loop.md](live-loop.md) (interactive fly-through) → [../../outputs/mario/metrics.md](../outputs/mario/metrics.md) (what "good" looks like) |
| **What changed & why (this fork)** | [repo-mods.md](repo-mods.md) → [learning/08-runtime-and-decode.md](learning/08-runtime-and-decode.md) (why the optimization matters) → [run-guide.md](run-guide.md) (how the changes are exercised) |
| **Compare against other methods** | [02](learning/02-comparison-framework.md) → [03](learning/03-methodology-reference.md) → [10](learning/10-glossary-lineage.md) → [worldfm.card.yaml](learning/worldfm.card.yaml) |
| **Quick orientation (15 min)** | [01](learning/01-big-picture.md) → [09](learning/09-diagrams.md) → [run-guide.md](run-guide.md) |

## Conventions

- **Paper refs:** `[pN]` = PDF page, `[§x]` = the paper's own section number, in `paper/2603.11911v3.pdf`.
- **Code refs:** `path:line`, relative to the repo root, verified against this checkout.
- **Status flags in learning/:** ✅ present in released inference code · 📝 paper-only (training code is withheld).
- **Verified-on date:** 2026-07-29 (each doc carries its own verification line).
- **All paths in this index are relative** — links resolve from this `docs/` folder.
