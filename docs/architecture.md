# Architecture

## Overview

```
                ┌──────────────────────────────────────────────┐
                │  Consumers: CLI demo, FastAPI service, CI    │
                └──────────────┬───────────────┬───────────────┘
                               │               │
                ┌──────────────▼───┐  ┌────────▼────────┐
                │  Explainers      │  │  Governance     │
                │  - global        │  │  - registry     │
                │  - local         │  │  - quality gates│
                │  - PDP / ICE     │  │  - drift        │
                └──────┬───────────┘  └────────┬────────┘
                       │                       │
                ┌──────▼───────────────────────▼────────┐
                │  Reports (Markdown + JSON)            │
                └───────────────────────────────────────┘
                       │
                ┌──────▼───────────────────────┐
                │  Explained model (any object │
                │  with predict/predict_proba)  │
                └──────────────────────────────┘
```

## Modules

| Module | Responsibility |
|---|---|
| `model.py` | From-scratch logistic regression; the demo's explained model. Any model with the same interface can be substituted. |
| `data.py` | Synthetic dataset generator (generic features, invented labels). |
| `registry.py` | Append-only, versioned, immutable explanation records keyed by (model version, dataset hash, kind). |
| `global_explainer.py` | Permutation importance, mean absolute effect, pairwise interaction screen. |
| `local_explainer.py` | Occlusion contributions, nearest-valid counterfactual search. |
| `pdp.py` | Partial dependence curves and ICE lines over binned feature grids. |
| `quality.py` | Faithfulness, stability, monotonicity gates; fail-closed promotion decision. |
| `drift.py` | Spearman comparison of importance rankings; alert/block thresholds. |
| `report.py` | Markdown + JSON report builders. |
| `service.py` | Optional FastAPI service; import fails cleanly without FastAPI. |
| `cli.py` | End-to-end demo: train → explain → gate → drift → report. |

## Data flow: a model release

1. Train (or load) a model; snapshot the evaluation dataset; hash it
   (`dataset_snapshot_hash`).
2. Compute global explanations and register them under
   (model version, dataset hash).
3. Run quality gates. If any gate fails, promotion is blocked and the
   report says so explicitly.
4. Compare the new importance ranking against the previous release's
   ranking; flag alert/block drift.
5. Emit the Markdown + JSON report and register the gate outcome.

## Design decisions

- **Model-agnostic interface.** Explainers depend only on
  `predict_proba`/`predict`, not on the model class, so the platform can
  wrap a production model behind a thin adapter.
- **Determinism.** Every stochastic step (shuffles, counterfactual
  fallback search) takes an explicit seed, so a report can be regenerated
  bit-for-bit from (model version, dataset hash, seed).
- **Fail-closed gates.** Missing or degenerate measurements fail, never
  pass. A gate failure is a human-review trigger, not an error to retry.
- **Plausible counterfactuals.** Search is confined to observed feature
  bounds; single-feature moves are preferred for actionability.
- **No workflow files in the repo root.** The CI definition is kept at
  `docs/github-actions-ci.yml` because the integration used to publish
  this repository cannot write `.github/workflows/`; move it there to
  activate CI.

## Scaling notes

- Permutation importance costs O(repeats × features × inference). For
  wide feature sets, sample the evaluation set and reduce repeats.
- PDP/ICE cost O(bins × rows × inference); sample ICE lines (default 20).
- The registry here is in-memory; a production deployment would persist
  records to an append-only store (object storage + a metadata table)
  with the same key and immutability contract.
