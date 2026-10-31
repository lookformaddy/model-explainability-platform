# Model Explainability Platform

## The enterprise problem

Regulated enterprises cannot ship a model they cannot explain. Reviewers,
auditors, and product owners ask the same questions before every release:
*which features drive this model, why did it decide this way for this
customer, has what it relies on drifted since the last release, and can we
prove the explanations themselves are trustworthy?* Ad-hoc notebooks answer
those questions once and then rot. This platform turns explanations into
versioned, auditable artifacts with quality gates that fail closed: a model
whose explanations are unfaithful or unstable cannot be promoted, no matter
how good its accuracy looks.

## What it does

- **Explanation registry** — versioned, immutable records tied to a model
  version and a SHA-256 hash of the exact dataset snapshot explained.
- **Global explanations** — permutation importance (seeded shuffles),
  mean-absolute-effect ranking, and a pairwise feature-interaction screen.
- **Local explanations** — per-prediction occlusion contributions and a
  nearest-valid counterfactual search (minimal normalised change within
  observed feature bounds).
- **Partial dependence + ICE** — 1-D PDP curves with individual conditional
  expectation lines, so averaged-away heterogeneity stays visible.
- **Quality gates** — faithfulness (importance vs. actual performance drop),
  stability (rank correlation across seeds), monotonicity spot-checks.
  Promotion is allowed only when every gate passes.
- **Explanation drift** — Spearman rank correlation between current and
  reference importance rankings, with alert and block thresholds.
- **Reports** — Markdown + JSON explanation reports per model version.
- **Service + CLI** — a FastAPI service (import-guarded; core runs without
  it) and a CLI demo that trains a from-scratch model and produces a report.

## Architecture overview

```
src/explainability/
  model.py            from-scratch logistic regression (the explained model)
  data.py             synthetic demo dataset (generic, invented)
  registry.py         versioned, immutable explanation records
  global_explainer.py permutation importance, mean effect, interactions
  local_explainer.py  occlusion contributions, counterfactual search
  pdp.py              partial dependence + ICE lines
  quality.py          faithfulness / stability / monotonicity gates
  drift.py            Spearman-based explanation drift + thresholds
  report.py           Markdown + JSON report generation
  service.py          optional FastAPI service (guarded import)
  cli.py              demo CLI (train -> explain -> gate -> report)
```

The core is standard-library only and model-agnostic: any object exposing
`predict_proba(rows)` / `predict(rows)` can be explained. See
[`docs/architecture.md`](docs/architecture.md) for the full design.

## How to run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # pytest; FastAPI/uvicorn optional
make test                          # run the test suite
python3 -m explainability.cli --out-dir reports   # demo + reports
```

With FastAPI installed, start the service:

```bash
python3 -m explainability.cli --serve   # http://localhost:8000
curl localhost:8000/importance
curl -X POST localhost:8000/explain -H 'Content-Type: application/json' \
  -d '{"row": [2, 12, 9, 1]}'
```

Or with Docker:

```bash
docker build -t model-explainability-platform .
docker run -p 8000:8000 model-explainability-platform
```

## Key design decisions and trade-offs

- **Stdlib-first core.** Permutation importance, Spearman/Pearson, PDP, and
  the model itself are implemented from scratch. This removes dependency
  risk and makes every number in a report reproducible from the source.
  The cost is speed on very large datasets — permutation importance is
  O(repeats × features × inference) — so production use should sample.
- **Median occlusion baseline.** Local contributions replace a feature with
  its training median rather than zero: zero is often out-of-distribution
  for real features, while the median is a plausible "typical" value.
- **Fail-closed gates.** A degenerate measurement (constant model, zero
  variance) fails its gate instead of passing vacuously. A blocked
  promotion is a review trigger, not a bug.
- **Counterfactual validity by construction.** Candidates are confined to
  observed feature bounds, so counterfactuals are plausible inputs rather
  than adversarial artifacts. Single-feature moves are tried first because
  they are the most actionable ("change X, outcome flips").
- **Registry immutability.** Records are frozen dataclasses and the registry
  is append-only with version bumps. An explanation can be superseded but
  never silently rewritten — the property auditors actually need.
- **CI shipped as a doc.** The GitHub Actions workflow lives at
  `docs/github-actions-ci.yml`; moving it to `.github/workflows/ci.yml`
  activates it (the push integration used for this repo cannot write
  workflow files, so it is deliberately not placed there).

## What it demonstrates

Model-agnostic explainability engineering: versioned explanation artifacts,
global and local explanation methods implemented from first principles,
statistical quality gates that fail closed, drift detection on explanation
rankings, and report/service surfaces suitable for a model-review workflow
— all on a clean-room codebase with synthetic data only.
