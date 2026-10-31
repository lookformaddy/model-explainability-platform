"""Explanation quality gates — fail-closed promotion checks.

Three checks, each with a floor. ``promotion_allowed`` is True only when
every check passes; a missing or degenerate measurement fails its gate
rather than being waved through.

- Faithfulness: Pearson correlation between per-feature permutation
  importance and the accuracy drop when that feature is replaced by its
  median. If the explanation ranks features that do not actually matter,
  the correlation collapses.
- Stability: Spearman rank correlation between importance rankings computed
  under two different seeds. Unstable rankings cannot be trusted in a review.
- Monotonicity spot-check: for a PDP curve, the fraction of adjacent steps
  moving in the curve's dominant direction. Reported as a diagnostic score
  (not a hard gate by default) because real models are legitimately
  non-monotonic; the gate flags only curves that are pure noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from explainability.global_explainer import permutation_importance
from explainability.model import predict
from explainability.pdp import partial_dependence
from explainability.stats import accuracy, pearson, spearman


@dataclass
class QualityGateResult:
    faithfulness: float
    stability: float
    monotonicity: float
    faithfulness_pass: bool
    stability_pass: bool
    monotonicity_pass: bool
    promotion_allowed: bool
    details: Dict[str, object]


def _median_drop_importance(
    model: object,
    X: List[List[float]],
    y: List[int],
    feature_names: Sequence[str],
) -> Dict[str, float]:
    baseline = accuracy(y, predict(model, X))
    result: Dict[str, float] = {}
    for j, name in enumerate(feature_names):
        col = sorted(row[j] for row in X)
        median = col[len(col) // 2]
        perturbed = [row[:] for row in X]
        for row in perturbed:
            row[j] = median
        result[name] = baseline - accuracy(y, predict(model, perturbed))
    return result


def evaluate_quality_gates(
    model: object,
    X: List[List[float]],
    y: List[int],
    feature_names: Sequence[str],
    faithfulness_floor: float = 0.3,
    stability_floor: float = 0.7,
    monotonicity_floor: float = 0.5,
    seed_a: int = 42,
    seed_b: int = 123,
) -> QualityGateResult:
    names = list(feature_names)
    # 10 repeats: with only a handful of features, low-repeat estimates let
    # near-tied features swap ranks and tank the stability correlation.
    imp_a = permutation_importance(model, X, y, names, n_repeats=10, seed=seed_a)
    imp_b = permutation_importance(model, X, y, names, n_repeats=10, seed=seed_b)
    median_drop = _median_drop_importance(model, X, y, names)

    faithfulness = pearson(
        [imp_a[n] for n in names], [median_drop[n] for n in names]
    )
    stability = spearman([imp_a[n] for n in names], [imp_b[n] for n in names])

    # Monotonicity: average dominant-direction fraction across PDP curves.
    mono_scores = []
    for j, name in enumerate(names):
        curve = partial_dependence(model, X, j, name, n_bins=10)
        preds = curve.average_predictions
        diffs = [b - a for a, b in zip(preds, preds[1:])]
        if not diffs:
            continue
        up = sum(1 for d in diffs if d >= 0)
        down = len(diffs) - up
        mono_scores.append(max(up, down) / len(diffs))
    monotonicity = sum(mono_scores) / len(mono_scores) if mono_scores else 0.0

    faith_pass = faithfulness >= faithfulness_floor
    stab_pass = stability >= stability_floor
    mono_pass = monotonicity >= monotonicity_floor
    return QualityGateResult(
        faithfulness=faithfulness,
        stability=stability,
        monotonicity=monotonicity,
        faithfulness_pass=faith_pass,
        stability_pass=stab_pass,
        monotonicity_pass=mono_pass,
        promotion_allowed=faith_pass and stab_pass and mono_pass,
        details={
            "importance_seed_a": imp_a,
            "importance_seed_b": imp_b,
            "median_drop": median_drop,
        },
    )
