"""Local explanations: why *this* prediction came out the way it did.

- Occlusion contributions: for each feature, p(x) - p(x with that feature
  replaced by its training median). Positive means the feature's actual
  value pushes the prediction up.
- Counterfactual search: the nearest valid input (within observed feature
  bounds) that flips the predicted class, minimising normalised L1 change.
  Single-feature grid moves are tried first; a seeded multi-feature random
  search is the fallback. Returns ``None`` when no flip is found.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from explainability.model import predict, predict_proba


@dataclass
class Counterfactual:
    changed_features: Dict[str, float]
    original_prediction: int
    new_prediction: int
    distance: float  # normalised L1 distance from the original row
    row: List[float]


def _medians(X: List[List[float]]) -> List[float]:
    out = []
    for j in range(len(X[0])):
        col = sorted(row[j] for row in X)
        out.append(col[len(col) // 2])
    return out


def occlusion_contributions(
    model: object,
    row: List[float],
    X_reference: List[List[float]],
    feature_names: Sequence[str],
) -> Dict[str, float]:
    """Per-feature leave-one-out deltas for a single prediction."""
    base = predict_proba(model, [row])[0]
    medians = _medians(X_reference)
    result: Dict[str, float] = {}
    for j, name in enumerate(feature_names):
        perturbed = row[:]
        perturbed[j] = medians[j]
        result[name] = base - predict_proba(model, [perturbed])[0]
    return result


def counterfactual_search(
    model: object,
    row: List[float],
    X_reference: List[List[float]],
    feature_names: Sequence[str],
    grid_steps: int = 11,
    random_tries: int = 2000,
    seed: int = 7,
) -> Optional[Counterfactual]:
    """Nearest valid class-flipping input, minimal normalised L1 change."""
    original = predict(model, [row])[0]
    bounds: List[Tuple[float, float]] = [
        (min(r[j] for r in X_reference), max(r[j] for r in X_reference))
        for j in range(len(row))
    ]
    spans = [max(hi - lo, 1e-9) for lo, hi in bounds]

    def _distance(candidate: List[float]) -> float:
        return sum(abs(c - o) / s for c, o, s in zip(candidate, row, spans))

    def _flips(candidate: List[float]) -> bool:
        return predict(model, [candidate])[0] != original

    best: Optional[Counterfactual] = None

    def _consider(candidate: List[float]) -> None:
        nonlocal best
        if not _flips(candidate):
            return
        dist = _distance(candidate)
        if best is None or dist < best.distance:
            changed = {
                name: candidate[j]
                for j, name in enumerate(feature_names)
                if candidate[j] != row[j]
            }
            best = Counterfactual(
                changed_features=changed,
                original_prediction=original,
                new_prediction=1 - original,
                distance=dist,
                row=candidate,
            )

    # Single-feature grid moves first (most interpretable counterfactuals).
    for j, (lo, hi) in enumerate(bounds):
        for step in range(grid_steps):
            value = lo + (hi - lo) * step / (grid_steps - 1) if grid_steps > 1 else lo
            candidate = row[:]
            candidate[j] = value
            _consider(candidate)

    # Fallback: seeded random multi-feature search.
    rng = random.Random(seed)
    for _ in range(random_tries):
        candidate = [
            lo + rng.random() * (hi - lo) for lo, hi in bounds
        ]
        _consider(candidate)
    return best
