"""Global explanations: what drives the model overall.

- Permutation importance: seeded shuffles of one feature at a time; the
  importance is the drop in accuracy versus the unshuffled baseline.
- Mean absolute effect: mean |prediction change| when a feature is replaced
  by its median (a cheap occlusion-based ranking).
- Interaction screen: for each feature pair, the joint-shuffle drop minus
  the sum of the two single-feature drops. A large positive residual flags
  a pair whose effect is more than the sum of its parts.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence

from explainability.model import predict
from explainability.stats import accuracy, mean


def _shuffled(
    X: List[List[float]], feature_idx: int, rng: random.Random
) -> List[List[float]]:
    col = [row[feature_idx] for row in X]
    shuffled = col[:]
    rng.shuffle(shuffled)
    out = [row[:] for row in X]
    for i, row in enumerate(out):
        row[feature_idx] = shuffled[i]
    return out


def permutation_importance(
    model: object,
    X: List[List[float]],
    y: List[int],
    feature_names: Sequence[str],
    n_repeats: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """Mean accuracy drop per feature over ``n_repeats`` seeded shuffles."""
    rng = random.Random(seed)
    baseline = accuracy(y, predict(model, X))
    result: Dict[str, float] = {}
    for j, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            shuffled = _shuffled(X, j, rng)
            drops.append(baseline - accuracy(y, predict(model, shuffled)))
        result[name] = mean(drops)
    return result


def mean_absolute_effect(
    model: object,
    X: List[List[float]],
    feature_names: Sequence[str],
) -> Dict[str, float]:
    """Mean |p(x) - p(x with feature replaced by its median)| per feature."""
    from explainability.model import predict_proba

    base = predict_proba(model, X)
    medians = []
    for j in range(len(feature_names)):
        col = sorted(row[j] for row in X)
        medians.append(col[len(col) // 2])
    result: Dict[str, float] = {}
    for j, name in enumerate(feature_names):
        perturbed = [row[:] for row in X]
        for row in perturbed:
            row[j] = medians[j]
        alt = predict_proba(model, perturbed)
        result[name] = mean([abs(a - b) for a, b in zip(base, alt)])
    return result


def interaction_screen(
    model: object,
    X: List[List[float]],
    y: List[int],
    feature_names: Sequence[str],
    n_repeats: int = 3,
    seed: int = 42,
) -> Dict[str, float]:
    """Pairwise interaction score: joint drop - sum of single drops.

    Positive values mean the pair matters jointly beyond its parts.
    """
    singles = permutation_importance(model, X, y, feature_names, n_repeats, seed)
    rng = random.Random(seed + 1)
    baseline = accuracy(y, predict(model, X))
    scores: Dict[str, float] = {}
    names = list(feature_names)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            drops = []
            for _ in range(n_repeats):
                shuffled = _shuffled(X, a, rng)
                shuffled = _shuffled(shuffled, b, rng)
                drops.append(baseline - accuracy(y, predict(model, shuffled)))
            joint = mean(drops)
            scores[f"{names[a]} x {names[b]}"] = joint - (
                singles[names[a]] + singles[names[b]]
            )
    return scores
