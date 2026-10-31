"""Partial dependence (PDP) and individual conditional expectation (ICE).

PDP: for a feature, the mean model output as that feature sweeps a grid of
binned values while every other feature stays at its observed value.
ICE: the same curve for individual instances, so heterogeneity (curves
going in different directions) is visible instead of averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from explainability.model import predict_proba


@dataclass
class PdpCurve:
    feature: str
    grid_values: List[float]
    average_predictions: List[float]


def _grid(X: List[List[float]], feature_idx: int, n_bins: int) -> List[float]:
    col = sorted(row[feature_idx] for row in X)
    lo, hi = col[0], col[-1]
    if hi == lo:
        return [lo]
    # Quantile-ish grid: evenly spaced between min and max.
    return [lo + (hi - lo) * i / (n_bins - 1) for i in range(n_bins)]


def partial_dependence(
    model: object,
    X: List[List[float]],
    feature_idx: int,
    feature_name: str,
    n_bins: int = 10,
) -> PdpCurve:
    grid = _grid(X, feature_idx, n_bins)
    averages = []
    for value in grid:
        replaced = [row[:] for row in X]
        for row in replaced:
            row[feature_idx] = value
        preds = predict_proba(model, replaced)
        averages.append(sum(preds) / len(preds))
    return PdpCurve(feature=feature_name, grid_values=grid, average_predictions=averages)


def ice_lines(
    model: object,
    X: List[List[float]],
    feature_idx: int,
    n_bins: int = 10,
    sample_size: int = 20,
) -> List[List[float]]:
    """One ICE curve per sampled instance (first ``sample_size`` rows)."""
    grid = _grid(X, feature_idx, n_bins)
    lines = []
    for row in X[:sample_size]:
        curve = []
        for value in grid:
            candidate = row[:]
            candidate[feature_idx] = value
            curve.append(predict_proba(model, [candidate])[0])
        lines.append(curve)
    return lines
