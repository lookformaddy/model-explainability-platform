"""A tiny from-scratch binary classifier used as the explained model.

Pure standard library: a logistic regression trained by batch gradient
descent. This is deliberately small — the platform explains *any* model
exposing ``predict_proba(rows)`` / ``predict(rows)``; the stub exists so the
demo, service, and tests run on CPU with no third-party dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Sequence

Row = Sequence[float]
PredictProbaFn = Callable[[List[List[float]]], List[float]]


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class LogisticModel:
    """Logistic regression with an explicit weight vector and bias."""

    weights: List[float] = field(default_factory=list)
    bias: float = 0.0

    def fit(
        self,
        X: List[List[float]],
        y: List[int],
        epochs: int = 500,
        lr: float = 0.1,
    ) -> "LogisticModel":
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        n = len(X)
        for _ in range(epochs):
            grad_w = [0.0] * n_features
            grad_b = 0.0
            for row, label in zip(X, y):
                p = _sigmoid(sum(w * v for w, v in zip(self.weights, row)) + self.bias)
                err = p - label
                for j, v in enumerate(row):
                    grad_w[j] += err * v
                grad_b += err
            for j in range(n_features):
                self.weights[j] -= lr * grad_w[j] / n
            self.bias -= lr * grad_b / n
        return self

    def predict_proba(self, X: List[List[float]]) -> List[float]:
        return [
            _sigmoid(sum(w * v for w, v in zip(self.weights, row)) + self.bias)
            for row in X
        ]

    def predict(self, X: List[List[float]]) -> List[int]:
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]


def predict_proba(model: object, X: List[List[float]]) -> List[float]:
    """Duck-typed probability prediction for any model with predict_proba."""
    fn = getattr(model, "predict_proba")
    return list(fn(X))


def predict(model: object, X: List[List[float]]) -> List[int]:
    """Duck-typed class prediction; falls back to a 0.5 threshold."""
    if hasattr(model, "predict"):
        return list(model.predict(X))  # type: ignore[attr-defined]
    return [1 if p >= 0.5 else 0 for p in predict_proba(model, X)]
