"""Synthetic demo data — generic, invented, no real-world data.

Features: ``income_band``, ``account_age_months``, ``recent_activity``,
``support_contacts``. The label depends mostly on ``recent_activity`` and
``income_band`` so explanations have a real signal to recover.
"""

from __future__ import annotations

import random
from typing import List, Tuple

FEATURE_NAMES = ["income_band", "account_age_months", "recent_activity", "support_contacts"]


def make_dataset(n: int = 300, seed: int = 0) -> Tuple[List[List[float]], List[int]]:
    rng = random.Random(seed)
    X: List[List[float]] = []
    y: List[int] = []
    for _ in range(n):
        income = rng.randint(0, 4)
        age = rng.randint(0, 60)
        activity = rng.randint(0, 20)
        support = rng.randint(0, 5)
        score = 0.9 * activity + 0.8 * income - 0.02 * age - 0.3 * support
        score += rng.gauss(0, 2.0)
        X.append([float(income), float(age), float(activity), float(support)])
        y.append(1 if score > 8 else 0)
    return X, y
