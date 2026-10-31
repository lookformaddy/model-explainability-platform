"""Explanation drift: has what the model relies on changed?

Compares a current importance ranking against a reference ranking with
Spearman rank correlation. Below ``alert_threshold`` the drift is flagged
for review; below ``block_threshold`` it is treated as a breaking change
that should block promotion until a human signs off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from explainability.stats import spearman


@dataclass
class DriftReport:
    spearman_correlation: float
    alert_threshold: float
    block_threshold: float
    alert: bool
    blocked: bool
    top_movers: List[Dict[str, object]]


def drift_report(
    reference_importance: Dict[str, float],
    current_importance: Dict[str, float],
    alert_threshold: float = 0.8,
    block_threshold: float = 0.5,
) -> DriftReport:
    shared = sorted(set(reference_importance) & set(current_importance))
    corr = spearman(
        [reference_importance[n] for n in shared],
        [current_importance[n] for n in shared],
    )
    ref_rank = {n: i for i, n in enumerate(sorted(shared, key=lambda n: -reference_importance[n]))}
    cur_rank = {n: i for i, n in enumerate(sorted(shared, key=lambda n: -current_importance[n]))}
    movers = sorted(
        (
            {
                "feature": n,
                "reference_rank": ref_rank[n],
                "current_rank": cur_rank[n],
                "rank_change": abs(ref_rank[n] - cur_rank[n]),
                "reference_importance": reference_importance[n],
                "current_importance": current_importance[n],
            }
            for n in shared
        ),
        key=lambda d: -d["rank_change"],  # type: ignore[operator]
    )
    return DriftReport(
        spearman_correlation=corr,
        alert_threshold=alert_threshold,
        block_threshold=block_threshold,
        alert=corr < alert_threshold,
        blocked=corr < block_threshold,
        top_movers=movers,
    )
