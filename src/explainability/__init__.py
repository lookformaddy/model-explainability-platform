"""Model Explainability Platform — versioned, auditable explanations for ML models.

Core package: registry, global/local explainers, PDP/ICE, quality gates,
drift detection, and reporting. Standard-library only; the optional FastAPI
service lives in :mod:`explainability.service` behind a guarded import.
"""

from explainability.drift import drift_report, spearman
from explainability.global_explainer import (
    interaction_screen,
    mean_absolute_effect,
    permutation_importance,
)
from explainability.local_explainer import counterfactual_search, occlusion_contributions
from explainability.model import LogisticModel, predict, predict_proba
from explainability.pdp import ice_lines, partial_dependence
from explainability.quality import QualityGateResult, evaluate_quality_gates
from explainability.registry import ExplanationRecord, ExplanationRegistry
from explainability.report import build_report_json, build_report_markdown

__all__ = [
    "ExplanationRecord",
    "ExplanationRegistry",
    "QualityGateResult",
    "LogisticModel",
    "predict",
    "predict_proba",
    "permutation_importance",
    "mean_absolute_effect",
    "interaction_screen",
    "occlusion_contributions",
    "counterfactual_search",
    "partial_dependence",
    "ice_lines",
    "evaluate_quality_gates",
    "spearman",
    "drift_report",
    "build_report_markdown",
    "build_report_json",
]
