"""Optional FastAPI service — import-guarded.

Importing this module without FastAPI installed raises a clear error, and
the test suite never imports it unless FastAPI is present. The service
trains the from-scratch logistic model on synthetic data at startup and
exposes global importance, local explanations, PDP, and quality gates.
"""

from __future__ import annotations

try:
    from fastapi import FastAPI  # type: ignore[import-error]
    from pydantic import BaseModel  # type: ignore[import-error]
except ImportError as exc:  # pragma: no cover - exercised only with FastAPI
    raise RuntimeError(
        "FastAPI is not installed. Install it with: pip install -r requirements.txt"
    ) from exc

from typing import Dict, List

from explainability.data import FEATURE_NAMES, make_dataset
from explainability.global_explainer import mean_absolute_effect, permutation_importance
from explainability.local_explainer import counterfactual_search, occlusion_contributions
from explainability.model import LogisticModel
from explainability.pdp import partial_dependence
from explainability.quality import evaluate_quality_gates

app = FastAPI(title="Model Explainability Platform")

_X, _y = make_dataset()
MODEL = LogisticModel().fit(_X, _y)


class ExplainRequest(BaseModel):
    row: List[float]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "model": "logistic-stub-v1"}


@app.get("/importance")
def importance() -> Dict[str, float]:
    return permutation_importance(MODEL, _X, _y, FEATURE_NAMES)


@app.post("/explain")
def explain(req: ExplainRequest) -> Dict[str, object]:
    cf = counterfactual_search(MODEL, req.row, _X, FEATURE_NAMES)
    return {
        "prediction": MODEL.predict([req.row])[0],
        "probability": MODEL.predict_proba([req.row])[0],
        "occlusion_contributions": occlusion_contributions(
            MODEL, req.row, _X, FEATURE_NAMES
        ),
        "counterfactual": (
            {
                "changed_features": cf.changed_features,
                "distance": cf.distance,
                "row": cf.row,
            }
            if cf
            else None
        ),
    }


@app.get("/pdp/{feature}")
def pdp(feature: str) -> Dict[str, object]:
    idx = FEATURE_NAMES.index(feature)
    curve = partial_dependence(MODEL, _X, idx, feature)
    return {
        "feature": curve.feature,
        "grid_values": curve.grid_values,
        "average_predictions": curve.average_predictions,
    }


@app.get("/quality-gates")
def quality_gates() -> Dict[str, object]:
    result = evaluate_quality_gates(MODEL, _X, _y, FEATURE_NAMES)
    return {
        "faithfulness": result.faithfulness,
        "stability": result.stability,
        "monotonicity": result.monotonicity,
        "promotion_allowed": result.promotion_allowed,
    }


@app.get("/mean-effect")
def mean_effect() -> Dict[str, float]:
    return mean_absolute_effect(MODEL, _X, FEATURE_NAMES)
