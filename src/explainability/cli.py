"""CLI demo: train the stub model, explain it, gate it, report it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from explainability.data import FEATURE_NAMES, make_dataset  # noqa: E402
from explainability.drift import drift_report  # noqa: E402
from explainability.global_explainer import (  # noqa: E402
    interaction_screen,
    mean_absolute_effect,
    permutation_importance,
)
from explainability.local_explainer import (  # noqa: E402
    counterfactual_search,
    occlusion_contributions,
)
from explainability.model import LogisticModel  # noqa: E402
from explainability.pdp import partial_dependence  # noqa: E402
from explainability.quality import evaluate_quality_gates  # noqa: E402
from explainability.registry import (  # noqa: E402
    ExplanationRegistry,
    dataset_snapshot_hash,
)
from explainability.report import build_report_json, build_report_markdown, render_report_json  # noqa: E402


def run_demo(out_dir: Path) -> dict:
    X, y = make_dataset()
    model = LogisticModel().fit(X, y)
    model_version = "logistic-stub-v1"
    snap = dataset_snapshot_hash(X, FEATURE_NAMES)

    importance = permutation_importance(model, X, y, FEATURE_NAMES)
    effect = mean_absolute_effect(model, X, FEATURE_NAMES)
    interactions = interaction_screen(model, X, y, FEATURE_NAMES)
    quality = evaluate_quality_gates(model, X, y, FEATURE_NAMES)

    row = X[0]
    local = occlusion_contributions(model, row, X, FEATURE_NAMES)
    cf = counterfactual_search(model, row, X, FEATURE_NAMES)
    pdp = partial_dependence(model, X, FEATURE_NAMES.index("recent_activity"), "recent_activity")

    # Drift demo: reference ranking from a different seed's data.
    X_ref, y_ref = make_dataset(seed=99)
    ref_model = LogisticModel().fit(X_ref, y_ref)
    ref_importance = permutation_importance(ref_model, X_ref, y_ref, FEATURE_NAMES)
    drift = drift_report(ref_importance, importance)

    registry = ExplanationRegistry()
    registry.register(model_version, snap, "permutation_importance", {"importance": importance})
    registry.register(model_version, snap, "quality_gates", {
        "faithfulness": quality.faithfulness,
        "stability": quality.stability,
        "promotion_allowed": quality.promotion_allowed,
    })

    drift_text = (
        f"Spearman vs reference: {drift.spearman_correlation:.3f} — "
        f"alert={drift.alert}, blocked={drift.blocked}; "
        f"top mover: {drift.top_movers[0]['feature'] if drift.top_movers else 'none'}"
    )
    md = build_report_markdown(model_version, snap, FEATURE_NAMES, importance, effect, quality, drift_text)
    payload = build_report_json(model_version, snap, importance, effect, quality, {
        "spearman_correlation": drift.spearman_correlation,
        "alert": drift.alert,
        "blocked": drift.blocked,
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "explanation-report.md").write_text(md)
    (out_dir / "explanation-report.json").write_text(render_report_json(payload))

    print("Permutation importance (ranked):")
    for name, value in sorted(importance.items(), key=lambda kv: -kv[1]):
        print(f"  {name:22s} {value:.4f}")
    print(f"\nTop interaction: {max(interactions, key=interactions.get)} "
          f"({max(interactions.values()):.4f})")
    print(f"\nLocal explanation for row 0 (pred={model.predict([row])[0]}):")
    for name, value in sorted(local.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {name:22s} {value:+.4f}")
    if cf:
        print(f"\nCounterfactual (distance {cf.distance:.3f}): {cf.changed_features}")
    else:
        print("\nCounterfactual: none found within bounds")
    print(f"\nPDP recent_activity: {pdp.average_predictions[0]:.3f} -> "
          f"{pdp.average_predictions[-1]:.3f}")
    print(f"\nQuality gates: faithfulness={quality.faithfulness:.3f} "
          f"stability={quality.stability:.3f} monotonicity={quality.monotonicity:.3f}")
    print(f"Promotion allowed: {quality.promotion_allowed}")
    print(f"Drift vs reference: Spearman={drift.spearman_correlation:.3f} "
          f"alert={drift.alert} blocked={drift.blocked}")
    print(f"\nReports written to {out_dir}")
    return {"promotion_allowed": quality.promotion_allowed, "drift_blocked": drift.blocked}


def main() -> int:
    parser = argparse.ArgumentParser(description="Model Explainability Platform demo")
    parser.add_argument("--out-dir", default="reports", help="Directory for reports")
    parser.add_argument("--serve", action="store_true", help="Run the FastAPI service")
    args = parser.parse_args()
    if args.serve:
        try:
            import uvicorn  # type: ignore[import-error]
        except ImportError:
            print("uvicorn is not installed; install requirements.txt first.", file=sys.stderr)
            return 1
        from explainability.service import app

        uvicorn.run(app, host="0.0.0.0", port=8000)
        return 0
    run_demo(Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
