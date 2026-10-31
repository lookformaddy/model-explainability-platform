"""Tests: registry immutability/versioning, global + local explainers,
PDP/ICE, quality gates, drift, and reports."""

import pytest

from explainability.data import FEATURE_NAMES, make_dataset
from explainability.drift import drift_report
from explainability.global_explainer import (
    interaction_screen,
    mean_absolute_effect,
    permutation_importance,
)
from explainability.local_explainer import counterfactual_search, occlusion_contributions
from explainability.model import LogisticModel
from explainability.pdp import ice_lines, partial_dependence
from explainability.quality import evaluate_quality_gates
from explainability.registry import ExplanationRegistry, dataset_snapshot_hash
from explainability.report import build_report_json, build_report_markdown
from explainability.stats import pearson, spearman


@pytest.fixture(scope="module")
def dataset():
    X, y = make_dataset(n=200, seed=0)
    model = LogisticModel().fit(X, y)
    return X, y, model


def test_dataset_hash_is_deterministic_and_sensitive():
    X, y = make_dataset(n=50, seed=1)
    h1 = dataset_snapshot_hash(X, FEATURE_NAMES)
    h2 = dataset_snapshot_hash(X, FEATURE_NAMES)
    assert h1 == h2
    X2 = [row[:] for row in X]
    X2[0][0] += 1.0
    assert dataset_snapshot_hash(X2, FEATURE_NAMES) != h1


def test_registry_versions_and_immutability():
    reg = ExplanationRegistry()
    h = "abc123"
    r1 = reg.register("v1", h, "importance", {"a": 1.0})
    r2 = reg.register("v1", h, "importance", {"a": 2.0})
    assert r1.explanation_version == 1
    assert r2.explanation_version == 2
    assert reg.latest("v1", h, "importance").explanation_version == 2
    assert len(reg.history("v1", h, "importance")) == 2
    assert reg.latest("v1", h, "missing") is None
    with pytest.raises(Exception):
        r1.model_version = "tampered"  # frozen dataclass


def test_registry_record_payload_roundtrip():
    reg = ExplanationRegistry()
    rec = reg.register("v1", "h", "pdp", {"curve": [0.1, 0.2]})
    assert rec.payload_dict()["curve"] == [0.1, 0.2]


def test_permutation_importance_recovers_signal(dataset):
    X, y, model = dataset
    imp = permutation_importance(model, X, y, FEATURE_NAMES, n_repeats=3)
    ranked = sorted(imp, key=lambda n: -imp[n])
    # Signal features are recent_activity and income_band; noise-ish ones rank lower.
    assert "recent_activity" in ranked[:2]
    assert imp["recent_activity"] > imp["support_contacts"]


def test_mean_absolute_effect_ranking(dataset):
    X, y, model = dataset
    eff = mean_absolute_effect(model, X, FEATURE_NAMES)
    assert set(eff) == set(FEATURE_NAMES)
    assert all(v >= 0 for v in eff.values())
    assert eff["recent_activity"] > 0


def test_interaction_screen_keys_and_types(dataset):
    X, y, model = dataset
    scores = interaction_screen(model, X, y, FEATURE_NAMES, n_repeats=2)
    assert len(scores) == 6  # C(4, 2)
    assert all(isinstance(v, float) for v in scores.values())


def test_occlusion_contributions_shape(dataset):
    X, y, model = dataset
    contrib = occlusion_contributions(model, X[0], X, FEATURE_NAMES)
    assert set(contrib) == set(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in contrib.values())


def test_counterfactual_flips_prediction(dataset):
    X, y, model = dataset
    # Find a row predicted 0 with room to move; fall back to any row.
    target = next((row for row in X if model.predict([row])[0] == 0), X[0])
    cf = counterfactual_search(model, target, X, FEATURE_NAMES)
    assert cf is not None
    assert cf.new_prediction != cf.original_prediction
    assert model.predict([cf.row])[0] == cf.new_prediction
    assert cf.distance > 0
    # Within observed bounds.
    for j, value in enumerate(cf.row):
        assert min(r[j] for r in X) <= value <= max(r[j] for r in X)


def test_pdp_and_ice_shapes(dataset):
    X, y, model = dataset
    curve = partial_dependence(model, X, 2, "recent_activity", n_bins=8)
    assert len(curve.grid_values) == 8
    assert len(curve.average_predictions) == 8
    assert curve.average_predictions[-1] > curve.average_predictions[0]
    lines = ice_lines(model, X, 2, n_bins=8, sample_size=5)
    assert len(lines) == 5 and all(len(line) == 8 for line in lines)


def test_quality_gates_promotion_for_good_model(dataset):
    X, y, model = dataset
    result = evaluate_quality_gates(model, X, y, FEATURE_NAMES)
    assert result.promotion_allowed is True
    assert result.stability >= 0.7
    assert -1.0 <= result.faithfulness <= 1.0


def test_quality_gates_fail_closed_for_random_model(dataset):
    X, y, _ = dataset

    class RandomModel:
        def predict_proba(self, rows):
            return [0.5 for _ in rows]

        def predict(self, rows):
            return [0 for _ in rows]

    result = evaluate_quality_gates(RandomModel(), X, y, FEATURE_NAMES)
    # A constant model carries no signal: stability/faithfulness collapse.
    assert result.promotion_allowed is False


def test_spearman_and_pearson_basics():
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)


def test_drift_alert_and_block_thresholds():
    ref = {"a": 0.5, "b": 0.3, "c": 0.2, "d": 0.1}
    same = {"a": 0.5, "b": 0.3, "c": 0.2, "d": 0.1}
    report = drift_report(ref, same)
    assert report.spearman_correlation == pytest.approx(1.0)
    assert report.alert is False and report.blocked is False
    reversed_rank = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.5}
    report2 = drift_report(ref, reversed_rank)
    assert report2.blocked is True
    assert report2.top_movers[0]["rank_change"] == 3


def test_report_generation(dataset):
    X, y, model = dataset
    imp = permutation_importance(model, X, y, FEATURE_NAMES, n_repeats=2)
    eff = mean_absolute_effect(model, X, FEATURE_NAMES)
    quality = evaluate_quality_gates(model, X, y, FEATURE_NAMES)
    snap = dataset_snapshot_hash(X, FEATURE_NAMES)
    md = build_report_markdown("v1", snap, FEATURE_NAMES, imp, eff, quality)
    assert "Promotion allowed" in md
    assert "recent_activity" in md
    payload = build_report_json("v1", snap, imp, eff, quality)
    assert payload["quality_gates"]["promotion_allowed"] is True
    assert payload["model_version"] == "v1"
