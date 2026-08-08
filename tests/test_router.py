"""End-to-end router tests against synthetic fixtures (public-safe)."""

import json

import pytest

from hermes_intent_router import IntentRouter, Decision
from hermes_intent_router.train import train_from_csv


@pytest.fixture(scope="module")
def router(synthetic_train, tmp_path_factory):
    out = tmp_path_factory.mktemp("model") / "router.joblib"
    bundle = train_from_csv(synthetic_train, model_type="mlp", out_path=out,
                            version="test")
    return IntentRouter.load(out), out, bundle


def test_load_and_intents(router):
    r, _, _ = router
    assert set(r.intents) == {
        "PAYMENT", "RESCHEDULE", "CANCEL", "MENTORING_ENQUIRY",
        "PROGRAM_ENQUIRY", "BOOK_PURCHASE", "UNIT_TRUST", "COMPLAINT",
        "GENERAL", "OTHER",
    }
    assert r.model_type == "mlp"


def test_predict_route(router):
    r, _, _ = router
    result = r.predict("i want to make a payment")
    payload = json.loads(result.to_json())
    assert payload["decision"] == "ROUTE"
    assert payload["intent"] == "PAYMENT"
    assert payload["model"] == "mlp"
    assert 0.0 <= payload["confidence"] <= 1.0


def test_predict_malay(router):
    r, _, _ = router
    result = r.predict("saya nak buat bayaran sekarang")
    assert result.decision == Decision.ROUTE
    assert result.intent == "PAYMENT"


def test_predict_mixed(router):
    r, _, _ = router
    # mixed Malay-English synthetic — phrase from the RESCHEDULE template pool
    result = r.predict("saya nak tukar appointment esok")
    assert result.intent == "RESCHEDULE"


def test_predict_unknown_escalates_low_confidence(router):
    r, _, _ = router
    # OOD-style, out-of-vocabulary message
    result = r.predict("please book a flight to singapore for me tomorrow")
    assert result.decision in (Decision.ROUTE, Decision.ESCALATE)
    # If it routes, it must be with high confidence (threshold contract)
    if result.decision == Decision.ROUTE:
        assert result.confidence >= r.threshold


def test_bundle_saves_sidecar(router):
    _, out, bundle = router
    sidecar = out.with_suffix(".json")
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["model_type"] == "mlp"
    assert meta["confidence_threshold"] == bundle.confidence_threshold
    assert "eval_metrics" in meta


def test_threshold_reasonable(router):
    _, _, bundle = router
    assert 0.0 < bundle.confidence_threshold <= 0.99
    # sanity: threshold must be a float and in-bundle metadata coherent
    assert bundle.eval_metrics["validation_macro_f1"] > 0.5