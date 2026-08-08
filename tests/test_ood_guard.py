"""Tests for the OOD / ambiguity guard layer.

These tests use only the committed synthetic fixtures (no private data) and
exercise the short/ambiguous, margin, and reason-field behaviour added to
``IntentRouter.predict``.
"""
from hermes_intent_router import Decision, IntentRouter
from hermes_intent_router.train import train_from_csv


def _router(tmp_path, train_csv, *, ood_csv=None):
    out = tmp_path / "g.joblib"
    train_from_csv(train_csv, model_type="mlp", out_path=out, ood_path=ood_csv)
    return IntentRouter.load(out)


def test_known_ood_failures_now_escalate(synthetic_train, tmp_path):
    """The two failures documented in the acceptance report must now escalate."""
    router = _router(tmp_path, synthetic_train)
    for msg in ("what is the capital of france", "ok"):
        r = router.predict(msg)
        assert r.decision == Decision.ESCALATE, (msg, r.to_json())
        assert r.intent is None
        assert r.reason is not None


def test_known_failures_carry_reason(synthetic_train, tmp_path):
    router = _router(tmp_path, synthetic_train)
    r = router.predict("ok")
    assert r.reason == "SHORT_AMBIGUOUS"


def test_short_messages_escalate(synthetic_train, tmp_path):
    router = _router(tmp_path, synthetic_train)
    for msg in ("hi", "help", "yes", "250"):
        r = router.predict(msg)
        assert r.decision == Decision.ESCALATE
        assert r.intent is None


def test_general_knowledge_questions_escalate(synthetic_train, tmp_path):
    router = _router(tmp_path, synthetic_train)
    for msg in ("what is the capital of france", "what time is it now",
                "what is the weather like today", "how tall is mount everest"):
        r = router.predict(msg)
        assert r.decision == Decision.ESCALATE, (msg, r.to_json())
        assert r.intent is None


def test_valid_in_domain_messages_route(synthetic_train, tmp_path):
    """Guard must not block genuine in-domain messages."""
    router = _router(tmp_path, synthetic_train)
    valid = [
        "i want to make a payment",
        "saya nak bayar sekarang",
        "i need to reschedule my appointment tomorrow",
        "boleh tukar janji esok please",
    ]
    for msg in valid:
        r = router.predict(msg)
        # Each valid message should either route (best case) or, if the model
        # genuinely lacks confidence, escalate with a LOW_CONFIDENCE reason --
        # but never with SHORT_AMBIGUOUS (which would be a guard bug).
        if r.decision == Decision.ROUTE:
            assert r.intent is not None
            assert r.reason is None
        else:
            assert r.reason == "LOW_CONFIDENCE", (msg, r.to_json())


def test_route_result_has_no_reason(synthetic_train, tmp_path):
    """A ROUTE result must never carry a reason field."""
    router = _router(tmp_path, synthetic_train)
    routed = [r for r in router.predict_many(
        ["i want to make a payment", "saya nak bayar", "please reschedule"])
        if r.decision == Decision.ROUTE]
    assert routed, "expected at least one routed message in the synthetic data"
    for r in routed:
        assert r.reason is None


def test_escalate_result_carries_reason(synthetic_train, tmp_path):
    """An ESCALATE result from a guard must carry a reason code."""
    router = _router(tmp_path, synthetic_train)
    r = router.predict("ok")
    d = r.to_dict()
    assert d["decision"] == "ESCALATE"
    assert "reason" in d
    assert d["reason"] in {"SHORT_AMBIGUOUS", "LOW_CONFIDENCE", "LOW_MARGIN", "HIGH_ENTROPY"}


def test_route_json_omits_reason(synthetic_train, tmp_path):
    """Backward-compat: ROUTE JSON must not include 'reason'."""
    import json
    router = _router(tmp_path, synthetic_train)
    # find a routed sample
    samples = ["i want to make a payment", "saya nak bayar sekarang"]
    routed = [r for r in router.predict_many(samples) if r.decision == Decision.ROUTE]
    assert routed
    payload = json.loads(routed[0].to_json())
    assert "reason" not in payload


def test_round_trip_preserves_reason(synthetic_train, tmp_path):
    import json
    router = _router(tmp_path, synthetic_train)
    r = router.predict("ok")
    d = r.to_dict()
    payload = json.loads(r.to_json())
    assert payload["reason"] == "SHORT_AMBIGUOUS"
    # from_dict round-trips
    r2 = type(r).from_dict(d)
    assert r2.reason == r.reason


def test_guard_can_be_disabled(synthetic_train, tmp_path):
    """Pass guards=[] to disable the guard chain entirely."""
    out = tmp_path / "g.joblib"
    train_from_csv(synthetic_train, model_type="mlp", out_path=out)
    from hermes_intent_router.model import load_bundle
    bundle = load_bundle(out)
    router = IntentRouter.from_bundle(bundle, guards=[])
    # With guards disabled, "ok" is routed iff confidence >= threshold (legacy)
    r = router.predict("ok")
    assert r.reason is None  # no guard reason when disabled
    assert isinstance(r.confidence, float)
