"""Tests for the optional BinaryOODGuard.

Uses only the committed synthetic_train CSV fixture (public-safe) plus small
inline OOD lists. No private data.

Verifies:
  * the guard trains and predicts;
  * OOD texts get low in-domain probability vs in-domain texts;
  * the router escalates OOD messages with reason OOD_CLASSIFIER when the
    guard is attached;
  * the guard is OFF by default (backward compatibility);
  * in-domain messages still ROUTE with the guard attached.
"""
import pytest

from hermes_intent_router import IntentRouter, BinaryOODGuard
from hermes_intent_router.schema import Decision
from hermes_intent_router.train import train_from_csv


# Synthetic OOD snippets (clearly out-of-scope).
OOD_SMALL = [
    "what is the capital of france",
    "weather in tokyo tomorrow",
    "my wifi is down",
    "how to reset my router",
    "bitcoin price today",
    "latest election results",
    "best restaurant near me",
    "how to lose weight",
    "solve 2x + 5 = 15",
    "movie times at the cinema",
]


def _bundle(synthetic_train):
    return train_from_csv(str(synthetic_train), model_type="mlp", random_state=42)


def test_ood_guard_trains_and_threshold_in_range():
    g = BinaryOODGuard()
    g.fit(["i want to pay","saya nak bayar","boleh tolong buat payment","unit trust"],
          ["capital of france","wifi down","bitcoin price","movie times"], variant="word")
    assert g.vec is not None
    assert g.clf is not None
    assert 0 < g.threshold < 1


def test_ood_texts_get_low_probability():
    g = BinaryOODGuard()
    ind = ["saya nak buat bayaran","boleh tolong buat bayaran","i want to make a payment",
           "when is the payment deadline","reschedule my appointment","unit trust investment"]
    ood = ["what is the capital of france","weather today","my wifi broke","bitcoin price"]
    g.fit(ind, ood, variant="word")
    po = g.predict_proba(ood)
    pi = g.predict_proba(ind)
    assert po.mean() < pi.mean()
    assert (po < 0.5).mean() >= 0.7


def test_in_domain_texts_get_high_probability():
    g = BinaryOODGuard()
    ind = ["saya nak buat bayaran","boleh tolong buat bayaran","i want to make a payment",
           "when is the payment deadline","reschedule my appointment","unit trust investment"]
    ood = ["what is the capital of france","weather today","my wifi broke","bitcoin price"]
    g.fit(ind, ood, variant="word")
    pi = g.predict_proba(ind)
    assert (pi >= 0.5).mean() >= 0.8


def test_reject_returns_bool():
    g = BinaryOODGuard()
    g.fit(["i want to pay","saya nak bayar"], ["capital of france","wifi down"], variant="word")
    assert isinstance(g.reject("what is the capital of france"), bool)
    assert isinstance(g.reject("saya nak buat bayaran"), bool)


def test_save_load_roundtrip(tmp_path):
    g = BinaryOODGuard()
    g.fit(["i want to pay","saya nak bayar"], ["capital of france","wifi down"], variant="word")
    g.threshold = 0.65
    path = tmp_path / "ood.joblib"
    g.save(str(path))
    g2 = BinaryOODGuard.load(str(path))
    assert g2.threshold == 0.65
    orig = g.predict_proba(["capital of france", "wifi down"])
    loaded = g2.predict_proba(["capital of france", "wifi down"])
    assert all(abs(a-b) < 1e-6 for a, b in zip(orig, loaded))


def test_router_escalates_ood_with_reason(synthetic_train):
    import csv
    train_texts = [row["text"] for row in csv.DictReader(open(synthetic_train))]
    bundle = _bundle(synthetic_train)
    guard = BinaryOODGuard()
    guard.fit(train_texts, OOD_SMALL, variant="word")
    guard.threshold = 0.5
    router = IntentRouter.from_bundle(bundle, ood_model=guard)
    # in-domain should route
    r = router.predict("boleh tolong buat bayaran")
    assert r.decision == Decision.ROUTE
    # OOD should escalate with OOD_CLASSIFIER (or SHORT_AMBIGUOUS for short msgs)
    r2 = router.predict("what is the capital of france")
    assert r2.decision == Decision.ESCALATE
    assert r2.reason in ("OOD_CLASSIFIER", "SHORT_AMBIGUOUS")


def test_router_has_no_ood_guard_by_default(synthetic_train):
    bundle = _bundle(synthetic_train)
    router = IntentRouter.from_bundle(bundle)
    assert router._ood_guard is None
    r = router.predict("what is the capital of france")
    assert r.reason != "OOD_CLASSIFIER"


def test_ood_guard_preserves_in_domain_routing(synthetic_train):
    import csv
    train_texts = [row["text"] for row in csv.DictReader(open(synthetic_train))]
    bundle = _bundle(synthetic_train)
    guard = BinaryOODGuard()
    guard.fit(train_texts, OOD_SMALL, variant="word")
    guard.threshold = 0.5
    router = IntentRouter.from_bundle(bundle, ood_model=guard)
    r = router.predict("boleh tolong buat bayaran")
    assert r.decision == Decision.ROUTE
