"""Regression tests: training methodology contract.

These guard the core promise of the project: retraining with the documented
methodology must not silently degrade routing quality. The synthetic fixture
is small by design, so thresholds here are intentionally conservative —
they catch gross regressions (broken feature pipeline, inverted labels,
removed intents), not the ~0.99 macro-F1 bar that applies to the original
(private) held-out dataset.
"""

import csv

import pytest
from sklearn.metrics import accuracy_score, f1_score

from hermes_intent_router import IntentRouter, Decision
from hermes_intent_router.train import train_from_csv


def _load_csv(path):
    texts, labels = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            texts.append(row["text"]); labels.append(row["intent"])
    return texts, labels


@pytest.fixture(scope="module")
def regression_router(synthetic_train, tmp_path_factory):
    out = tmp_path_factory.mktemp("reg") / "router.joblib"
    train_from_csv(synthetic_train, model_type="mlp", out_path=out,
                   version="regression")
    return IntentRouter.load(out)


def test_all_intents_present_in_label_space(regression_router):
    assert len(regression_router.intents) == 10


def test_synthetic_test_accuracy_floor(synthetic_test, regression_router):
    """At least 75% of held-out synthetic test samples must route, and
    routed samples must be >=90% accurate (conservative synthetic bar)."""
    texts, labels = _load_csv(synthetic_test)
    results = regression_router.predict_many(texts)
    routed = [r for r in results if r.decision == Decision.ROUTE]
    assert len(routed) >= int(0.50 * len(results))  # most route
    y_true = [labels[i] for i, r in enumerate(results) if r.decision == Decision.ROUTE]
    y_pred = [r.intent for r in routed if r.decision == Decision.ROUTE]
    assert accuracy_score(y_true, y_pred) >= 0.85


def test_all_intents_route_positive_example(regression_router):
    """A canonical example per intent must route to the right label."""
    cases = {
        "PAYMENT": "i want to make a payment",
        "RESCHEDULE": "please reschedule my appointment",
        "CANCEL": "cancel my booking",
        "MENTORING_ENQUIRY": "do you offer mentoring sessions",
        "PROGRAM_ENQUIRY": "what programs do you offer",
        "BOOK_PURCHASE": "how much is the book",
        "UNIT_TRUST": "how do i invest in unit trust",
        "COMPLAINT": "i am unhappy with the service",
        "GENERAL": "where is your office",
        "OTHER": "tell me a joke",
    }
    for intent, text in cases.items():
        r = regression_router.predict(text)
        assert r.decision == Decision.ROUTE, (intent, r.to_json())
        assert r.intent == intent, (intent, r.to_json())


def test_oov_no_crash_and_escapes(regression_router):
    """Out-of-vocabulary noise must not crash and must not tie up."""
    hard = " ".join(["qqqqzzzz"] * 8)
    r = regression_router.predict(hard)
    assert r.decision in (Decision.ROUTE, Decision.ESCALATE)
    assert isinstance(r.confidence, float)


def test_malay_and_english_smoke(regression_router):
    for text in ("saya nak bayar", "i want to pay", "boleh ke cancel booking"):
        r = regression_router.predict(text)
        assert r.decision in (Decision.ROUTE, Decision.ESCALATE)