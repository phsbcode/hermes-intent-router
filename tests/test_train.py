"""Training pipeline tests: CSV parsing, model types, bundle round-trip."""

import csv

import joblib
import pytest

from hermes_intent_router.model import build_model, load_bundle
from hermes_intent_router.train import read_csv, train_from_csv


def test_read_csv_requires_headers(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\nx,y\n")
    with pytest.raises(ValueError):
        read_csv(bad)


def test_read_csv_skips_empty(tmp_path):
    good = tmp_path / "good.csv"
    good.write_text("text,intent\n\"hello there\",GENERAL\n,PAYMENT\n\"bye\",COMPLAINT\n")
    t, l = read_csv(good)
    assert t == ["hello there", "bye"]
    assert l == ["GENERAL", "COMPLAINT"]


def test_build_model_types():
    for mt in ("mlp", "logistic_regression"):
        pipe = build_model(mt)
        assert pipe is not None
    with pytest.raises(ValueError):
        build_model("dendritron")  # no Dendritron support


def test_train_mlp_and_lr(synthetic_train, tmp_path):
    for mt, out_name in (("mlp", "mlp.joblib"), ("logistic_regression", "lr.joblib")):
        out = tmp_path / out_name
        bundle = train_from_csv(synthetic_train, model_type=mt, out_path=out,
                                version="test-" + mt)
        assert bundle.model_type == mt
        assert len(bundle.intents) == 10
        assert bundle.confidence_threshold > 0
        loaded = load_bundle(out)
        assert loaded.intents == bundle.intents


def test_bundle_contains_expected_components(synthetic_train, tmp_path):
    out = tmp_path / "b.joblib"
    bundle = train_from_csv(synthetic_train, model_type="mlp", out_path=out)
    assert bundle.vectorizer is not None
    assert bundle.classifier is not None
    assert "training_meta" in bundle.__dict__
    assert "eval_metrics" in bundle.__dict__
    assert set(bundle.intents) == {
        "PAYMENT", "RESCHEDULE", "CANCEL", "MENTORING_ENQUIRY",
        "PROGRAM_ENQUIRY", "BOOK_PURCHASE", "UNIT_TRUST", "COMPLAINT",
        "GENERAL", "OTHER",
    }


def test_cannot_train_private_data_into_repo(synthetic_train, tmp_path):
    """The trainer writes only the bundle; raw texts never appear on disk
    except in the caller-provided CSV. We verify the bundle does NOT
    serialize the source texts (only learned weights)."""
    out = tmp_path / "check.joblib"
    bundle = train_from_csv(synthetic_train, model_type="mlp", out_path=out)
    data = joblib.load(out)
    raw = str(data.__dict__)
    # bundle should not contain the string "spider in my room" (fixture text)
    assert "spider in my room" not in raw
    assert bundle.classifier is not None