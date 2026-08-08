"""Training orchestration: CSV -> validated model bundle.

Public entry points:

    from hermes_intent_router.train import train_from_csv
    bundle = train_from_csv("data.csv", model_type="mlp", out_path="models/router.joblib")

CSV format (documented in README):

    text,intent
    "saya nak buat payment sekarang",PAYMENT
    ...

The training CSV is expected to live OUTSIDE the repo (private data). Nothing
here copies or stores the raw text; only the fitted vectorizer/classifier
(and metadata) are persisted into the model bundle.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .confidence import calibrate_threshold
from .model import ModelBundle, build_model, load_bundle


def read_csv(path: str | Path) -> tuple[list[str], list[str]]:
    """Read a text,intent CSV. Skips malformed rows, requires both columns."""
    texts: list[str] = []
    labels: list[str] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "text" not in reader.fieldnames or "intent" not in reader.fieldnames:
            raise ValueError(
                "CSV must have headers: text,intent (case-sensitive)"
            )
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("intent") or "").strip()
            if not text or not label:
                continue
            texts.append(text)
            labels.append(label)
    if not texts:
        raise ValueError(f"no usable rows in {path}")
    return texts, labels


def _fingerprint(texts: list[str], labels: list[str]) -> str:
    """Non-reversible content fingerprint (hash only) for provenance."""
    h = hashlib.sha256()
    for t, l in zip(texts, labels):
        h.update(t.encode("utf-8", "ignore"))
        h.update(b"\0")
        h.update(l.encode("utf-8", "ignore"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def train_from_csv(
    data_path: str | Path,
    model_type: str = "mlp",
    out_path: Optional[str | Path] = None,
    threshold_mode: str = "quality",
    min_auto_acc: float = 0.99,
    fallback_acc: Optional[float] = None,
    random_state: int = 42,
    version: Optional[str] = None,
    test_size: float = 0.2,
    ood_path: Optional[str | Path] = None,
    max_ood_false_accept: float = 0.30,
) -> ModelBundle:
    """Train, validate-split, calibrate threshold, and save a bundle.

    ``ood_path`` (optional) points at a CSV of out-of-distribution texts
    (e.g. off-topic messages). Their softmax probabilities are used at
    calibration time so the ROUTE threshold also rejects OOD inputs, not
    just in-distribution validation samples.

    Returns the ModelBundle (and saves it to ``out_path`` when given).
    """
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    texts, labels = read_csv(data_path)

    # Encode labels to integers: sklearn's MLP early_stopping scoring
    # cannot handle string targets in current versions, and the benchmark
    # methodology used integer labels. We keep the human-readable intents
    # in the bundle, ordered exactly like the classifier's classes_.
    le = LabelEncoder().fit(labels)
    y_all = le.transform(labels)

    X_tr, X_va, y_tr, y_va = train_test_split(
        texts, y_all, test_size=test_size, random_state=random_state,
        stratify=y_all,
    )

    pipe = build_model(model_type, random_state=random_state)
    pipe.fit(X_tr, y_tr)

    proba = pipe.predict_proba(X_va)
    classes_int = list(pipe.classes_)

    ood_proba = None
    if ood_path is not None:
        ood_texts, _ = read_csv(ood_path)
        ood_proba = pipe.predict_proba(ood_texts)

    threshold, calib_metrics = calibrate_threshold(
        proba, y_va.tolist(), classes_int, mode=threshold_mode,
        min_auto_acc=min_auto_acc, fallback_acc=fallback_acc,
        ood_proba=ood_proba, max_ood_false_accept=max_ood_false_accept,
    )

    # Full validation-set report (macro-F1 etc.)
    y_pred = pipe.predict(X_va)
    report = classification_report(
        y_va, y_pred, output_dict=True, zero_division=0
    )
    macro_f1 = report["macro avg"]["f1-score"]

    # Human-readable intents in classifier-class order
    intents = [str(le.classes_[i]) for i in classes_int]

    bundle = ModelBundle(
        version=version or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model_type=model_type,
        intents=intents,
        confidence_threshold=threshold,
        vectorizer=pipe.named_steps["tfidf"],
        classifier=pipe.named_steps["clf"],
        training_meta={
            "source_csv": str(Path(data_path).resolve()),
            "content_fingerprint": _fingerprint(texts, labels),
            "n_rows": len(texts),
            "n_train": len(X_tr),
            "n_validation": len(X_va),
            "random_state": random_state,
            "threshold_mode": threshold_mode,
            "min_auto_acc": min_auto_acc,
            "fallback_acc": fallback_acc,
            "test_size": test_size,
            "ood_csv": str(Path(ood_path).resolve()) if ood_path else None,
            "max_ood_false_accept": max_ood_false_accept,
        },
        eval_metrics={
            "validation_macro_f1": round(float(macro_f1), 4),
            "validation_accuracy": round(float(report["accuracy"]), 4),
            "threshold_calibration": calib_metrics,
        },
    )
    if out_path:
        bundle.save(out_path)
    return bundle


def train_from_arrays(
    texts: list[str],
    labels: list[str],
    model_type: str = "mlp",
    out_path: Optional[str | Path] = None,
    random_state: int = 42,
    test_size: float = 0.2,
    version: Optional[str] = None,
    threshold_mode: str = "quality",
    min_auto_acc: float = 0.99,
    fallback_acc: Optional[float] = None,
    ood_texts: Optional[list[str]] = None,
    max_ood_false_accept: float = 0.30,
) -> ModelBundle:
    """Same as train_from_csv but from in-memory arrays (used by tests)."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "intent"])
        for t, l in zip(texts, labels):
            w.writerow([t, l])
        tmp = fh.name
    ood_tmp = None
    if ood_texts:
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["text", "intent"])
            for t in ood_texts:
                w.writerow([t, "OOD"])
            ood_tmp = fh.name
    try:
        return train_from_csv(
            tmp, model_type=model_type, out_path=out_path,
            random_state=random_state, test_size=test_size, version=version,
            threshold_mode=threshold_mode, min_auto_acc=min_auto_acc,
            fallback_acc=fallback_acc, ood_path=ood_tmp,
            max_ood_false_accept=max_ood_false_accept,
        )
    finally:
        Path(tmp).unlink(missing_ok=True)
        if ood_tmp:
            Path(ood_tmp).unlink(missing_ok=True)