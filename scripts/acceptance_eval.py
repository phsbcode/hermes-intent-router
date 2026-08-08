#!/usr/bin/env python3
"""Acceptance evaluation following the benchmark methodology exactly.

The original intent benchmark used a FIXED split: train (2400) / val (600) /
test (600), with threshold calibration on the validation split and final
metrics on the held-out test split. This script reproduces that.

Usage:
    python scripts/acceptance_eval.py \
        --data-dir /path/to/fixed_splits \
        --model-type mlp \
        --out models/acceptance.joblib

The data directory must contain train.csv, val.csv, test.csv (and
optionally ood.csv), each with text,intent headers. PRIVATE DATA: keep the
CSV files outside the repo; only the model bundle is written.

Acceptance bars (from the original benchmark, Aug 2026):
    macro-F1  >= ~0.99 on the original held-out test set
    median latency < 1 ms per prediction on Jetson CPU
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import accuracy_score, classification_report, f1_score

from hermes_intent_router.confidence import calibrate_threshold
from hermes_intent_router.model import ModelBundle, build_model


def load_csv(path: Path) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            texts.append(row["text"]); labels.append((row.get("intent") or "").strip())
    return texts, labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-type", default="mlp", choices=["mlp", "logistic_regression"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-auto-acc", type=float, default=0.99)
    ap.add_argument("--max-ood-false-accept", type=float, default=0.30)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    d = Path(args.data_dir)
    X_tr, y_tr = load_csv(d / "train.csv")
    X_va, y_va = load_csv(d / "val.csv")
    X_te, y_te = load_csv(d / "test.csv")

    # Integer labels (matches the original benchmark methodology and avoids
    # sklearn MLP early_stopping string-target limitation).
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(y_tr + y_va + y_te)
    y_tr_i = le.transform(y_tr)
    y_va_i = le.transform(y_va)
    y_te_i = le.transform(y_te)

    pipe = build_model(args.model_type, random_state=42)
    t0 = time.time()
    pipe.fit(X_tr, y_tr_i)
    fit_seconds = time.time() - t0

    proba_va = pipe.predict_proba(X_va)
    classes_int = list(pipe.classes_)

    ood_proba = None
    ood_cal = d / "ood_calibrate.csv"
    if ood_cal.exists():
        ood_texts_cal, _ = load_csv(ood_cal)
        ood_proba = pipe.predict_proba(ood_texts_cal)

    threshold, calib = calibrate_threshold(
        proba_va, y_va_i.tolist(), classes_int, mode="quality",
        min_auto_acc=args.min_auto_acc, ood_proba=ood_proba,
        max_ood_false_accept=args.max_ood_false_accept,
    )

    pred_te = pipe.predict(X_te)
    report = classification_report(y_te_i, pred_te, output_dict=True, zero_division=0)
    macro_f1 = report["macro avg"]["f1-score"]
    acc = accuracy_score(y_te_i, pred_te)

    intents = [str(le.classes_[i]) for i in classes_int]

    # latency on the acceptance target (single thread, warm)
    lat = []
    for _ in range(5):
        t0 = time.time()
        for x in X_te:
            pipe.predict([x])
        lat.append((time.time() - t0) / len(X_te) * 1000)
    lat.sort()
    p50 = lat[len(lat) // 2]

    bundle = ModelBundle(
        version=f"accept-{args.model_type}",
        created_utc=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        model_type=args.model_type,
        intents=intents,
        confidence_threshold=threshold,
        vectorizer=pipe.named_steps["tfidf"],
        classifier=pipe.named_steps["clf"],
        training_meta={
            "n_train": len(X_tr), "n_validation": len(X_va), "n_test": len(X_te),
            "train_seconds": round(fit_seconds, 3),
        },
        eval_metrics={
            "test_macro_f1": round(float(macro_f1), 4),
            "test_accuracy": round(float(acc), 4),
            "threshold_calibration": calib,
            "latency_ms_p50": round(float(p50), 4),
        },
    )
    bundle.save(args.out)

    out = {
        "model_type": args.model_type,
        "n_train": len(X_tr), "n_val": len(X_va), "n_test": len(X_te),
        "threshold": round(float(threshold), 4),
        "calibration": calib,
        "test_macro_f1": round(float(macro_f1), 4),
        "test_accuracy": round(float(acc), 4),
        "latency_ms_p50": round(float(p50), 3),
        "train_seconds": round(fit_seconds, 3),
        "bundle": str(Path(args.out).resolve()),
        "bars": {"macro_f1_ge": 0.99, "latency_ms_lt": 1.0},
    }
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"model_type : {out['model_type']}")
        print(f"threshold  : {out['threshold']}")
        print(f"macro-F1   : {out['test_macro_f1']}  (bar >= 0.99)")
        print(f"accuracy   : {out['test_accuracy']}")
        print(f"latency    : {out['latency_ms_p50']} ms p50  (bar < 1.0)")
        print(f"train      : {out['train_seconds']} s")
        print(f"bundle     : {out['bundle']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())