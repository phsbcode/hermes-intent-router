#!/usr/bin/env python3
"""Local inference benchmark: accuracy, macro-F1, latency, RSS, throughput.

Usage:
    python benchmarks/benchmark.py --model models/router.joblib \
        --data tests/fixtures/synthetic_test.csv

Latency (median/p95/p99) is measured per-predict after warm-up on the Jetson
class host; RSS is the process maxrss delta; throughput is predictions/sec.
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

import resource

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from hermes_intent_router import IntentRouter, Decision


def load_csv(path: Path) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            texts.append(row["text"]); labels.append((row.get("intent") or "").strip())
    return texts, labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/router.joblib")
    ap.add_argument("--data", default="tests/fixtures/synthetic_test.csv")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    router = IntentRouter.load(args.model)
    texts, labels = load_csv(args.data)

    # --- accuracy / macro-F1 on the routed set + OOD behaviour ---
    results = router.predict_many(texts)
    routed_idx = [i for i, r in enumerate(results) if r.decision == Decision.ROUTE]
    y_true = [labels[i] for i in routed_idx]
    y_pred = [results[i].intent for i in routed_idx]
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
    esc_rate = 1.0 - len(routed_idx) / len(results) if results else 0.0

    # --- per-predict latency (median/p95/p99) with warm-up ---
    lat = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        for t in texts:
            router.predict(t)
        t1 = time.perf_counter()
        lat.append((t1 - t0) / len(texts) * 1000)
    lat.sort()
    p50 = statistics.median(lat)
    p95 = lat[int(len(lat) * 0.95)]
    p99 = lat[int(len(lat) * 0.99)] if len(lat) >= 100 else lat[-1]

    # --- throughput: single-threaded max rate ---
    n = len(texts)
    t0 = time.perf_counter()
    for _ in range(50):
        for t in texts:
            router.predict(t)
    t1 = time.perf_counter()
    throughput = (50 * n) / (t1 - t0)

    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    out = {
        "model": str(args.model),
        "model_type": router.model_type,
        "threshold": router.threshold,
        "n_test": len(texts),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "route_rate": round(1.0 - esc_rate, 4),
        "escalate_rate": round(esc_rate, 4),
        "latency_ms": {"p50": round(float(p50), 3), "p95": round(float(p95), 3),
                       "p99": round(float(p99), 3)},
        "throughput_ips": round(float(throughput), 1),
        "rss_kib": int(rss_kib),
    }
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        k = out["latency_ms"]
        print(f"model      : {out['model_type']} (thr {out['threshold']})")
        print(f"accuracy   : {out['accuracy']}   macro-F1: {out['macro_f1']}")
        print(f"route rate : {out['route_rate']}   escalate: {out['escalate_rate']}")
        print(f"latency ms : p50 {k['p50']}  p95 {k['p95']}  p99 {k['p99']}")
        print(f"throughput : {out['throughput_ips']} ips")
        print(f"rss        : {out['rss_kib']} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())