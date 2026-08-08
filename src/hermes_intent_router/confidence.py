"""Confidence scoring and threshold calibration.

The router's decision rule is:

    max(softmax) >= threshold  -> ROUTE  (use the local intent)
    max(softmax) <  threshold  -> ESCALATE (host invokes a downstream model)

Thresholds are calibrated on VALIDATION data, never on the held-out test set.

Two calibration modes are supported:

* ``quality`` (default): choose the LOWEST threshold that keeps the local
  auto-answer accuracy at or above ``min_auto_acc`` on validation. This is
  the safe default for a router whose job is to avoid wrong local answers:
  it routes everything it can while never letting local accuracy drop below
  the bar (max coverage subject to a quality guarantee).
* ``hybrid``: maximise an estimate of end-to-end accuracy,
      auto_rate * auto_acc + (1 - auto_rate) * fallback_acc,
  where ``fallback_acc`` is the expected accuracy of the downstream model
  that handles escalated samples. Use this when the host knows its fallback
  tier well (e.g. an LLM that scores ~0.98 on the same task).

Both modes report the operating point they chose (auto rate, auto accuracy,
escalate rate) so the caller can audit the trade-off.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def calibrate_threshold(
    proba: np.ndarray,
    y: list[str],
    classes: list[str],
    mode: str = "quality",
    min_auto_acc: float = 0.99,
    fallback_acc: Optional[float] = None,
    default_threshold: float = 0.80,
    ood_proba: Optional[np.ndarray] = None,
    max_ood_false_accept: float = 0.30,
) -> tuple[float, dict]:
    """Pick the ROUTE threshold from validation probabilities.

    Returns (threshold, metrics) where metrics describes the operating point:
    threshold, auto_rate, auto_acc, escalate_rate, hybrid_acc (hybrid mode).

    When ``ood_proba`` (an out-of-distribution calibration pool's softmax
    probabilities) is provided, the threshold must additionally keep the
    OOD false-acceptance rate (fraction of OOD samples routed) below
    ``max_ood_false_accept``. This is the standard way to make the router
    reject unknown inputs rather than over-route on them.
    """
    if len(proba) == 0:
        return default_threshold, {"threshold": default_threshold, "auto_rate": 0.0}

    conf = proba.max(axis=1)
    pred_idx = proba.argmax(axis=1)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx.get(v, -1) for v in y])
    correct = pred_idx == y_idx

    ood_conf = ood_proba.max(axis=1) if ood_proba is not None else None

    # Candidate thresholds from unique confidence values, plus the default.
    cands = sorted(set(conf.tolist()) | {default_threshold, 0.5, 0.99})

    def _accepts_ood(thr: float) -> bool:
        if ood_conf is None:
            return True
        if len(ood_conf) == 0:
            return True
        return float((ood_conf >= thr).mean()) <= max_ood_false_accept

    best_thr, best_metric = default_threshold, _operating_point(
        default_threshold, conf, correct
    )

    if mode == "quality":
        # Lowest threshold that still routes at >= min_auto_acc on
        # validation AND keeps OOD false-acceptance within budget. If no
        # candidate meets both, relax to the quality bar alone, then the
        # most accurate point as a last resort.
        valid = [
            (thr, _operating_point(thr, conf, correct))
            for thr in cands
            if _operating_point(thr, conf, correct)["auto_acc"] >= min_auto_acc
            and _accepts_ood(thr)
        ]
        if not valid:
            valid = [
                (thr, _operating_point(thr, conf, correct))
                for thr in cands
                if _operating_point(thr, conf, correct)["auto_acc"] >= min_auto_acc
            ]
        if valid:
            best_thr, best_metric = min(valid, key=lambda t_op: t_op[0])
        else:
            best_thr, best_metric = max(
                ((thr, _operating_point(thr, conf, correct)) for thr in cands),
                key=lambda t_op: t_op[1]["auto_acc"],
            )
    elif mode == "hybrid":
        if fallback_acc is None:
            raise ValueError("hybrid mode requires fallback_acc")
        for thr in cands:
            if not _accepts_ood(thr):
                continue
            op = _operating_point(thr, conf, correct)
            op["hybrid_acc"] = op["auto_rate"] * op["auto_acc"] + (
                1 - op["auto_rate"]
            ) * fallback_acc
            if op["hybrid_acc"] > best_metric.get("hybrid_acc", -1.0):
                best_thr, best_metric = thr, op
        if best_metric.get("hybrid_acc") is None:
            # no candidate met the OOD budget; fall back to any thr
            for thr in cands:
                op = _operating_point(thr, conf, correct)
                op["hybrid_acc"] = op["auto_rate"] * op["auto_acc"] + (
                    1 - op["auto_rate"]
                ) * fallback_acc
                if op["hybrid_acc"] > best_metric.get("hybrid_acc", -1e9):
                    best_thr, best_metric = thr, op
    else:
        raise ValueError(f"unknown calibration mode {mode!r}")

    best_metric["threshold"] = round(float(best_thr), 4)
    if ood_conf is not None and len(ood_conf):
        best_metric["ood_false_accept"] = round(
            float((ood_conf >= best_thr).mean()), 4
        )
    return float(best_thr), best_metric


def _operating_point(thr: float, conf: np.ndarray, correct: np.ndarray) -> dict:
    keep = conf >= thr
    n = len(conf)
    if n == 0:
        return {"auto_rate": 0.0, "auto_acc": 0.0, "escalate_rate": 1.0}
    auto_rate = float(keep.mean())
    auto_acc = float(correct[keep].mean()) if keep.any() else 0.0
    return {
        "auto_rate": round(auto_rate, 4),
        "auto_acc": round(auto_acc, 4),
        "escalate_rate": round(1.0 - auto_rate, 4),
    }


def confidence_from_proba(proba: np.ndarray) -> float:
    """Maximum softmax probability as the routing confidence."""
    return float(proba.max())