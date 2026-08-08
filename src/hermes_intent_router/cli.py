"""Command-line interface.

Commands:

    hermes-intent-router predict "Saya dah buat payment"
    hermes-intent-router train   --data /path/to/private/local/data.csv
    hermes-intent-router evaluate --data /path/to/test.csv
    hermes-intent-router ood     --data /path/to/ood.csv
    hermes-intent-router info    [--model models/router.joblib]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .router import IntentRouter
from .train import read_csv, train_from_csv

DEFAULT_MODEL = Path("models/router.joblib")


def _load_model(args) -> IntentRouter:
    model_path = getattr(args, "model", None) or DEFAULT_MODEL
    if not Path(model_path).exists():
        print(
            f"model not found: {model_path}\n"
            "train one first, e.g.: hermes-intent-router train --data data.csv",
            file=sys.stderr,
        )
        sys.exit(2)
    return IntentRouter.load(model_path)


def cmd_predict(args) -> int:
    router = _load_model(args)
    result = router.predict(args.text)
    print(result.to_json())
    return 0


def cmd_train(args) -> int:
    print(f"training {args.model_type} on {args.data} ...", file=sys.stderr)
    bundle = train_from_csv(
        args.data,
        model_type=args.model_type,
        out_path=args.out,
        threshold_mode=args.threshold_mode,
        min_auto_acc=args.min_auto_acc,
        fallback_acc=args.fallback_acc,
        random_state=args.random_state,
        version=args.version,
        ood_path=args.ood_data,
        max_ood_false_accept=args.max_ood_false_accept,
    )
    print(json.dumps({
        "out_path": str(Path(args.out).resolve()),
        "model_type": bundle.model_type,
        "intents": bundle.intents,
        "confidence_threshold": bundle.confidence_threshold,
        "eval_metrics": bundle.eval_metrics,
        "training_meta": bundle.training_meta,
    }, indent=2, default=str))
    return 0


def cmd_evaluate(args) -> int:
    router = _load_model(args)
    texts, labels = read_csv(args.data)
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    preds = router.predict_many(texts)
    routed = np.array([r.decision.value == "ROUTE" for r in preds])
    y_true = np.array(labels)
    y_pred = np.array([r.intent if r.intent is not None else "ESCALATED" for r in preds])
    acc_all = accuracy_score(y_true, [p for p, r in zip(y_pred, preds) if r.intent is not None] or y_true)
    # For macro-F1 on all rows, ESCALATED rows count as wrong
    macro_f1_all = f1_score(y_true, y_pred, average="macro", zero_division=0)
    local_idx = [i for i, r in enumerate(preds) if r.intent is not None]
    macro_f1_local = (
        f1_score(y_true[local_idx], [y_pred[i] for i in local_idx], average="macro", zero_division=0)
        if local_idx else 0.0
    )
    out = {
        "n": len(texts),
        "accuracy_all": round(float(acc_all), 4),
        "macro_f1_all": round(float(macro_f1_all), 4),
        "macro_f1_local_only": round(float(macro_f1_local), 4),
        "route_rate": round(float(routed.mean()), 4),
        "escalate_rate": round(float((~routed).mean()), 4),
        "model": router.model_type,
        "threshold": router.threshold,
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_ood(args) -> int:
    """Report how OOD inputs behave: escalate rate, mean confidence."""
    router = _load_model(args)
    texts, _ = read_csv(args.data)
    preds = router.predict_many(texts)
    confs = [r.confidence for r in preds]
    esc = sum(1 for r in preds if r.decision.value == "ESCALATE")
    out = {
        "n": len(texts),
        "escalate_rate": round(esc / len(texts), 4),
        "mean_confidence": round(sum(confs) / len(confs), 4),
        "min_confidence": round(min(confs), 4),
        "max_confidence": round(max(confs), 4),
        "threshold": router.threshold,
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_info(args) -> int:
    router = _load_model(args)
    print(json.dumps({
        "version": router.version,
        "model_type": router.model_type,
        "intents": router.intents,
        "confidence_threshold": router.threshold,
        "training_meta": router.bundle.training_meta,
        "eval_metrics": router.bundle.eval_metrics,
    }, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hermes-intent-router", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("predict", help="classify one message and print decision JSON")
    sp.add_argument("text")
    sp.add_argument("--model", default=None)
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("train", help="train a model bundle from a text,intent CSV")
    sp.add_argument("--data", required=True, help="path to private CSV (text,intent)")
    sp.add_argument("--model-type", dest="model_type", default="mlp",
                    choices=["mlp", "logistic_regression"])
    sp.add_argument("--out", default=DEFAULT_MODEL)
    sp.add_argument("--threshold-mode", dest="threshold_mode", default="quality",
                    choices=["quality", "hybrid"])
    sp.add_argument("--min-auto-acc", dest="min_auto_acc", type=float, default=0.99)
    sp.add_argument("--fallback-acc", dest="fallback_acc", type=float, default=None)
    sp.add_argument("--random-state", dest="random_state", type=int, default=42)
    sp.add_argument("--version", default=None)
    sp.add_argument("--ood-data", dest="ood_data", default=None,
                    help="CSV of out-of-distribution texts for threshold calibration")
    sp.add_argument("--max-ood-false-accept", dest="max_ood_false_accept",
                    type=float, default=0.30,
                    help="max OOD fraction allowed to route during calibration")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("evaluate", help="evaluate a model bundle on a labeled CSV")
    sp.add_argument("--data", required=True)
    sp.add_argument("--model", default=None)
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("ood", help="OOD/escalation behaviour on a CSV")
    sp.add_argument("--data", required=True)
    sp.add_argument("--model", default=None)
    sp.set_defaults(func=cmd_ood)

    sp = sub.add_parser("info", help="show model bundle metadata")
    sp.add_argument("--model", default=None)
    sp.set_defaults(func=cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())