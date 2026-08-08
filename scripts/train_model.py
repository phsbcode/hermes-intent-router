#!/usr/bin/env python3
"""Thin CLI wrapper for training a router on an EXTERNAL private CSV.

    python scripts/train_model.py --data /path/to/private/data.csv \
        --model-type mlp --out models/router.joblib

Equivalent to: hermes-intent-router train [...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hermes_intent_router.cli import main

if __name__ == "__main__":
    # forward everything but the program name
    args = sys.argv[1:]
    if args and args[0] != "train":
        args = ["train"] + args
    sys.exit(main(args))