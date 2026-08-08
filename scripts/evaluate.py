#!/usr/bin/env python3
"""Thin CLI wrapper for evaluating a router on a labeled CSV.

    python scripts/evaluate.py --data /path/to/test.csv --model models/router.joblib

Equivalent to: hermes-intent-router evaluate [...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hermes_intent_router.cli import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] != "evaluate":
        args = ["evaluate"] + args
    sys.exit(main(args))