"""Lightweight, dependency-free OOD / ambiguity guards that run before the
intent-classifier decision.

These guards are deliberately cheap: they operate on the raw message string
and (optionally) the classifier's probability vector, at sub-millisecond
cost on a Jetson-class CPU. They exist to catch inputs the MLP is not
designed to handle:

  * very short / barely-there messages ("ok", "hi", "yes")
  * general-knowledge questions ("what is the capital of france")
  * number/name/date/amount-only tokens ("250", "John Smith", "rm 500")
  * near-empty or whitespace-only input

Design notes
------------
* No guard is trained on the held-out OOD *test* set (30 samples). The
  heuristic thresholds below were chosen from domain inspection of the
  in-domain data and the OOD *calibration* pool (35 samples) only.
* Each guard is stateless after ``fit`` (it is a no-op for the heuristic),
  so it adds negligible inference overhead.
* The router treats any guard that returns ``True`` for ``reject`` as an
  unconditional escalation, overriding the classifier's top pick.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

# Reason codes surfaced in RoutingResult.reason.
LOW_CONFIDENCE = "LOW_CONFIDENCE"
SHORT_AMBIGUOUS = "SHORT_AMBIGUOUS"
LOW_MARGIN = "LOW_MARGIN"
HIGH_ENTROPY = "HIGH_ENTROPY"

# Minimum token count and minimum character count below which a message is
# treated as too short / ambiguous to route autonomously.
DEFAULT_MIN_TOKENS = 2
DEFAULT_MIN_CHARS = 6

# Factual general-knowledge questions the in-domain corpus has no intent for.
# NOTE: we deliberately match *fact* question patterns, not generic question
# stems (who/what/where...), because in-domain intents themselves use those
# stems ("what programs do you offer", "how much is the book").
_GENERAL_KNOWLEDGE_RE = re.compile(
    r"^\s*(what|how|tell me|give me|where can i find).*"
    r"(capital|weather|temperature|time|date|population|currency|"
    r"distance|height|definition|meaning|recipe|driving.*\blicense\b|"
    r"stock price|share price)\b",
    re.IGNORECASE,
)
_GENERAL_QUESTION_TOO_SHORT = re.compile(
    r"^\s*(what|how|who|where|when|why)\b[\?\!\s]*$",
    re.IGNORECASE,
)

# Tokens that are almost certainly a bare number / amount / name / date.
# The router's in-domain class distribution never includes pure-numeric or
# pure-name messages as actionable intents, so they are escalated.
_AMOUNT_OR_NUMBER_RE = re.compile(
    r"^\s*[\$RM\s]*[0-9]{1,3}(?:[,.][0-9]{3})*(?:\.[0-9]+)?\s*[\$RM\s]*$"
)
_PURE_NUMERIC_RE = re.compile(r"^\s*[0-9\s\-/.,]+\s*$")
_NAMEISH_RE = re.compile(r"^\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s*$")
_DATEISH_RE = re.compile(r"^\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*$")


class ShortAmbiguousGuard:
    """Rejects messages that are too short, ambiguous, or obviously out-of-scope.

    Returns ``True`` from ``reject`` when the message should be escalated before
    the classifier's confidence is trusted.
    """

    def __init__(
        self,
        min_tokens: int = DEFAULT_MIN_TOKENS,
        min_chars: int = DEFAULT_MIN_CHARS,
    ) -> None:
        self.min_tokens = min_tokens
        self.min_chars = min_chars

    def fit(self, texts, *args, **kwargs) -> "ShortAmbiguousGuard":
        return self

    def reject(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return True
        if len(t) < self.min_chars or len(t.split()) < self.min_tokens:
            return True
        if _GENERAL_KNOWLEDGE_RE.match(t) or _GENERAL_QUESTION_TOO_SHORT.match(t):
            return True
        if _AMOUNT_OR_NUMBER_RE.match(t) or _PURE_NUMERIC_RE.match(t):
            return True
        if _NAMEISH_RE.match(t) or _DATEISH_RE.match(t):
            return True
        return False

    def score(self, texts, proba, preds, topk=None) -> np.ndarray:
        return np.array([0.0 if self.reject(t) else 1.0 for t in texts])


class PredictionMarginGuard:
    """Rejects when the top-1 / top-2 probability margin is below a threshold.

    A near-tie means the classifier cannot decide between two intents, which is
    a common OOD / ambiguity signature.
    """

    name = "margin"

    def __init__(self, min_margin: float = 0.10) -> None:
        self.min_margin = min_margin

    def fit(self, texts, proba, preds, topk=None) -> "PredictionMarginGuard":
        return self

    def score(self, texts, proba, preds, topk=None) -> np.ndarray:
        if proba.shape[1] < 2:
            return np.ones(len(texts))
        s = np.sort(proba, axis=1)
        margin = s[:, -1] - s[:, -2]
        return np.clip(margin / self.min_margin, 0, 1)


class EntropyGuard:
    """Rejects when predictive entropy is high relative to the in-domain set."""

    name = "entropy"

    def __init__(self) -> None:
        self.max_entropy: Optional[float] = None

    def fit(self, texts, proba, preds, topk=None) -> "EntropyGuard":
        ent = -np.sum(proba * np.log(proba + 1e-12), axis=1)
        self.max_entropy = float(np.quantile(ent, 0.5))
        return self

    def score(self, texts, proba, preds, topk=None) -> np.ndarray:
        if self.max_entropy is None or self.max_entropy <= 0:
            return np.ones(len(texts))
        ent = -np.sum(proba * np.log(proba + 1e-12), axis=1)
        return np.clip(1.0 - ent / self.max_entropy, 0, 1)
