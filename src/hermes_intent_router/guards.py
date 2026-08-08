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
OOD_CLASSIFIER = "OOD_CLASSIFIER"

# Threshold for the BinaryOODGuard in-domain probability below which a message
# is escalated as out-of-domain. Tuned on the calibration dev split only.
DEFAULT_OOD_THRESHOLD = 0.6848

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


class BinaryOODGuard:
    """Binary in-domain-vs-OOD classifier.

    Trained on in-domain texts (label 1) and out-of-domain texts (label 0),
    then exposed to the router via ``predict_proba``. A message whose
    in-domain probability falls below ``threshold`` is escalated with reason
    ``OOD_CLASSIFIER``.

    The detector is **optional** and **off by default**. Its artifact is
    loaded separately from the intent model bundle (via ``ood_model_path`` on
    ``IntentRouter``) so that the existing bundle stays unchanged and the
    guard can be upgraded independently.

    Implementation detail: a char-word TF-IDF union + LogisticRegression
    (char_wb n-grams 2-5 dominate because they capture subword morphology
    that separates Malay/English code-switch from in-domain finance phrasing).
    """
    name = "binary_ood"

    def __init__(self, threshold: float = DEFAULT_OOD_THRESHOLD) -> None:
        self.threshold = threshold
        self.vec = None
        self.clf = None

    def fit(self, in_domain_texts, ood_texts, random_state: int = 42, variant: str = "char", min_df: int = 1) -> "BinaryOODGuard":
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import FeatureUnion
        except ImportError:
            raise ImportError("BinaryOODGuard requires scikit-learn")
        if variant == "char":
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                       min_df=min_df, sublinear_tf=True)
        elif variant == "word":
            self.vec = TfidfVectorizer(token_pattern=r"(?u)\b\w[\w']+\b",
                                       ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)
        elif variant == "word_char":
            self.vec = FeatureUnion([
                ("w", TfidfVectorizer(token_pattern=r"(?u)\b\w[\w']+\b", ngram_range=(1, 2),
                                      min_df=min_df, sublinear_tf=True)),
                ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                      min_df=min_df, sublinear_tf=True)),
            ])
        else:
            raise ValueError(f"unknown variant: {variant}")
        self.variant = variant
        self.clf = LogisticRegression(C=1.0, class_weight="balanced",
                                    max_iter=2000, random_state=random_state)
        X_texts = list(in_domain_texts) + list(ood_texts)
        y = [1] * len(in_domain_texts) + [0] * len(ood_texts)
        X = self.vec.fit_transform(X_texts)
        self.clf.fit(X, y)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        """Return in-domain probability (column 1) for each text."""
        if self.vec is None or self.clf is None:
            return np.ones(len(texts))
        X = self.vec.transform(list(texts))
        return self.clf.predict_proba(X)[:, 1]

    def score(self, texts, proba, preds, topk=None) -> np.ndarray:
        """Return in-domain probability; below ``threshold`` means OOD."""
        return self.predict_proba(texts)

    def reject(self, text: str) -> bool:
        return bool(self.predict_proba([text])[0] < self.threshold)

    @classmethod
    def load(cls, path: str) -> "BinaryOODGuard":
        import joblib
        data = joblib.load(path)
        g = cls.__new__(cls)
        g.threshold = data.get("threshold", DEFAULT_OOD_THRESHOLD)
        g.variant = data.get("variant", "char")
        g.vec = data["vec"]
        g.clf = data["clf"]
        return g

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({"vec": self.vec, "clf": self.clf,
                     "threshold": self.threshold,
                     "variant": getattr(self, "variant", "char")}, path)
