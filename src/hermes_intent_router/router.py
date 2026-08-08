"""Public routing API.

    from hermes_intent_router import IntentRouter

    router = IntentRouter.load("models/router.joblib")
    result = router.predict("Saya nak tukar appointment esok")
    result.to_json()  # {"intent": ..., "confidence": ..., "decision": ..., "model": ...}

The router itself never calls an LLM. It returns a structured decision
(ROUTE or ESCALATE) and leaves the downstream choice to the host.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .model import ModelBundle, load_bundle
from .schema import Decision, RoutingResult


class IntentRouter:
    """TF-IDF -> classifier first-stage intent router.

    Thread-safe for concurrent ``predict`` calls once loaded (sklearn
    estimators are read-only after fit).
    """

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle
        # Rebuild the pipeline so `predict` is a single call; the pipeline is
        # only used read-only after construction.
        from sklearn.pipeline import Pipeline

        self._pipe = Pipeline(
            [
                ("tfidf", bundle.vectorizer),
                ("clf", bundle.classifier),
            ]
        )
        self._threshold = float(bundle.confidence_threshold)
        self._intents = list(bundle.intents)
        self._model_type = bundle.model_type

    @classmethod
    def load(cls, path: str | Path) -> "IntentRouter":
        return cls(load_bundle(path))

    @classmethod
    def from_bundle(cls, bundle: ModelBundle) -> "IntentRouter":
        return cls(bundle)

    # -- properties ------------------------------------------------------
    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def intents(self) -> list[str]:
        return list(self._intents)

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def version(self) -> str:
        return self.bundle.version

    # -- prediction ------------------------------------------------------
    def predict(self, text: str) -> RoutingResult:
        """Classify one message and produce the routing decision."""
        if not text or not text.strip():
            raise ValueError("empty message")
        proba = self._pipe.predict_proba([text])[0]
        conf = float(proba.max())
        idx = int(proba.argmax())
        if conf >= self._threshold:
            intent = str(self._intents[idx])
            decision = Decision.ROUTE
        else:
            intent = None
            decision = Decision.ESCALATE
        top = sorted(
            ((float(p), self._intents[i]) for i, p in enumerate(proba)),
            reverse=True,
        )
        return RoutingResult(
            intent=intent,
            confidence=conf,
            decision=decision,
            model=self._model_type,
            top_k=tuple(top[:5]),
        )

    def predict_many(self, texts: Iterable[str]) -> list[RoutingResult]:
        """Classify a batch; used by the benchmark and evaluation paths."""
        return [self.predict(t) for t in texts]

    def predict_proba(self, text: str) -> tuple[np.ndarray, list[str]]:
        """Return raw probabilities + label order (advanced callers only)."""
        proba = self._pipe.predict_proba([text])[0]
        return proba, self._intents