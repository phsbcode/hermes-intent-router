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
from .guards import (
    ShortAmbiguousGuard,
    PredictionMarginGuard,
    EntropyGuard,
    BinaryOODGuard,
    LOW_CONFIDENCE,
    SHORT_AMBIGUOUS,
    LOW_MARGIN,
    HIGH_ENTROPY,
    OOD_CLASSIFIER,
    DEFAULT_OOD_THRESHOLD,
)


# Default guard chain used when none is supplied. Only the cheap,
# low-overhead short/ambiguous heuristic is on by default; it catches the
# two documented failure classes ("ok", "what is the capital of france")
# and bare-number / general-knowledge-fact questions with zero added
# in-domain false-escalations. The margin and entropy guards are available
# but OFF by default: they do not reduce held-out OOD false-accept and they
# raise false-escalation of valid in-domain messages.
_DEFAULT_GUARDS = (
    "short_ambiguous",
)


class IntentRouter:
    """TF-IDF -> classifier first-stage intent router.

    Thread-safe for concurrent predict calls once loaded (sklearn
    estimators are read-only after fit).
    """

    def __init__(
        self,
        bundle: ModelBundle,
        *,
        guards: Optional[list[str]] = None,
        ood_model: Optional["BinaryOODGuard"] = None,
    ) -> None:
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
        self._guards = guards if guards is not None else list(_DEFAULT_GUARDS)
        self._short_guard = ShortAmbiguousGuard()
        self._margin_guard = PredictionMarginGuard(min_margin=0.10)
        self._entropy_guard = EntropyGuard()
        # Optional binary OOD detector. Loaded separately from the intent
        # bundle so it can be upgraded independently. Off by default.
        self._ood_guard = ood_model

    @classmethod
    def load(cls, path: str | Path, *, guards: Optional[list[str]] = None, ood_model_path: Optional[str | Path] = None) -> "IntentRouter":
        bundle = load_bundle(path)
        ood_model = BinaryOODGuard.load(ood_model_path) if ood_model_path else None
        return cls(bundle, guards=guards, ood_model=ood_model)

    @classmethod
    def from_bundle(
        cls,
        bundle: ModelBundle,
        *,
        guards: Optional[list[str]] = None,
        ood_model: Optional["BinaryOODGuard"] = None,
    ) -> "IntentRouter":
        return cls(bundle, guards=guards, ood_model=ood_model)

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
        """Classify one message and produce the routing decision.

        Default guard chain (``short_ambiguous``):

        1. short / ambiguous / general-knowledge-fact guard  -> ``SHORT_AMBIGUOUS``
        2. confidence threshold                             -> default ``ROUTE``

        ``margin`` and ``entropy`` guards are available by passing
        ``guards=["short_ambiguous","margin","entropy"]`` to the constructor;
        they are OFF by default (see ``_DEFAULT_GUARDS``) because they do not
        reduce held-out OOD false-accept and raise false-escalation of valid
        in-domain messages.

        Any guard that rejects short-circuits to ``ESCALATE`` with a reason.
        A ``ROUTE`` result never carries a reason.
        """
        if not text or not text.strip():
            raise ValueError("empty message")
        proba = self._pipe.predict_proba([text])[0]
        conf = float(proba.max())
        idx = int(proba.argmax())

        top = sorted(
            ((float(p), self._intents[i]) for i, p in enumerate(proba)),
            reverse=True,
        )

        # 1. short / ambiguous / general-knowledge -- hard escalate
        if "short_ambiguous" in self._guards and self._short_guard.reject(text):
            return RoutingResult(
                intent=None,
                confidence=conf,
                decision=Decision.ESCALATE,
                model=self._model_type,
                top_k=tuple(top[:5]),
                reason=SHORT_AMBIGUOUS,
            )

        # 2. optional binary OOD detector (runs before confidence check)
        if self._ood_guard is not None:
            ood_p = float(self._ood_guard.predict_proba([text])[0])
            if ood_p < self._ood_guard.threshold:
                return RoutingResult(
                    intent=None,
                    confidence=conf,
                    decision=Decision.ESCALATE,
                    model=self._model_type,
                    top_k=tuple(top[:5]),
                    reason=OOD_CLASSIFIER,
                )

        reason = None
        # 2. prediction margin
        if "margin" in self._guards and proba.shape[0] >= 2:
            sorted_proba = np.sort(proba)
            margin = sorted_proba[-1] - sorted_proba[-2]
            if margin < 0.10:
                reason = LOW_MARGIN

        # 3. entropy
        if reason is None and "entropy" in self._guards:
            if not hasattr(self, "_entropy_max"):
                self._entropy_max = float(np.log(max(len(self._intents), 2)))
            ent = -np.sum(proba * np.log(proba + 1e-12))
            if ent > 0.5 * self._entropy_max:
                reason = HIGH_ENTROPY

        if conf >= self._threshold:
            intent = str(self._intents[idx])
            decision = Decision.ROUTE
        else:
            intent = None
            decision = Decision.ESCALATE

        if decision == Decision.ESCALATE and reason is None:
            reason = LOW_CONFIDENCE

        return RoutingResult(
            intent=intent,
            confidence=conf,
            decision=decision,
            model=self._model_type,
            top_k=tuple(top[:5]),
            reason=reason if decision == Decision.ESCALATE else None,
        )

    def predict_many(self, texts: Iterable[str]) -> list[RoutingResult]:
        """Classify a batch; used by the benchmark and evaluation paths."""
        return [self.predict(t) for t in texts]

    def predict_proba(self, text: str) -> tuple[np.ndarray, list[str]]:
        """Return raw probabilities + label order (advanced callers only)."""
        proba = self._pipe.predict_proba([text])[0]
        return proba, self._intents
