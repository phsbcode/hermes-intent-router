"""Output schema for the intent router.

The decision payload is a plain, JSON-serialisable structure so Hermes (or
any other host) can decide what to do with the result:

    ROUTE   -> the local classifier is confident; use `intent`.
    ESCALATE -> the classifier is not confident; Hermes must invoke a
                downstream (typically larger) model. `intent` is null.

Nothing here depends on the ML stack, so callers can depend on `schema`
alone and stay decoupled from sklearn.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class Decision(str, Enum):
    ROUTE = "ROUTE"
    ESCALATE = "ESCALATE"


@dataclass(frozen=True)
class RoutingResult:
    """Structured decision produced for one message.

    The ``reason`` field (when set) explains *why* the router chose to escalate.
    It is optional and omitted from the JSON output when ``None`` so existing
    callers that parse ``intent``/``confidence``/``decision``/``model`` continue
    to work. Values:

        LOW_CONFIDENCE      - classifier confidence below the decision threshold
        SHORT_AMBIGUOUS     - message rejected by the short/ambiguity guard
                              (e.g. "ok", "hi", general-knowledge questions)
        LOW_MARGIN          - top-1 and top-2 probabilities nearly tied
        HIGH_ENTROPY        - predictive distribution too flat

    A ``ROUTE`` result never carries a reason.
    """

    intent: Optional[str]
    confidence: float
    decision: Decision
    model: str
    # Optional diagnostics (not part of the core wire contract, kept for tooling)
    top_k: tuple = field(default_factory=tuple, repr=False, compare=False)
    reason: Optional[str] = field(default=None, repr=False, compare=False)

    def to_json(self) -> str:
        """JSON exactly matching the documented output format.

        ``reason`` is only included when set, keeping the payload backward-compatible
        with callers that expect a fixed key set for ``ROUTE`` decisions.
        """
        payload: dict[str, Any] = {
            "intent": self.intent,
            "confidence": round(float(self.confidence), 4),
            "decision": self.decision.value,
            "model": self.model,
        }
        if self.reason is not None:
            payload["reason"] = self.reason
        return json.dumps(payload)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "intent": self.intent,
            "confidence": round(float(self.confidence), 4),
            "decision": self.decision.value,
            "model": self.model,
        }
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "RoutingResult":
        return cls(
            intent=data.get("intent"),
            confidence=float(data["confidence"]),
            decision=Decision(data["decision"]),
            model=data["model"],
            top_k=tuple(data.get("top_k", ())),
            reason=data.get("reason"),
        )