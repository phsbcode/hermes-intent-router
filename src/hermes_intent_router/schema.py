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
    """Structured decision produced for one message."""

    intent: Optional[str]
    confidence: float
    decision: Decision
    model: str
    # Optional diagnostics (not part of the wire contract, kept for tooling)
    top_k: tuple = field(default_factory=tuple, repr=False, compare=False)

    def to_json(self) -> str:
        """JSON exactly matching the documented output format."""
        payload: dict[str, Any] = {
            "intent": self.intent,
            "confidence": round(float(self.confidence), 4),
            "decision": self.decision.value,
            "model": self.model,
        }
        return json.dumps(payload)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "intent": self.intent,
            "confidence": round(float(self.confidence), 4),
            "decision": self.decision.value,
            "model": self.model,
        }
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "RoutingResult":
        return cls(
            intent=data.get("intent"),
            confidence=float(data["confidence"]),
            decision=Decision(data["decision"]),
            model=data["model"],
        )