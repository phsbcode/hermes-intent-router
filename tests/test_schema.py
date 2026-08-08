"""Schema contract tests: the JSON output format is the wire contract."""

import json

import pytest

from hermes_intent_router import Decision, RoutingResult


def test_route_payload_shape():
    r = RoutingResult(
        intent="PAYMENT", confidence=0.982, decision=Decision.ROUTE, model="mlp"
    )
    payload = json.loads(r.to_json())
    assert payload == {
        "intent": "PAYMENT",
        "confidence": 0.982,
        "decision": "ROUTE",
        "model": "mlp",
    }


def test_escalate_payload_shape():
    r = RoutingResult(
        intent=None, confidence=0.61, decision=Decision.ESCALATE, model="mlp"
    )
    payload = json.loads(r.to_json())
    assert payload == {
        "intent": None,
        "confidence": 0.61,
        "decision": "ESCALATE",
        "model": "mlp",
    }


def test_json_keys_are_exact():
    """No extra diagnostic keys may leak into the wire format."""
    r = RoutingResult(
        intent="CANCEL", confidence=0.9, decision=Decision.ROUTE, model="mlp",
        top_k=(("CANCEL", 0.9),),
    )
    assert set(json.loads(r.to_json()).keys()) == {
        "intent", "confidence", "decision", "model"
    }


def test_confidence_rounding():
    r = RoutingResult(
        intent="OTHER", confidence=0.9123456, decision=Decision.ROUTE, model="lr"
    )
    assert json.loads(r.to_json())["confidence"] == 0.9123


def test_decision_enum():
    assert Decision.ROUTE.value == "ROUTE"
    assert Decision.ESCALATE.value == "ESCALATE"


def test_from_dict_roundtrip():
    d = {"intent": "PAYMENT", "confidence": 0.5, "decision": "ESCALATE",
         "model": "mlp"}
    r = RoutingResult.from_dict(d)
    assert r.intent is None or r.intent == "PAYMENT"
    assert r.decision == Decision.ESCALATE
    assert json.loads(r.to_json())["decision"] == "ESCALATE"