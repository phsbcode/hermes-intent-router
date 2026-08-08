"""hermes-intent-router — lightweight first-stage intent router for Hermes."""

from .model import ModelBundle, build_model, load_bundle
from .router import IntentRouter
from .schema import Decision, RoutingResult
from .guards import (
    BinaryOODGuard,
    ShortAmbiguousGuard,
    LOW_CONFIDENCE,
    SHORT_AMBIGUOUS,
    OOD_CLASSIFIER,
)
from .train import train_from_arrays, train_from_csv

__version__ = "0.1.0"

__all__ = [
    "IntentRouter",
    "ModelBundle",
    "RoutingResult",
    "Decision",
    "BinaryOODGuard",
    "ShortAmbiguousGuard",
    "LOW_CONFIDENCE",
    "SHORT_AMBIGUOUS",
    "OOD_CLASSIFIER",
    "build_model",
    "load_bundle",
    "train_from_csv",
    "train_from_arrays",
    "__version__",
]