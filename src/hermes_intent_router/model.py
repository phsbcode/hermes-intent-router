"""Model construction and persistence.

Two classifier families are supported (selected by config):

* ``mlp``                - sklearn MLPClassifier on TF-IDF features. Default.
                           This is the benchmark-winning model (macro-F1
                           0.995 on the original held-out intent set).
* ``logistic_regression`` - sklearn LogisticRegression. Retained as a simple,
                           cheaper, more explainable fallback (0.988 macro-F1
                           on the same set).

Model bundles are joblib files holding a small dataclass with the vectorizer,
the classifier, the label space, the calibrated threshold, and training /
evaluation metadata. Bundles are versioned by the caller via the ``version``
argument and the ``created_utc`` timestamp.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

SUPPORTED_MODEL_TYPES = ("mlp", "logistic_regression")

MODEL_ALIASES = {
    "mlp": "mlp",
    "logistic_regression": "logistic_regression",
    "logreg": "logistic_regression",
    "lr": "logistic_regression",
}


@dataclass
class ModelBundle:
    """Single versioned artifact containing everything needed for inference."""

    version: str
    created_utc: str
    model_type: str
    intents: list[str]
    confidence_threshold: float
    vectorizer: TfidfVectorizer
    classifier: Any  # sklearn estimator
    training_meta: dict[str, Any] = field(default_factory=dict)
    eval_metrics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        # Sidecar JSON for human-readable metadata (never the weights).
        sidecar = Path(path).with_suffix(".json")
        sidecar.write_text(
            __import__("json").dumps(
                {
                    "version": self.version,
                    "created_utc": self.created_utc,
                    "model_type": self.model_type,
                    "intents": self.intents,
                    "confidence_threshold": self.confidence_threshold,
                    "training_meta": self.training_meta,
                    "eval_metrics": self.eval_metrics,
                    "extra": self.extra,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        return joblib.load(path)


def build_model(model_type: str, random_state: int = 42) -> Pipeline:
    """Build the TF-IDF -> classifier pipeline for the given model type."""
    mt = MODEL_ALIASES.get(model_type, model_type)
    if mt not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type {model_type!r}; "
            f"expected one of {SUPPORTED_MODEL_TYPES}"
        )
    if mt == "mlp":
        classifier: Any = MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            max_iter=400,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.2,
            random_state=random_state,
        )
    else:
        classifier = LogisticRegression(
            C=3.0,
            max_iter=2000,
            solver="lbfgs",
            random_state=random_state,
        )
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                      sublinear_tf=True, min_df=2,
                                      lowercase=True, strip_accents=None)),
            ("clf", classifier),
        ]
    )


def train_model(
    texts: list[str],
    labels: list[str],
    model_type: str = "mlp",
    random_state: int = 42,
    version: str = "dev",
    training_meta: Optional[dict[str, Any]] = None,
) -> ModelBundle:
    """Train a complete router bundle from raw text/label lists.

    The confidence threshold is calibrated on a validation split inside this
    function (see confidence.calibrate_threshold). Callers that need precise
    split control (e.g. benchmark methodology) should build the pipeline with
    build_model() and assemble the bundle manually.
    """
    from sklearn.model_selection import train_test_split

    from .confidence import calibrate_threshold

    X_tr, X_va, y_tr, y_va = train_test_split(
        texts, labels, test_size=0.2, random_state=random_state, stratify=labels
    )
    pipe = build_model(model_type, random_state=random_state)
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    train_seconds = time.time() - t0

    proba = pipe.predict_proba(X_va)
    classes = list(pipe.classes_)
    threshold, metrics = calibrate_threshold(proba, y_va, classes)

    bundle = ModelBundle(
        version=version,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model_type=model_type,
        intents=list(classes),
        confidence_threshold=threshold,
        vectorizer=pipe.named_steps["tfidf"],
        classifier=pipe.named_steps["clf"],
        training_meta={
            "n_train": len(X_tr),
            "n_validation": len(X_va),
            "train_seconds": round(train_seconds, 3),
            "random_state": random_state,
            **(training_meta or {}),
        },
        eval_metrics=metrics,
    )
    return bundle


def load_bundle(path: str | Path) -> ModelBundle:
    return ModelBundle.load(path)