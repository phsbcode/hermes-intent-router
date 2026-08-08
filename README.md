# hermes-intent-router

A lightweight, deterministic, first-stage intent router for Hermes.
Winning approach from the Jetson intent-routing benchmark, packaged for real
use: **TF-IDF + MLP** by default, with **logistic regression** retained as a
simpler, more explainable baseline/fallback.

```
incoming message
      |
      v
 TF-IDF classifier
      |
      +-- confidence >= threshold ----------> ROUTE (return intent locally)
      |
      +-- confidence < threshold ----------> ESCALATE (host picks downstream)
```

The router **never invokes an LLM itself**. It returns a structured decision;
Hermes chooses what happens next (local answer, or escalate to a larger
model).

## Why a tiny classifier ahead of an LLM?

A first-stage router for a chat host must be cheap and predictable:

* **Faster** — a TF-IDF + MLP forward pass is ~0.4 ms median on the Jetson
  CPU (see benchmark). An LLM call costs seconds. Routing every message
  through the LLM is wasted latency for the ~90% of messages that are
  unambiguous.
* **Cheaper** — no tokens burned per message at the router stage; LLM compute
  (cloud or local Ollama) is reserved for genuinely ambiguous/OOD messages.
* **Deterministic** — the same message always yields the same intent and
  confidence. LLMs are sampling-based; a router must be stable for
  auditing, caching, and regression tests.
* **Preserves LLM calls for ambiguous cases** — the confidence threshold is
  calibrated so only low-confidence messages escalate, keeping the LLM for
  the messages that actually need judgement.

## Install

```bash
pip install -e .            # Python 3.10+, scikit-learn
hermes-intent-router --help
```

## Quick start

```python
from hermes_intent_router import IntentRouter

router = IntentRouter.load("models/router.joblib")
result = router.predict("Saya nak tukar appointment esok")
print(result.to_json())
# {"intent": "RESCHEDULE", "confidence": 0.9812, "decision": "ROUTE", "model": "mlp"}
```

CLI:

```bash
hermes-intent-router predict "Saya dah buat payment"
hermes-intent-router info --model models/router.joblib
```

## Training (private data stays outside the repo)

Training data format (CSV, headers `text,intent`):

```csv
text,intent
"saya nak buat bayaran sekarang",PAYMENT
"please reschedule my appointment",RESCHEDULE
```

**Private training data must live outside this repository.** Never copy
customer messages into the repo. Point the trainer at your local file:

```bash
hermes-intent-router train --data /path/to/private/local/data.csv \
    --model-type mlp --out models/router.joblib
```

or using the wrapper script:

```bash
python scripts/train_model.py --data /path/to/private/local/data.csv
```

The trained bundle contains only learned weights (vectorizer vocabulary,
network weights) and metadata — it does not contain the source texts. Model
artifacts are gitignored by default. The fingerprint in the metadata is a
non-reversible hash, stored so you can confirm provenance.

## Evaluation

```bash
hermes-intent-router evaluate --data /path/to/test.csv --model models/router.joblib
hermes-intent-router ood --data /path/to/ood.csv --model models/router.joblib
python benchmarks/benchmark.py --model models/router.joblib --data tests/fixtures/synthetic_test.csv
```

## Configuration

* `configs/intents.yaml` — the label space and default `model_type` /
  `confidence_threshold`. Extending intents requires retraining.
* `--model-type mlp|logistic_regression` — MLP is the default (benchmark
  winner); LR is the simpler fallback.
* `--threshold-mode quality|hybrid` and `--min-auto-acc` (or
  `--fallback-acc`) control threshold calibration on validation data:

  * `quality` (default): lowest threshold that keeps local auto-accuracy
    >= `--min-auto-acc` (default 0.99) on validation — max coverage subject
    to a quality guarantee.
  * `hybrid`: maximise estimated end-to-end accuracy given a known fallback
    tier accuracy (`--fallback-acc`); use when the host knows the escalated
    model's expected accuracy.

## Model bundle

One joblib artifact holds everything needed for inference:

```bash
models/router.joblib            # vectorizer + classifier + labels + threshold
models/router.joblib.json       # human-readable metadata (version, metrics)
```

`hermes-intent-router info --model models/router.joblib` prints the metadata.

## OOD / escalation behaviour

Unknown inputs (out-of-domain topics, gibberish) tend to produce low
confidence; the router then returns `decision: "ESCALATE"` with `intent:
null`, and Hermes decides what to do. The threshold is calibrated on
validation data, and tests cover this behaviour (`tests/test_confidence.py`,
`tests/test_regression.py`).

## Benchmark reference

On the original private intent dataset (Aug 2026, 2,400 train / 600 val /
600 test, 10 classes, Malay/English/code-switch):

| model | macro-F1 | median latency (Jetson CPU) |
|---|---|---|
| TF-IDF + MLP | 0.995 | ~0.37 ms |
| TF-IDF + logistic regression | 0.988 | ~0.33 ms |

The repo target: macro-F1 >= ~0.99 on the original held-out set when
retrained with the same methodology, and local inference comfortably below
1 ms median.

## Tests

```bash
pip install -e . pytest
pytest
```

The fixture generator (`tests/fixtures/generate_fixtures.py`) creates
public-safe synthetic data on first use; it is small, deterministic, and
never contains real messages.

## License

MIT. See LICENSE.