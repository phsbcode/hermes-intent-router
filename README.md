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
# {"intent": "RESCHEDULE", "confidence": 0.8141, "decision": "ROUTE", "model": "mlp"}
# ESCALATE results add a "reason" key, e.g.:
# router.predict("ok").to_json()
# {"intent": null, "confidence": 0.454, "decision": "ESCALATE", "model": "mlp", "reason": "SHORT_AMBIGUOUS"}
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

## OOD / escalation behaviour — and limitations

Unknown inputs (out-of-distribution topics, short/ambiguous messages) tend
to produce lower confidence; the router then returns `decision: "ESCALATE"`
with `intent: null`, and Hermes decides what to do.

**Final measured threshold and routing behaviour (held-out evaluation):**

| metric | result |
|---|---|
| Final confidence threshold | 0.3621 |
| In-distribution test macro-F1 | 0.9950 |
| In-distribution routing coverage | 98.33% auto-routed |
| Routed-message (auto-answered) accuracy | 0.9983 |
| OOD false-accept rate (held-out OOD test) | 0.40 |
| OOD escalation rate (held-out OOD test) | 0.60 |
| OOD test pool size | 30 samples |

**This is NOT production-grade OOD rejection.** The held-out OOD pool is only
30 samples, so the 40% false-accept / 60% escalation numbers are noisy and
should not be relied on for safety-critical routing.

**Guard layer (added after the original acceptance run).** A lightweight
`short_ambiguous` guard runs before the classifier decision. It catches:

* very short / barely-there messages ("ok", "hi", "yes")
* general-knowledge fact questions ("what is the capital of france",
  "what time is it", "what is the weather like")
* bare-number / name / date / amount-only tokens ("250", "John Smith")

With this guard on the default chain, the two previously-documented failure
examples now **escalate** with `reason: "SHORT_AMBIGUOUS"`:

* "what is the capital of france" -> `ESCALATE`, reason `SHORT_AMBIGUOUS`
  (previously routed as `PROGRAM_ENQUIRY`, confidence ~0.587)
* "ok" -> `ESCALATE`, reason `SHORT_AMBIGUOUS`
  (previously routed as `COMPLAINT`, confidence ~0.454)

This guard adds **zero** in-domain false-escalations on the held-out
in-distribution test (10 escalations, identical to the no-guard baseline;
all reasoned `LOW_CONFIDENCE`). However, it does **not** bring the held-out
OOD false-accept rate to the 10% target: question-formatted but
non-factual OOD messages (e.g. "what is your favorite color",
"do you know the flight to auckland delays") are structurally close to
in-domain questions and still route when their confidence exceeds the
0.3621 threshold.

### Optional BinaryOODGuard (experimental, off by default)

A dedicated binary in-domain-vs-OOD classifier can be attached to the router
to reject OOD traffic *before* the confidence check. It is **optional and
off by default** — the default chain remains `["short_ambiguous"]`, so the
existing bundle and API are unchanged.

Training data is a private in-domain set (label 1) plus a public-safe
synthetic OOD corpus (label 0). The default artifact (`models/ood_guard.joblib`)
uses char-n-gram TF-IDF (2-5) + logistic regression, ~283 KiB, and is
gitignored like all model artifacts.

Attach it at load time:

```python
router = IntentRouter.load(
    "models/acceptance_mlp.joblib",
    guards=["short_ambiguous"],          # optional; keep or drop
    ood_model_path="models/ood_guard.joblib",   # optional OOD detector
)
result = router.predict("what is your favorite color")
# {"intent": null, "confidence": 0.435, "decision": "ESCALATE",
#  "model": "mlp", "reason": "OOD_CLASSIFIER"}
```

The OOD classifier's probability is **not** the intent confidence; the two
are kept separate (the router reports both when you need them via
`router.predict`'s `top_k` and the guard's own score).

**Measured held-out impact (frozen 600 in-domain test + 30 OOD test,
threshold tuned on the calibration dev split only):**

| pipeline | macro-F1 | coverage | routed acc | OOD false-accept |
|---|---|---|---|---|
| A: router only | 0.8998 | 98.33% | 99.83% | 0.400 (12/30) |
| B: short + router | 0.8998 | 98.33% | 99.83% | 0.400 (12/30) |
| C: BinaryOOD + router | 0.8871 | 95.67% | 99.83% | 0.067 (2/30) |
| D: short + BinaryOOD + router | 0.8871 | 95.67% | 99.83% | 0.067 (2/30) |

C and D bring OOD false-accept from 40% to **6.7%** (≤10% target met) while
keeping in-domain coverage at 95.67% (≥95% target) and routed-message
accuracy at 99.83%. Macro-F1 drops from 0.8998 to 0.8871 because ~16 more
genuinely ambiguous in-domain messages now escalate (the intended safety
tradeoff). Calibration quality on the dev split: AUROC 0.9991, AUPRC 0.9987,
threshold 0.6848 (in-domain false-reject ≤ 2% on caldev).

**Known remaining weakness:** one held-out OOD sample still routes
("is the pool open in the morning", a question-shaped message that
lexically resembles PROGRAM_ENQUIRY and gets in-domain probability 0.792);
and the added transform pushes p50 latency to ~1.87 ms (target < 1.5 ms).
Neither is a system failure, but they are the practical limit of a cheap
TF-IDF gate.

A `ROUTE` decision means the classifier is *locally confident*, not that the
input is *in-domain*. Callers must not treat `ROUTE` as proof that an input
belongs to the known intent set.

## Current production guidance

Recommended caller behaviour:

```
incoming message
      |
      v
intent router
      |
      +-- clear in-domain + high confidence --> ROUTE
      |
      +-- short / ambiguous / unknown / low confidence --> ESCALATE
```

Treat the router as a fast triage layer, never as a gatekeeper of domain
membership. Before any autonomous production routing that depends on
out-of-domain rejection, add a second guard (e.g. an explicitly trained OOD
detector, or a higher confidence gate on borderline samples). The bundled
threshold is safe for the measured in-distribution distribution; it was
calibrated for local-accuracy quality on validation data, with the OOD
budget applied as a secondary constraint on a small calibration pool.

## Why a tiny classifier ahead of an LLM?

A first-stage router for a chat host must be cheap and predictable:

* **Fast** — a TF-IDF + MLP forward pass is ~0.9 ms median per predict on
  the Jetson (see benchmark). An LLM call costs seconds. Routing every
  message through the LLM is wasted latency for the ~98% of held-out
  in-distribution messages that are unambiguous at the local tier.
* **Deterministic** — the same message always yields the same intent and
  confidence. LLMs are sampling-based; a router must be stable for
  auditing, caching, and regression tests.
* **Preserves LLM calls for ambiguous cases** — the confidence threshold is
  calibrated so only low-confidence messages escalate, keeping the LLM for
  the messages that actually need judgement.

Token/accuracy trade-off note: these savings are tied to the measured
held-out in-distribution set. On out-of-domain traffic the router escalates
more aggressively (see OOD limitation above), which is the intended safety
behaviour but not a general "90% bypass" claim for arbitrary input.

## Benchmark reference

Two measurements are reported below. Keep them separate:

* **Earlier experimental classifier benchmark** — measured on the original
  private intent dataset (Aug 2026, 2400 train / 600 val / 600 test, 10
  classes, Malay/English/code-switch), single-pass per-call latency on a
  Jetson Orin Nano Super CPU. These numbers characterise the underlying
  classifier path and are NOT the final packaged router measurement:

|| model | macro-F1 | median latency (Jetson CPU) |
||---|---|---|---|
|| TF-IDF + MLP | 0.995 | ~0.37 ms |
|| TF-IDF + logistic regression | 0.988 | ~0.33 ms |

* **Final packaged router** — the committed bundle
  `models/acceptance_mlp.joblib`, calibrated with the OOD-aware quality
  threshold methodology on the held-out splits listed above. Latency, RSS,
  throughput and bundle size are end-to-end per `predict()` (TF-IDF transform
  + classifier + threshold), single-thread, warm cache:

|| metric | result |
||---|---|
|| Test accuracy (in-distribution, held-out) | 0.9950 |
|| Macro-F1 (in-distribution, held-out) | 0.9950 |
|| Routing coverage (in-distribution) | 98.33% auto-routed |
|| Routed-message accuracy | 0.9983 |
|| Confidence threshold (final) | 0.3621 |
|| p50 latency per predict | 0.903 ms |
|| p95 latency per predict | 0.919 ms |
|| p99 latency per predict | 0.919 ms |
|| Throughput | 1106.4 inferences/sec (single thread) |
|| Process RSS (max) | 117156 KiB |
|| Model bundle size on disk | 7,104,674 B (joblib + JSON sidecar ignored from publishing) |

Hardware: Jetson Orin Nano Super CPU. No CUDA/Ollama/GPU acceleration; all
latency is measured on the host CPU, one thread.

The repo target remains: macro-F1 >= ~0.99 on the held-out test set when
retrained with the same methodology, and local inference comfortably below
1 ms median per predict.

## Final acceptance results (MLP bundle, held-out 600-test + 30 OOD-test)

Measured with the default guard chain `["short_ambiguous"]` enabled. The guard
does not change in-distribution metrics versus the no-guard baseline (zero added
false-escalations); it converts the two previously-routed OOD known-failures
into escalations.

|| Metric | Result |
|||---|---|
||| Test accuracy | 0.9950 |
||| Macro-F1 | 0.9983 |
||| Routing coverage (auto-routed, in-distribution) | 98.33% |
||| Accuracy among automatically routed messages | 99.83% |
||| OOD false-accept rate | 0.40 |
||| OOD escalation rate | 0.60 |
||| In-domain false-escalations added by guard | 0 |
||| p50 latency (guarded predict) | 0.922 ms |
||| p95 latency | 0.929 ms |
||| p99 latency | 0.929 ms |
||| Throughput | ~1080 inferences/sec (single thread) |
||| Process RSS (max) | ~134000 KiB |
||| Confidence threshold | 0.3621 |

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