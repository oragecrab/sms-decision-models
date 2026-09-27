# Local scam-text triage with Laya

A small `uv` project for experimenting with Laya on SMS and email text. In one call, the model returns a scam category, the apparent industry or service, whether the message urges immediate action, its sentiment (`happy`, `neutral`, or `sad`), the primary requested action, whether it asks for multiple distinct actions, the message's main hook, and whether it requests sensitive information. Choice questions include a probability distribution; yes/no questions report the probability of “yes.” These are judgments about the supplied text: the industry guess does not verify the sender's actual identity, and Laya does not inspect URLs or sender authentication.

## Setup

Python 3.12 is pinned for this project. Install dependencies and run the examples:

```sh
uv sync
uv run system-one-models --demo
```

Laya downloads its checkpoint from Hugging Face the first time inference runs. The files are then cached locally and predictions run on this machine. The first run needs an internet connection.

All question instructions and options are checked against the selected checkpoint's
actual tokenizer before inference. The complete message, including sender and subject,
must fit every question's remaining token budget. Oversized inputs are rejected with
“No prediction was made” and need review; the tool does not classify a truncated prefix.
This check can load or download the model before reporting that an input is too long.

Choose a device explicitly, or leave `--device auto` to let Laya select an
available accelerator (including Apple GPU/MPS). The demo and single-message
command report the actual loaded device, including fallback to CPU:

```sh
uv run system-one-models --demo --device cpu
uv run system-one-models --demo --simulate-server --device mps --repeats 2
```

For CPU inference, tune the number of PyTorch computation threads:

```sh
uv run system-one-models --demo --simulate-server --device cpu --offline --cpu-threads 4
```

The demo prints the actual CPU thread setting. A single inference worker can use
several cores through PyTorch's native tensor operations; this flag does not
create more Routers or request workers. Without the flag, PyTorch's existing
setting is preserved. Thread count is not a guarantee that every operation uses
all those cores; tokenization, loading, and smaller operations can be serial.
Compare counts such as 1, 4, and 8 using the timing summary. More threads can add
overhead or memory contention rather than improve throughput. The setting applies
to PyTorch CPU computations; it does not increase GPU parallelism. It configures
the PyTorch runtime rather than reserving separate cores for each Router.

Checkpoint files are already cached by Hugging Face (normally under
`~/.cache/huggingface/hub`, configurable through `HF_HOME`/`HF_HUB_CACHE`). Online
runs may check for newer revisions and show file-fetch progress even when weights
are cached. Each new process still loads cached weights into memory; this is not
a fresh download. To skip remote checks and use the cached checkpoint only:

```sh
uv run system-one-models --demo --device cpu --offline
```

`--offline` resolves only the required inference files to a local snapshot and
loads from that path. If those files are missing, it asks you to run once without
`--offline`. The same flags work in the server simulation and single-message CLI.

Classify one message:

```sh
uv run system-one-models \
  --channel email \
  --sender 'security@example.com' \
  --subject 'Action required: account locked' \
  --body 'Sign in immediately using this link to restore access.'
```

The demo and evaluation default to [`datasets/base_eval_v3.jsonl`](datasets/base_eval_v3.jsonl): 44 synthetic SMS and email messages covering 22 scenarios, each in English and French with identical expected labels. The original English scenarios are translated into French, and the newer French scenarios are translated into English. [Paired dataset notes](datasets/PAIRED_V3.md) describe translations, metadata and measured results. Each case contains a manually assigned expected value for every question, and the compact output shows predictions, expected values, per-question matches, inference time, and a score summary. Add `--details` to show every choice probability. In an interactive terminal, headings and categories are colored; set `NO_COLOR=1` to disable color. The first timing includes loading the model and, on the very first run, downloading its checkpoint. Later demo timings are closer to inference time alone.

The [label review](datasets/LABEL_REVIEW.md) records the annotation correction and rubric-dependent cases. The [prompt tuning report](datasets/PROMPT_TUNING.md) records measured improvements, remaining disagreements, and additional paraphrase checks. Because the prompts were tuned on these examples, their scores are development results rather than an independent evaluation.

This is a **base evaluation set**, not a benchmark: it is small, synthetic, and manually labeled. Its aggregate scores only help reveal obvious behavior and regressions on these examples; they do not estimate real-world scam detection accuracy or replace a representative, reviewed, held-out dataset. Treat uncertain messages as needing review.

```sh
uv run system-one-models --demo --details
```

Some smoke-test messages previously exposed errors such as confusing a parcel-fee scam with marketing and missing a sign-in request. The checkpoint also warns that some confidence values, including large-choice questions, are uncalibrated. The base set is intended to make such behavior visible, not to imply that a particular score is reliable.

## Checkpoint calibration warning

The cached checkpoint contains `choice:11+` temperature `0.10058280825614929`.
Laya accepts temperatures in `[0.5, 5]`, clamps this value to `0.5`, and warns that
the affected probabilities are uncalibrated. The demo's 12-choice requested-action
question uses that bucket. This temperature scales decision logits; it is not
a GPU temperature or a download error. Inference continues, but its reported
requested-action probabilities should not be interpreted as calibrated accuracy.
The warning is kept visible; the cached checkpoint is not edited.

## Simulating an HTTP serving workload

```sh
uv run system-one-models --demo --simulate-server --queue-size 4 --repeats 3
```

`--repeats 3` submits 132 concurrent requests: three copies of the 44-message set,
all in one burst through the same worker. The default is one repeat. Repeats also
work with the sequential demo, reusing one Router across every pass. One worker
thread loads and warms both English and multilingual checkpoints in one Router, then reuses it for every
request. Blocking inference stays outside the asyncio event loop. A bounded queue
holds up to four waiting requests; additional callers await capacity rather than
being rejected. This opens no HTTP listener and does not batch model inference.

Each result shows inference time, waiting time (including waiting for queue
capacity), and total request latency. Startup/loading/warmup is timed separately
from serving time and throughput. Results are displayed in dataset order. Under a
burst, later requests wait longer; this simulation does not increase model speed.
The summary includes p50, p95 (nearest rank), and maximum inference, wait, and
request latency, plus label consistency against the first pass. Repeated inputs
increase the workload, not the number of independent evaluation examples.

## Brand names and further analysis

Laya returns closed-choice, score, and yes/no answers; it does not generate arbitrary brand-name strings. To extract mentioned names, add a separate local named-entity extractor, then keep the extracted mention distinct from a verified sender identity. The industry answer is only the broad sector suggested by the message.

For more reliable analysis, the next useful inputs are email authentication results (SPF/DKIM/DMARC), reply-to and sender-domain mismatches, actual link targets and redirects, attachment names/types, and prior thread context. Treat URLs and attachments as untrusted; do not open them as part of analysis. The FTC recommends checking a suspected message through a known-good contact route instead of using its links or phone numbers ([phishing guidance](https://consumer.ftc.gov/articles/how-recognize-avoid-phishing-scams)).

Before relying on predictions, build a labeled set from the messages you expect to receive. Keep a time-separated holdout set, review false negatives in high-risk action classes, measure precision/recall by scam family and channel, and set thresholds that send uncertain or consequential cases to human review.

## Tests

```sh
uv run pytest
```

The pytest suite checks input shaping, the adapter contract, token-budget boundaries,
and completeness of the base evaluation labels without downloading a model. To also
verify full prompt encoding and the base examples against a locally cached checkpoint
tokenizer, set `LAYA_TEST_TOKENIZER` to its tokenizer directory when running pytest. Use `--demo` for real local inference against the bundled base set.

## Code layout

- `questions.py`: question instructions and category definitions.
- `classifier.py`: input shaping, token-budget validation, and model inference.
- `formatting.py`: terminal output and display labels.
- `evaluation.py`: base dataset loading and scoring.
- `cli.py`: command-line arguments and orchestration.
- `simulation.py`: bounded concurrent-request simulation with one preloaded worker.
- `runtime.py`: device configuration, cache-only loading, and actual device reporting.

`__init__.py` contains only the package docstring. Import functions from their
own modules, such as `from system_one_models.classifier import classify`.
You can also run the CLI with `uv run python -m system_one_models`.

## Comparing local models

Python 3.12 is used because the Decider dependency stack pins NumPy 1.x.
Install the optional Decider backend, then evaluate Laya and Decider on identical
questions and labels (models are loaded sequentially):

```sh
uv sync --extra decider
uv run system-one-models --eval --model laya --model Mapika/decider-4b \
  --device auto --output results/laya-vs-decider.json
```

Repeat `--model` for other Decider checkpoints, such as `Mapika/decider-0.8b`
or `Mapika/decider-2b`, or supply a local Decider checkpoint directory. This
adapter supports Decider checkpoints, not arbitrary Hugging Face architectures.
Use `--revision COMMIT_SHA` to pin the Decider snapshot; the same revision applies
to all Decider models in that invocation. Laya uses its default Router checkpoints.
Reports record the resolved Decider snapshot, installed package versions, full
question definitions, dataset hash, expected labels, answers and probabilities.
They include per-question accuracy, negative log likelihood (NLL), multiclass
Brier score (sum over classes), exact-message accuracy, and p50/p95 latency.
Lower NLL and Brier scores are better. NLL clips gold probability at 1e-12.

Loading and one warmup prediction are timed separately. Measured latency includes
adapter validation and inference, excludes printing and scoring, and uses eager
Decider inference without CUDA graphs. The warmup uses the first dataset row;
that row is evaluated again and scored exactly once. Each completed model is saved
to the output report, so a later model failure leaves earlier results available.

For a held-out dataset, use `--dataset path/to/holdout.jsonl`. Each line must have
a unique `id`, a `state` object, and an `expected` object with all eight question
labels, matching the bundled dataset. `--details` prints individual predictions.
The bundled 44 messages represent 22 paired scenarios and are development checks, not an independent benchmark. Reports include translation agreement and per-scenario differences. Keep both translations together in any future data split. Reports show English and French results separately as well as combined. The `language` annotation stays outside the model input; Laya routes French messages to its multilingual checkpoint automatically, and reports retain actual routing decisions. The original English-only `base_eval_v1.jsonl` and unpaired bilingual `base_eval_v2.jsonl` are preserved for reproducing earlier runs.

For a single message:

```sh
uv run system-one-models --model Mapika/decider-4b --body 'Your parcel is held. Pay a fee now.'
```

Decider downloads weights on first use. The 4B checkpoint has 8.4 GB of bf16
weights plus runtime memory overhead; CUDA is the upstream-tested path for the
current release, while MPS is supported by the package but unverified for that
checkpoint. CPU is also available. See the [model card](https://huggingface.co/Mapika/decider-4b).
Decider rejects states whose `Context:` prefix and content exceed 32,768 tokenizer tokens before inference;
Laya retains its existing per-question token-budget checks.

## JevK5

Install the official runtime (locked to its v0.3.0 Git release) and compare:

```sh
uv sync --extra jevk5
uv run --extra jevk5 system-one-models --eval \
  --model laya --model alibiserikbay/JevK5 \
  --device auto --output results/laya-vs-jevk5.json
```

To include Decider, use both `--extra jevk5 --extra decider` and add another
`--model Mapika/decider-4b`. JevK5, JevK5-2B, and JevK5-9B repo names select the
JevK5 adapter automatically. Local directories with `jevk5_config.json` are also
detected; use `--model jevk5:REPO_OR_DIRECTORY` for other JevK5 checkpoints.
`--revision` applies to every non-Laya remote checkpoint in a run; use separate
invocations if different repositories need different revisions.

The adapter uses JevK5's own prompt format and checkpoint calibration settings.
It evaluates each of the eight questions separately and returns the existing
choice distributions and P(true) contract. Prompt encoding is included in measured
latency; all complete prompts are checked before any question is evaluated.
The current adapter supports up to 16 options per question (the bundled maximum
is 12) and rejects prompts over 16,384 tokens without truncating evidence.

JevK5 runs eagerly, with CUDA graphs disabled to match the Decider evaluation mode.
`auto` chooses CUDA when available, otherwise CPU. CUDA uses bf16; the CPU path
uses float32, requires roughly 16 GB for weights plus runtime overhead, and is
not validated against a full checkpoint here. Native MPS is not supported by
this adapter. The upstream documented path uses CUDA (~9 GB GPU memory); for
Apple GPU inference upstream provides a separate GGUF/llama.cpp runtime, which
this adapter does not integrate. See the [official runtime](https://github.com/allebee/jevk5).

## Zerank-2 as a candidate-label reranker

```sh
uv sync --extra zerank
uv run --extra zerank system-one-models --eval \
  --model laya --model zeroentropy/zerank-2-reranker \
  --output results/laya-vs-zerank.json
```

For the Transformers 5 backends, use `uv run --extra decider --extra jevk5
--extra zerank --extra clm system-one-models --eval` and repeat `--model` for
each checkpoint. GLiNER requires a separate environment (see below). `zerank:REPO_OR_DIRECTORY` explicitly selects this
adapter for a local copy or another compatible Zerank snapshot. Revisions and
snapshot paths are recorded as with the other Hugging Face backends.

Zerank is a retrieval reranker, so this is an experimental classification
adaptation. Each question and the complete JSON message form the query; each
candidate label and its description form a candidate document. The official
Sentence Transformers CrossEncoder returns raw relevance logits. The adapter
ranks those logits and uses `softmax(logits / 5)` only to display relative weights
and choose a yes/no label. These are not calibrated class probabilities; NLL
and Brier metrics are omitted for this backend. Raw logits and score semantics
are saved in the report. See the [model card](https://huggingface.co/zeroentropy/zerank-2-reranker).

The official chat template and readout are loaded without remote Python code.
Every fully rendered pair is checked against the 32,768-token limit, with
truncation disabled during both validation and inference. Inference uses batches
of one candidate to limit peak memory; latency covers all candidates across all
eight questions. This is a different workload from a decision model's latency.
CUDA, MPS, and CPU selection is available; full-checkpoint inference on this
machine has not yet been verified. Models download their weights on first use.

## GLiNER2.5-Decide and CLM

GLiNER requires Transformers 4, while CLM, Decider, and JevK5 require Transformers
5. Their extras are declared mutually exclusive. Keep GLiNER in its own environment
so switching models does not replace the existing backend dependencies:

```sh
UV_PROJECT_ENVIRONMENT=.venv-gliner uv run --extra gliner system-one-models \
  --eval --model fastino/GLiNER2.5-Decide --device cpu \
  --output results/gliner-decide.json

uv run --extra clm system-one-models \
  --eval --model clm --device mps --output results/clm-reference.json
```

`clm` selects the released `Contrastive-LM/CLM-v0.1-8B` projection head and its
matching `Qwen/Qwen3-8B` encoder. It downloads both on first use. `--revision`
pins the head repository; set `CLM_ENCODER_REVISION` to a commit SHA to pin the
encoder separately. Both resolved snapshot paths are recorded. The 8B encoder
uses float16 on MPS/CUDA and float32 on CPU; the latter has substantially higher
memory requirements. Loading occurs on CPU before transfer to the selected
accelerator because direct MPS checkpoint loading crashed on this machine.

The local CLM adapter uses Transformers with raw text, last-token pooling, and
L2 normalization. It uses the upstream head architecture, checkpoint loading,
question rendering, and answer assembly vendored from a pinned Apache-2.0 commit
(with LICENSE and NOTICE). It evaluates the released CLM model without running
vLLM. This is a local implementation of the reference readout, not a measurement
of the published vLLM serving speed. Exact-text embeddings for states and
candidates are cached (4096 entries); warmup populates the candidate cache and
first example. Inputs longer than 2048 tokens are rejected, never truncated.

GLiNER uses the pinned official runtime's classification scorer to return every
label probability. All eight single-label heads share one encoder pass, with the
same instructions and descriptions as the other adapters. Yes/no heads have
explicit true/false options and return P(true). The scorer uses softmax at
temperature 1; these probabilities are evaluated without a calibration claim.
The full schema plus text is checked against a conservative adapter limit of
4096 subword tokens, with word truncation disabled and strict preprocessing errors.

Use `gliner:REPO_OR_DIRECTORY` or `clm:HEAD_DIRECTORY` for explicit adapter
selection. A local CLM directory must contain `CLM_v0.1-8B.pt`; custom heads must
use the released Qwen3-8B encoder and reference checkpoint format.

## Separate negative confirmation

```sh
uv run system-one-models --eval --consistency --model laya \
  --output results/laya-consistency.json
```

The first inference request returns the normal positive predictions. A second,
separate request evaluates their negative forms. For choice questions it asks
whether the nominated label is **not** the appropriate best answer, with that
label's description and the original rubric. For boolean questions it asks
whether the original condition is false, with true/false criteria reversed.
The negative request sees no positive probability or claim that the model picked
the candidate. This is targeted confirmation of the nominated label, not an
exhaustive one-versus-rest evaluation of every candidate.

A choice is accepted when its negative answer is no. A boolean is accepted when
the positive and negative decisions are opposites. Probabilities exactly at 0.5
are rejected as ties. Reports retain both actual model outputs and both request
schemas; negative probabilities are inferred rather than computed as 1 minus the
positive probability. Every example remains in the denominator for coverage.

The report shows baseline accuracy, accepted-label coverage, accuracy on accepted
labels, accepted errors, rejected errors, and exact-match accuracy for messages
where all eight labels are accepted. Empty accepted sets have null accuracy.
Agreement can still be wrong; this measures whether the filter helps on the
chosen data. It approximately doubles inference work. Use the same adapter extras
and separate GLiNER environment as in ordinary evaluation.

Reports also record package versions and p50/p95 positive, negative, and combined
request latency, excluding load and warmup. The measured Laya development-set
result is recorded in [the comparison report](datasets/MODEL_COMPARISON.md#separate-positivenegative-consistency-check).

## Third request on disagreements

```sh
uv run system-one-models --eval --consistency --third-request --model laya \
  --output results/laya-third-request.json
```

Only disputed questions are retried in a separate request using fixed paraphrases
and the original rubric and label definitions. The third prompt contains the
message but no earlier answers, probabilities, conflict notice, or expected labels.
The same model provides another decision; this is not an independent source of
truth. The extra warmup evaluates all paraphrased questions once and is excluded
from scored inference latency.

The report compares original versus third accuracy on disputed labels, errors
fixed, correct answers broken, and accuracy if all disputed answers are replaced.
It also evaluates acceptance with two supporting decisions: matching original and
third answers corroborate a candidate; for booleans the inverted negative answer
can support the third answer instead. For choices, a negative veto does not name
an alternative, so a changed third category remains unresolved. Exact third ties
remain unresolved. Previously accepted answers are retained in both policies.

This experiment hurt accuracy on the bundled development examples; see the
[measured results](datasets/MODEL_COMPARISON.md#third-request-experiment).

## French questions and categories

For the paired bilingual dataset, compare the original English schema with
French instructions, category names, descriptions and yes/no criteria on French
messages:

```sh
uv run system-one-models --eval --model laya --question-language match \
  --output results/laya-paired-v3-matched-questions.json
```

`en` (the default) uses English questions for all messages; `fr` uses French
questions for all messages; `match` follows each dataset row's language annotation.
The latter is evaluation metadata, not a production language detector. Dataset
annotations are not sent to the model, and Laya still routes from the message text.
Outputs map back to the same canonical labels, while the JSON report retains raw
outputs and the model-facing schema. French schema selection currently supports
ordinary `--eval`, not `--consistency`. The matched-language experiment reduced
French accuracy on these examples; [full results](datasets/PAIRED_V3.md#french-question-and-category-schema)
are recorded alongside the message translation comparison.

## Guidelines and planned improvements

Questions, instructions and category descriptions remain in English for both
message languages (`--question-language en`, the default). The optional French
schema is retained for experiments.

[Decision-model guidelines and next steps](docs/DECISION_MODEL_GUIDELINES.md)
records the research sources, current decisions, measured experiments and the
ordered implementation plan. It distinguishes completed annotation and schema
work from pending sensitivity experiments, decomposition, calibration and
deployment validation.

## Reviewed annotation rules

[Annotation rubric v1](docs/ANNOTATION_RUBRIC.md) resolves urgency, optional
actions, primary action, factual adverse tone and overlapping hooks. A review
of all 44 paired messages produced six label corrections across three scenarios
in `datasets/base_eval_v4.jsonl`. Message text and translations are unchanged.
The default remains v3 for reproducing earlier runs; use the reviewed set explicitly:

```sh
uv run system-one-models --eval --model laya \
  --dataset datasets/base_eval_v4.jsonl --output results/laya-paired-v4.json
```

Existing scores are historical v3 results. Baseline English/French prompts remain
unchanged pending a separately versioned rubric-aligned schema. v4 is synthetic
development data, not a holdout or production accuracy estimate.

## Versioned question schemas

Evaluate a question variant without changing the baseline definitions:

```sh
uv run system-one-models --eval --model laya \
  --dataset datasets/base_eval_v4.jsonl \
  --question-schema schemas/rubric-v1.json \
  --output results/laya-rubric-v4.json
```

A schema JSON contains `version` (a nonempty string) and `questions` (the full
question dictionary). Variants may change instructions, descriptions, option
order and distinct boolean display labels. They must preserve all canonical
question IDs, types and criterion keys. The true/false semantic slots remain
fixed. Duplicate keys and malformed definitions are rejected before model loading;
the selected backend retains its complete-prompt and message token checks.

`--question-schema` supports `--eval`, including `--consistency`, with English
questions for both message languages. It cannot be combined with `--demo` or
French/matching-language questions. No matching-language run is required.
Reports (format version 2) retain schema version, source, SHA-256, exact
definitions and the model-facing schemas used. Question and option order affect
the hash; JSON whitespace and definition-property order do not. Validation,
scoring, language summaries and paired comparisons use the selected schema.

`rubric-en-v1` encodes the reviewed boundaries, with a fixed primary-action
precedence independent of option order. It also changes action option order and
shortens descriptions to fit the checkpoint prompt budget, so its combined result
cannot isolate the effect of any single wording or ordering change. It is an
experimental variant; the default baseline remains unchanged.

The [first rubric-schema experiment](datasets/SCHEMA_VARIANTS.md) reduced accuracy
to 74.4% English and 56.2% French, compared with the saved baseline rescored on v4
(85.2% and 57.4%). It remains experimental; no default schema was changed.

## Broad schema experiments and request grouping

The [254-definition English-schema sweep](datasets/SCHEMA_SWEEP.md) compares
12 wording families per question, two description styles, option orders and
Boolean labels. It records every candidate, rejected token budgets, language
scores and an experimental combined schema. The same study compares all eight
questions together with eight separate requests, including CPU, fixed-precision
Apple GPU and default Apple GPU settings. Defaults remain unchanged.

On the reviewed development set under normal Apple GPU settings, the balanced
candidate scored 84.1% English / 70.5% French. The aggregate-English-preserving
candidate scored 85.2% / 69.3%, compared with baseline 85.2% / 57.4%. Both are
saved under `schemas/`; neither changes the default. Fixed-precision grouping
changed no answers; default Apple GPU grouping changed three labels out of 352
because its grouped requests enabled a different numerical precision.

## Opposition agreement as a retained measure

[Positive/negative checks for v4](datasets/OPPOSITION_V4.md) record actual
independent negative-question inference for the baseline and both selected schemas.
Agreement was 168/352, 156/352 and 163/352 respectively; accuracy among agreeing
answers was 71.4%, 84.6% and 83.4%. Wrong agreements remained (48, 24 and 27).
Future sweeps retain baseline and selected-schema opposition results automatically;
every additional finalist should also be evaluated with `--consistency`.
Reports include coverage and accepted/rejected errors by question and language.

### Confidence versus opposition filtering

The offline [v4 filtering comparison](datasets/FILTER_COMPARISON_V4.md) matches acceptance counts by question and language, retains threshold curves and tie diagnostics, and compares accepted/rejected errors. Run `uv run python -m system_one_models.filter_comparison REPORT.json --output COMPARISON.json` on saved opposition reports; no inference runs.

### Action-decomposition experiment

The [three-prototype action study](datasets/ACTION_DECOMPOSITION_V1.md) evaluates source-span request selection and independent-task relationships against the existing `multiple_actions` question. None improved overall accuracy. Reports retain actual opposition checks and component diagnostics; the existing defaults remain in place.


### Experimental independent-task policy

The [reviewed task-count study](datasets/ACTION_IMPROVEMENT_V2.md) compares new schemas, request decomposition and cached local models. Its fixed binary GLiNER policy reached 54/64 correct on a fresh paired English/French test, compared with 41/64 for balanced Laya. It reduced false positives but missed more genuine multi-task messages, so the eight-question default remains unchanged.

Run the separate single-message option using the GLiNER environment:

```sh
HF_HUB_OFFLINE=1 .venv-gliner/bin/python -m system_one_models \
  --task-count-policy schemas/task-count-gliner-binary-v2.json \
  --device mps --body 'Please pay the invoice and separately email your signed agreement.'
```

The policy pins the evaluated checkpoint, packages, five-head grouping and 0.5 binary cutoff. Output preserves the categorical probabilities and task-count band alongside the derived `multiple_actions` answer. Add `--task-count-opposition` to measure an independently inferred complementary answer; agreement is diagnostic and performed poorly as an acceptance filter. These are experimental task-count results, not complete demoset accuracy.


The [CPU latency benchmark](datasets/CPU_LATENCY_V1.md) measures these task-count policies at two and four threads, including actual opposition separately, and provides repeatable commands for the target OpenShift CPU. At four threads on the local M1 Max, median positive latency was 761 ms for the frozen GLiNER policy and 134 ms for balanced Laya.
