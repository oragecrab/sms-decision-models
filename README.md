# Local scam-text triage with Laya

A small `uv` project for experimenting with Laya on SMS and email text. In one call, the model returns a scam category, the apparent industry or service, whether the message urges immediate action, its sentiment (`happy`, `neutral`, or `sad`), the primary requested action, whether it asks for multiple distinct actions, the message's main hook, and whether it requests sensitive information. Choice questions include a probability distribution; yes/no questions report the probability of “yes.” These are judgments about the supplied text: the industry guess does not verify the sender's actual identity, and Laya does not inspect URLs or sender authentication.

## Setup

Python 3.14 is pinned for this project. Install dependencies and run the examples:

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

Classify one message:

```sh
uv run system-one-models \
  --channel email \
  --sender 'security@example.com' \
  --subject 'Action required: account locked' \
  --body 'Sign in immediately using this link to restore access.'
```

The demo reads [`datasets/base_eval_v1.jsonl`](datasets/base_eval_v1.jsonl), a small, versioned set of 11 synthetic SMS and email examples. Each case contains a manually assigned expected value for every question, and the compact output shows predictions, expected values, per-question matches, inference time, and a score summary. Add `--details` to show every choice probability. In an interactive terminal, headings and categories are colored; set `NO_COLOR=1` to disable color. The first timing includes loading the model and, on the very first run, downloading its checkpoint. Later demo timings are closer to inference time alone.

The [label review](datasets/LABEL_REVIEW.md) records the annotation correction and rubric-dependent cases. The [prompt tuning report](datasets/PROMPT_TUNING.md) records measured improvements, remaining disagreements, and additional paraphrase checks. Because the prompts were tuned on these examples, their scores are development results rather than an independent evaluation.

This is a **base evaluation set**, not a benchmark: it is small, synthetic, and manually labeled. Its aggregate scores only help reveal obvious behavior and regressions on these examples; they do not estimate real-world scam detection accuracy or replace a representative, reviewed, held-out dataset. Treat uncertain messages as needing review.

```sh
uv run system-one-models --demo --details
```

Some smoke-test messages previously exposed errors such as confusing a parcel-fee scam with marketing and missing a sign-in request. The checkpoint also warns that some confidence values, including large-choice questions, are uncalibrated. The base set is intended to make such behavior visible, not to imply that a particular score is reliable.

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

`__init__.py` contains only the package docstring. Import functions from their
own modules, such as `from system_one_models.classifier import classify`.
You can also run the CLI with `uv run python -m system_one_models`.
