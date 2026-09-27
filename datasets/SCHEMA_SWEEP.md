# Broad English-schema sweep and question grouping

Measured on 2026-09-26/27, using reviewed `base_eval_v4.jsonl`: 44 synthetic
messages, 22 paired scenarios, eight judgments. English model-facing questions
were used for both message languages. No French or matching-language schema ran.

## Experiment scope

Authored 12 wording families for every question: direct, concise rubric,
evidence, rules, contrast, boundary cases, generic examples, imperative,
plain question, checklist, category reasoning and balanced wording. Each was
crossed with original and compact descriptions. Baseline/rubric definitions,
choice-order permutations and Boolean display-label variants were added.

There were **254 unique question definitions**, not that many full
8-question schemas: 192 wording/description variants,
31 option-order-only variants, 9
Boolean-label variants, 8 baselines,
8 description-only variants and
6 distinct earlier rubric definitions.
**252 completed** on all 44 messages; **2 were rejected**
by complete-prompt checks before any scored inference. Nothing was truncated.
Duplicate definitions were removed. Noul dictionary order was not counted as an
option permutation because Laya always renders false before true.

The instruction-only subset retains original descriptions. Compact-description
variants change two factors and cannot isolate wording from description effects.
Order-only variants preserve baseline wording/descriptions. Some terse action
variants omit detailed precedence rules: these test underspecification sensitivity,
not a different gold-label definition. Generic examples derive from annotation
rules; all candidates share the same reviewed expected labels.

## Scores and selected candidate

Ranked per-question candidates by total correct (equally balanced English and
French counts), then by the weaker language accuracy, preferring the baseline in
a remaining tie. Combined the winners into one English schema and verified it in
fresh full-schema and isolated-question runs. Selection and assessment use the
same development examples, so these scores are optimistically selected results,
not independent validation. Baseline defaults remain unchanged.

| Message language | Fresh baseline, FP32 | Selected schema, FP32 | Selected schema, default MPS |
| --- | --- | --- | --- |
| English | 152/176 (86.4%) | 150/176 (85.2%) | 148/176 (84.1%) |
| French | 101/176 (57.4%) | 125/176 (71.0%) | 124/176 (70.5%) |

The fixed-precision selected schema changed 53 labels, fixed 37 baseline
errors and broke 15 correct answers. Its full-request exact-message counts
were 9/44, compared
with 5/44 for the live
baseline. See the saved reports for per-language exact matches and paired agreement.

| Question | Winning candidate | Baseline EN/FR | Winner EN/FR | Correct /44 |
| --- | --- | --- | --- | --- |
| classification | direct-original | 19/22, 15/22 | 18/22, 17/22 | 35/44 |
| industry | evidence-compact | 19/22, 14/22 | 18/22, 17/22 | 35/44 |
| urgent_action | examples-compact | 19/22, 16/22 | 20/22, 21/22 | 41/44 |
| sentiment | checklist-original | 21/22, 15/22 | 21/22, 18/22 | 39/44 |
| requested_action | baseline | 19/22, 10/22 | 19/22, 10/22 | 29/44 |
| multiple_actions | balanced-original | 17/22, 7/22 | 17/22, 9/22 | 26/44 |
| message_hook | checklist-compact | 17/22, 14/22 | 18/22, 15/22 | 33/44 |
| sensitive_data_requested | baseline-compact | 21/22, 10/22 | 19/22, 18/22 | 37/44 |

## Additional complete schema comparisons

Reconstructed 24 complete wording-family schemas by assembling independently
measured question answers: 22 were valid, two had a rejected long action prompt.
These are cached full-schema comparisons, not 22 additional grouped inference
runs. The 22 exact schemas and assembled predictions are saved in
`results/schema-sweep-v1-mps/family-schemas/` and `family-results/`, with a
sortable `family-leaderboard.csv`. The strongest complete family was
`rules-original`, at 246/352 (69.9%), below the live baseline's 253/352 (71.9%).
A single wording style did not help all questions; selecting variants per question
performed better on these development examples.

The English-preserving composite maximizes combined correctness subject to total
English correctness at least the fresh FP32 baseline (152/176). Dynamic programming
compares the achievable English/French totals across all per-question candidates.
This preserves **aggregate English label accuracy**, not every question or message:
individual regressions remain possible. The selected schema reached 152/176
English (86.4%) and 123/176 French (69.9%) in a fresh FP32 grouped run, with zero
answer differences from its cached reconstruction.

Here are all three under the ordinary Apple GPU precision policy, evaluated fresh:

| Message language | Default MPS baseline | Balanced candidate | English-preserving candidate |
| --- | --- | --- | --- |
| English | 150/176 (85.2%) | 148/176 (84.1%) | 150/176 (85.2%) |
| French | 101/176 (57.4%) | 124/176 (70.5%) | 122/176 (69.3%) |

`schemas/sweep-selected-english-preserving-v1.json` retains that candidate;
`schemas/sweep-selected-balanced-v1.json` retains the balanced one. Both remain
experiments. Selection used the same development set, including the preservation
constraint, so these are not held-out policy results.

For the English-preserving candidate, default MPS changed 3/352 answers between
request policies. The eight-question p50 was 342.0 ms versus 366.4 ms for the
complete eight-request sequence. FP32 eight-question accuracy was 86.4% English,
69.9% French; default-MPS isolated requests reproduced those counts. The default
MPS eight-question run was 85.2% English, 69.3% French. This is the same precision
confound seen for the other schemas, not evidence of cross-question reasoning.

## All questions together versus individually

For each schema, evaluated the same full message and exact question definitions
as one eight-question request and eight separate requests. No earlier answer or
probability was supplied to a later request. Both paths were warmed for both
checkpoints; policy execution order alternated by message. Timing includes adapter
validation and dispatch and synchronizes the accelerator. One-by-one latency is
the complete eight-request sequence. The sweep uses isolated questions to avoid
re-evaluating unchanged definitions, then verifies the final combined schema.

Laya treats questions as independent rows in a forward pass; it does not reason
jointly about previous answers. On default MPS, eight rows enable fp16 autocast,
while one row runs fp32 (`mps_amp_min_rows=5`). We also tested CPU FP32 and MPS
with `LAYA_MPS_AMP_MIN_ROWS=10000`, keeping both request policies FP32. The default
MPS run therefore measures batching plus its precision policy. MPS metadata's
`dtype=float16` describes the configured autocast dtype; actual autocast is gated
by row count. Individual reported probabilities are rounded to four decimals;
zero difference means equality at that reported precision, not bit-identical logits.

| Runtime/schema | Changed labels /352 | Largest reported P difference | All-at-once p50 ms | One-by-one p50 ms |
| --- | --- | --- | --- | --- |
| CPU FP32 baseline | 0 | 0.0001 | 1363.4 | 1341.2 |
| MPS fixed FP32 baseline | 0 | 0.0000 | 349.7 | 355.3 |
| MPS fixed FP32 selected | 0 | 0.0001 | 329.5 | 380.5 |
| MPS default baseline | 3 | 0.1222 | 350.2 | 364.0 |
| MPS default selected | 3 | 0.0441 | 353.1 | 363.8 |

The default MPS baseline's eight-question outputs differ from the historical
saved baseline in **0/352**
labels. Fresh FP32 results differ slightly from earlier saved baseline scores.
Identical schema and package versions do not guarantee identical numerical output;
historical reports omit checkpoint snapshot/precision details. The within-run
baseline is used for the sweep comparison. Precision is a supported explanation
when default MPS reproduces the old outputs, but historical snapshot identity
cannot be established retrospectively.

Latency figures are observations on this local workload, not a rigorous speed
benchmark. The sweep and CPU comparison used two PyTorch CPU threads. Final default-precision GPU comparisons ran sequentially after the sweep. The
initial fixed-FP32 GPU comparison briefly overlapped the CPU comparison, another
reason to interpret timings as approximate workload observations.
Confidence remains uncalibrated; the existing large-choice temperature fallback
warning was retained, and no production confidence threshold was selected.

## Saved artifacts and reproduction

- `schemas/wording-catalog-v1.json`: all authored instructions and compact criteria.
- `schemas/sweep-selected-balanced-v1.json`: the balanced full experimental schema.
- `schemas/sweep-selected-english-preserving-v1.json`: the aggregate-English-preserving candidate.
- `results/schema-sweep-v1-mps/family-comparisons.json` and `family-leaderboard.csv`: all 24 whole-family comparisons.
- `results/schema-sweep-v1-mps/english-preserving-*.json`: selection, grouped verification and reported scores.
- `results/schema-sweep-v1-mps/manifest.json`: exact candidate definitions, hashes,
  dataset hash, snapshot/weight identity, calibration configuration, precision,
  device, package versions and thread count.
- `results/schema-sweep-v1-mps/candidates/`: checkpointed outputs for every candidate.
- `results/schema-sweep-v1-mps/leaderboard.csv`: complete sortable results.
- `results/schema-sweep-v1-mps/summary.json`: rankings and selected candidates.
- `results/schema-sweep-v1-mps/grouping-*.json`: full predictions and grouping metrics.
- `results/schema-sweep-v1/grouping-baseline.json`: complete CPU grouping comparison;
  the redundant CPU candidate sweep was stopped after saving it.

```sh
HF_HUB_OFFLINE=1 LAYA_MPS_AMP_MIN_ROWS=10000 uv run python -m system_one_models.schema_sweep \
  --output results/schema-sweep-v2-mps --device mps --threads 2

uv run system-one-models --eval --model laya \
  --dataset datasets/base_eval_v4.jsonl \
  --question-schema schemas/sweep-selected-balanced-v1.json \
  --output results/laya-selected-v4.json
```

The first command resumes compatible completed checkpoints. Dataset, model,
catalog, package, thread or harness identity changes require a new output directory.
The second uses ordinary inference settings, including default MPS precision if
selected by the backend. It need not reproduce fixed-FP32 counts exactly.

## Rejected variants

- `requested_action--concise_rubric-original`: Question 'requested_action' exceeds the checkpoint's prompt budget; shorten its instructions or options before inference.
- `requested_action--balanced-original`: Question 'requested_action' exceeds the checkpoint's prompt budget; shorten its instructions or options before inference.

## Separate opposition measurement

The [v4 positive/negative opposition study](OPPOSITION_V4.md) subsequently evaluated
the baseline and both selected candidates with actual negative requests. It is
distinct from request-grouping and translation agreement. Future sweep runs retain
this measure automatically. Because the harness changed, use a new output directory
(e.g. `results/schema-sweep-v2-mps`) instead of resuming the original v1 manifest.
