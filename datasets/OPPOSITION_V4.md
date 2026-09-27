# Positive/negative opposition checks on the reviewed v4 set

Ran all 44 paired messages with English questions for the baseline and both
selected schemas. Every case gets an actual positive request and a separate
negative request; negative probabilities are inferred, never calculated as
1 minus positive probability. No matching-language questions or third request ran.

For a choice question, the second request asks whether the nominated candidate
is NOT an appropriate best answer, using its actual schema's instruction and
candidate definition. Agreement means the veto answers no. This confirms one
candidate; it is not a second independent choice among all categories.
For a Boolean question, the second request asks whether the original condition
is false and reverses the selected criteria. Agreement requires decisive opposite
Boolean decisions. Exact 0.5 ties do not count as agreement. Probabilities, earlier
selection claims and expected labels are excluded from the second request.

Both requests use eight question rows. All three schemas were measured with the
same local MPS device, default row-gated fp16 autocast (threshold 5), two CPU
threads and v4 labels. Positive accuracy reproduced the preceding default-MPS
schema comparison: baseline 251/352; balanced and English-preserving 272/352 each.
Reports retain dataset/schema hashes, exact positive and generated negative
schemas, routing, snapshot/weight identity, calibration/precision settings,
package versions and separate positive/negative latency. Both routed checkpoints
were warmed with both requests before scoring; complete-input checks passed.

## Overall agreement and correctness

| Schema | Agreed labels | Agreement / coverage | Accuracy among agreed | Agreed but wrong | Errors rejected |
| --- | --- | --- | --- | --- | --- |
| baseline | 168/352 | 47.7% | 71.4% | 48 | 53 |
| balanced | 156/352 | 44.3% | 84.6% | 24 | 56 |
| english-preserving | 163/352 | 46.3% | 83.4% | 27 | 53 |

Agreement is not proof of correctness. The baseline accepted 48 incorrect labels;
the new schemas accepted 24 and 27. The baseline's agreeing subset has nearly the
same accuracy as all labels (71.4% versus 71.3%). New schemas show a more accurate
accepted subset (84.6% and 83.4% versus 77.3%), at less than half coverage. These
policies have different coverage; this does not compare them at a matched
acceptance rate or establish production thresholds. Results remain synthetic
in-sample development checks, with 22 underlying paired scenarios.

## By language

| Schema | Language | Agreed /176 | Accuracy among agreed | Agreed but wrong |
| --- | --- | --- | --- | --- |
| baseline | en | 101 | 86.1% | 14 |
| baseline | fr | 67 | 49.3% | 34 |
| balanced | en | 93 | 90.3% | 9 |
| balanced | fr | 63 | 76.2% | 15 |
| english-preserving | en | 97 | 90.7% | 9 |
| english-preserving | fr | 66 | 72.7% | 18 |

## Agreement by question

Each question has 44 evaluations across both languages. Full per-language
question counts, accepted errors and rejected errors are retained in the JSON
reports and `results/opposition-v4-comparison.csv`.

| Question | Baseline agreed /44 | Balanced agreed /44 | English-preserving agreed /44 |
| --- | --- | --- | --- |
| classification | 9 | 10 | 10 |
| industry | 13 | 10 | 10 |
| urgent_action | 37 | 33 | 33 |
| sentiment | 19 | 18 | 18 |
| requested_action | 10 | 10 | 10 |
| multiple_actions | 32 | 33 | 33 |
| message_hook | 11 | 5 | 12 |
| sensitive_data_requested | 37 | 37 | 37 |

## Required measure going forward

Every future schema comparison should retain baseline accuracy AND independently
inferred positive/negative opposition agreement. Report agreed/disagreed counts,
coverage, accuracy among agreed answers, accepted errors and rejected errors
per question and language. Keep exact schemas and runtime identity, because
agreement is itself sensitive to wording, label form, model and precision.
Do not select solely for a high agreement percentage. Review wrong agreements.

`--eval --consistency --question-schema FILE` now supports custom English schemas.
The sweep automatically saves `opposition-baseline.json` and
`opposition-selected.json` and includes their summary in its main report. Cached
opposition files must match dataset, schema, model and runtime metadata before
reuse. Every additional finalist (e.g. the English-preserving candidate) should
receive the same check. This is an evaluation metric, not a new production gate.

```sh
uv run system-one-models --eval --consistency --model laya --device mps \
  --dataset datasets/base_eval_v4.jsonl \
  --question-schema schemas/sweep-selected-english-preserving-v1.json \
  --output results/opposition-v4-english-preserving.json
```

Saved reports:
`results/opposition-v4-baseline.json`,
`results/opposition-v4-balanced.json`,
`results/opposition-v4-english-preserving.json`.
