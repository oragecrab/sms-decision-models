# Rubric-aligned English schema experiment

Measured 2026-09-26 on the paired synthetic v4 development set (44 messages,
22 paired scenarios). English questions were used for both message languages.
No matching-language questions were run.

The baseline is the saved `results/laya-paired-v3.json` predictions rescored
against v4 labels, saved as `results/laya-baseline-v4-rescored.json`. Its inference
and timing were not rerun. The variant is a fresh local Laya run using
`schemas/rubric-v1.json`, saved as `results/laya-rubric-v4.json`. The variant
report records exact definitions and their order-sensitive hash. Complete prompt
and message checks passed without truncation. Laya emitted its existing
large-choice temperature fallback warning; confidence is not claimed calibrated.

| Message language | Baseline, reviewed labels | Rubric-en-v1 |
| --- | --- | --- |
| English | 150/176 (85.2%) | 131/176 (74.4%) |
| French | 101/176 (57.4%) | 99/176 (56.2%) |
| Combined | 251/352 (71.3%) | 230/352 (65.3%) |

Exact-message matches fell from 5/22 to 3/22 in English and stayed 0/22 in French.
Translation agreement increased from 102/176 to 103/176 labels; this does not
establish correctness. Variant p50 latency was 1538.6 ms, p95 1738.5 ms, with
5066.7 ms load and warmup, on this local run. No controlled speed comparison is
claimed against the previously saved baseline timing.

| Question | English baseline → variant | French baseline → variant |
| --- | --- | --- |
| classification | 19/22 → 19/22 | 15/22 → 15/22 |
| industry | 19/22 → 19/22 | 14/22 → 14/22 |
| urgent_action | 19/22 → 18/22 | 16/22 → 18/22 |
| sentiment | 21/22 → 20/22 | 15/22 → 14/22 |
| requested_action | 17/22 → 11/22 | 10/22 → 7/22 |
| multiple_actions | 17/22 → 10/22 | 7/22 → 7/22 |
| message_hook | 17/22 → 17/22 | 14/22 → 14/22 |
| sensitive_data_requested | 21/22 → 17/22 | 10/22 → 10/22 |

The variant changes wording, action descriptions and action option order together,
so these results cannot attribute effects to one change. The fixed primary-action
tie-break is independent of option order. The largest English losses are on
multiple actions (seven) and primary action (six). Classification and industry,
whose prompts were unchanged, retained identical accuracy in each language.
The clearer annotation contract did not guarantee better model adherence.

Keep the baseline schema as the default and retain the rubric variant for controlled
experiments. Next isolate one question/wording/order change at a time, rather than
promoting this combined variant. These previously inspected synthetic examples
are regression and sensitivity checks, not an independent holdout.

```sh
uv run system-one-models --eval --model laya \
  --dataset datasets/base_eval_v4.jsonl \
  --question-schema schemas/rubric-v1.json \
  --output results/laya-rubric-v4.json
```
