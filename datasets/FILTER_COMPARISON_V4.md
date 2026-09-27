# Confidence versus opposition at matched coverage

Offline analysis of the saved v4 baseline, balanced and English-preserving opposition runs. No model inference or matching-language questions ran. Both filters retain the original positive answer; neither changes its value. These are 352 labels from 22 synthetic paired scenarios, already used for schema selection.

## Primary comparison

Match the number of accepted labels **within each question × message language**. Rank by the probability of the selected positive answer: the nominated category probability for Choice and `max(p, 1-p)` for Noul. This controls acceptance composition across tasks and language checkpoints. The required counts come from opposition, without using correctness to choose counts or thresholds.

| Schema | Accepted /352 | Opposition accuracy | Confidence accuracy | Opposition errors | Confidence errors |
| --- | --- | --- | --- | --- | --- |
| Baseline | 168 (47.7%) | 71.4% | 75.6% | 48 | 41 |
| Balanced | 156 (44.3%) | 84.6% | 84.6% | 24 | 24 |
| English-preserving | 163 (46.3%) | 83.4% | 84.7% | 27 | 25 |

Confidence retains seven fewer errors for the baseline, ties for balanced, and retains two fewer for English-preserving. There is no aggregate advantage for opposition on these examples at matched task/language coverage. This is a ranking diagnostic, not validation of production thresholds.

## By question and language

Counts below are identical for both filters. Errors are incorrect answers retained by that filter.

| Schema | Question | Language | Accepted /22 | Opposition errors | Confidence errors |
| --- | --- | --- | --- | --- | --- |
| Baseline | classification | en | 8 | 1 | 0 |
| Baseline | classification | fr | 1 | 0 | 0 |
| Baseline | industry | en | 12 | 2 | 1 |
| Baseline | industry | fr | 1 | 0 | 0 |
| Baseline | urgent_action | en | 22 | 3 | 3 |
| Baseline | urgent_action | fr | 15 | 2 | 2 |
| Baseline | sentiment | en | 10 | 0 | 1 |
| Baseline | sentiment | fr | 9 | 5 | 2 |
| Baseline | requested_action | en | 8 | 1 | 0 |
| Baseline | requested_action | fr | 2 | 0 | 2 |
| Baseline | multiple_actions | en | 12 | 5 | 4 |
| Baseline | multiple_actions | fr | 20 | 14 | 14 |
| Baseline | message_hook | en | 7 | 1 | 0 |
| Baseline | message_hook | fr | 4 | 2 | 2 |
| Baseline | sensitive_data_requested | en | 22 | 1 | 1 |
| Baseline | sensitive_data_requested | fr | 15 | 11 | 9 |
| Balanced | classification | en | 6 | 0 | 0 |
| Balanced | classification | fr | 4 | 0 | 0 |
| Balanced | industry | en | 8 | 2 | 1 |
| Balanced | industry | fr | 2 | 1 | 0 |
| Balanced | urgent_action | en | 19 | 2 | 2 |
| Balanced | urgent_action | fr | 14 | 0 | 0 |
| Balanced | sentiment | en | 10 | 0 | 1 |
| Balanced | sentiment | fr | 8 | 1 | 2 |
| Balanced | requested_action | en | 8 | 1 | 0 |
| Balanced | requested_action | fr | 2 | 0 | 2 |
| Balanced | multiple_actions | en | 18 | 2 | 4 |
| Balanced | multiple_actions | fr | 15 | 11 | 10 |
| Balanced | message_hook | en | 4 | 0 | 0 |
| Balanced | message_hook | fr | 1 | 0 | 0 |
| Balanced | sensitive_data_requested | en | 20 | 2 | 1 |
| Balanced | sensitive_data_requested | fr | 17 | 2 | 1 |
| English-preserving | classification | en | 6 | 0 | 0 |
| English-preserving | classification | fr | 4 | 0 | 0 |
| English-preserving | industry | en | 8 | 2 | 1 |
| English-preserving | industry | fr | 2 | 1 | 0 |
| English-preserving | urgent_action | en | 19 | 2 | 2 |
| English-preserving | urgent_action | fr | 14 | 0 | 0 |
| English-preserving | sentiment | en | 10 | 0 | 1 |
| English-preserving | sentiment | fr | 8 | 1 | 2 |
| English-preserving | requested_action | en | 8 | 1 | 0 |
| English-preserving | requested_action | fr | 2 | 0 | 2 |
| English-preserving | multiple_actions | en | 18 | 2 | 4 |
| English-preserving | multiple_actions | fr | 15 | 11 | 10 |
| English-preserving | message_hook | en | 8 | 0 | 0 |
| English-preserving | message_hook | fr | 4 | 3 | 1 |
| English-preserving | sensitive_data_requested | en | 20 | 2 | 1 |
| English-preserving | sensitive_data_requested | fr | 17 | 2 | 1 |

For example, balanced French `requested_action` retains 0 errors with opposition versus 2 with confidence, while French `message_hook` retains 0 versus 1. Confidence does better on French sensitive-data requests (1 versus 2 errors), but both retain 10 incorrect French multiple-action answers. Neither filter solves that underlying weakness.

## Ties, native confidence and pooled comparisons

Saved scores are rounded. For an exact acceptance count, equal boundary scores are ordered by message ID then question ID, never by the reviewed label. The report records tie size and best/worst possible retained-error counts. All boundary ties in the primary question/language comparison have identical correctness, so the headline error counts are unaffected by tie ordering.

Also retain genuine `confidence >= threshold` curves: entire ties enter together, from zero to full coverage. Exact top-k matching is a batch ranking experiment; a single threshold may not reproduce its acceptance count. No threshold was optimized using gold labels.

The saved native `confidence` uses a categorical margin rather than nominated probability. It produces the same within-question/language rankings and primary results here. Pooled results differ because raw scores across questions/checkpoints are not calibrated to each other.

| Schema | Pooled nominated-probability errors | Pooled native-confidence errors | Opposition errors |
| --- | --- | --- | --- |
| Baseline | 40 | 44 | 48 |
| Balanced | 28 | 31 | 24 |
| English-preserving | 27 | 31 | 27 |

Pooled matching preserves only total coverage, not which questions or languages are accepted. It is supplementary; it changes the task mix and must not replace the stratified comparison.

## Error types and recorded latency

JSON retains accepted/rejected errors, rejected correct answers, Boolean false positives/negatives, missed scams, false scam flags, category confusion, overlap and filter-only errors for each group. It also retains each label’s confidence, prediction and correctness. The CSV provides all 48 question/language comparisons.

Confidence ranking uses the existing positive outputs and needs no second model request. Opposition requires a separate negative request. Recorded full-message latency from the original runs is shown below; these are workload observations, not new timing measurements or per-label routing benchmarks.

| Schema | Positive p50 ms | Positive + negative p50 ms |
| --- | --- | --- |
| Baseline | 349.6 | 569.9 |
| Balanced | 358.5 | 594.4 |
| English-preserving | 348.4 | 572.7 |

## Reproduction and next use

```sh
uv run python -m system_one_models.filter_comparison \
  results/opposition-v4-baseline.json \
  results/opposition-v4-balanced.json \
  results/opposition-v4-english-preserving.json \
  --output results/filter-comparison-v4.json
```

The tool verifies schema hashes, recomputes correctness and opposition masks from raw outputs, rejects inconsistent saved metrics, and requires matching dataset hashes. The output retains source hashes and runtime metadata. It has no model-loading or inference path.

Use this tool on subsequent opposition reports to retain threshold curves and matched-coverage comparisons. Confidence remains uncalibrated for this application. Representative calibration and held-out policy assessment are still pending. Keep opposition as a measured comparison; these results do not justify making it mandatory for every accepted answer.
