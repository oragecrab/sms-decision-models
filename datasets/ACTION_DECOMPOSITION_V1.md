# Independent-action decomposition experiment

Tested three source-span decomposition prototypes on reviewed v4: 44 messages, 22 paired scenarios, English questions for both message languages. **None improved overall multiple-action accuracy.** All remain experimental; the ordinary question defaults are unchanged.

## Accuracy

| Method | English correct /22 | French correct /22 | Total correct /44 | False positives | False negatives |
| --- | --- | --- | --- | --- | --- |
| Original question | 17 | 7 | 24 (54.5%) | 20 | 0 |
| Balanced question | 17 | 9 | 26 (59.1%) | 18 | 0 |
| Noul gate + relationship Choice | 7 | 12 | 19 (43.2%) | 23 | 2 |
| Choice gate + relationship Choice | 9 | 12 | 21 (47.7%) | 19 | 4 |
| Choice gate + joint/independence Noul | 19 | 3 | 22 (50.0%) | 19 | 3 |

The first two prototypes improve French from 9/22 to 12/22 compared with the balanced question, while substantially regressing English. Splitting the relationship into two Noul questions produces false for every English message and true for every French message: it misses all three English multi-task cases and falsely labels all 19 French single/no-task cases. This behavior is tied to the tested schemas/checkpoints; it does not establish a general property of decomposition.

Only six of the 44 messages are genuinely multi-task. An always-false reference scores 38/44 (86.4%) but misses all six; it is not a proposed solution. Accuracy alone therefore hides important recall failures. The table retains false positives and false negatives.

## Method

1. Propose spans using a generic punctuation/conjunction splitter over subject/body. Preserve source text and offsets. No message-specific rules or gold labels enter proposals. There were 156 spans, at most six per message.
2. Judge whether each span explicitly requests a recipient operation in full context. The first prototype uses Noul; the other two share a two-category Choice gate. Shared verbs, conditions, negation and optional opt-outs must be interpreted from the complete message.
3. For selected spans, either classify pairs as independent together, same-task steps, alternatives, unsupported or unclear; or ask two separate Noul questions: are both actually requested, and are they independent tasks?
4. Code counts supported independent-pair witnesses and returns true if at least one exists. A witness supports a lower bound of two tasks; pair counts are not an exact task count. Exact ties and unclear pair answers are retained as unresolved. Binary accuracy includes these unresolved cases; this is an experiment, not a deployed review policy.

Laya is a closed-choice decision model, so this is deterministic span proposal followed by semantic selection, not generative extraction. Every stage retains full original state, exact question schemas, raw outputs, routed checkpoint identity and timing. Token checks rejected truncation; all scored inputs fit. No French-question or matching-language-schema experiments ran.

All local MPS inference uses fixed FP32 (`LAYA_MPS_AMP_MIN_ROWS=10000`) with two CPU threads. Variable candidate/pair counts therefore cannot switch autocast precision. The original and balanced questions were freshly evaluated as isolated questions in the same runtime; their 17/7 and 17/9 counts reproduce the earlier fixed-FP32 sweep. Do not compare these directly with default-MPS fp16 counts.

## Component review

A separate source/rubric review annotated requested spans and task groups without consulting model outputs or gold labels during annotation. Earlier agent context included dataset examples, so this is diagnostic review, not a blinded holdout. It found 50 requested spans and no missing operation in these proposals. It notes oversplitting of inherited negation, shared fields, alternatives and promotional benefits.

| Prototype | Gate precision | Gate recall | Incorrect independent-pair witnesses / all witnesses |
| --- | --- | --- | --- |
| Noul gate + relationship Choice | 38.1% | 96.0% | 88/94 |
| Choice gate + relationship Choice | 38.1% | 90.0% | 66/70 |
| Choice gate + joint/independence Noul | 38.1% | 90.0% | 83/86 |

The request gate selects many nonrequests: the original gate retains 78 false candidate spans, the categorical gate 73. Relationship judgments also fail to reject unsupported pairs. Every false-positive message from the original prototype has at least one witness containing a nonrequested span. Some also miscount fields or alternatives; causes can overlap.

The original prototype misses two genuine multi-task messages: one loses its true pair at the gate, and one keeps the spans but fails the relationship judgment. The categorical-gate prototype misses four: two extraction failures and two relationship failures. Full per-language diagnostics and examples are saved.

## Opposition agreement

Each decomposition message received an actual, separate complementary question asking whether at most one independent task was requested, with no positive result or audit label supplied. Since this question is identical across the three semantically equivalent decomposition variants, its raw fixed-FP32 result is reused across variants. No negative probability is fabricated. Original/balanced questions receive their own schema-derived negative requests.

| Method | Agreed /44 | Accuracy among agreed | Agreed but wrong | Errors rejected |
| --- | --- | --- | --- | --- |
| Original question | 32 (72.7%) | 40.6% | 19 | 1 |
| Balanced question | 32 (72.7%) | 62.5% | 12 | 6 |
| Noul gate + relationship Choice | 34 (77.3%) | 52.9% | 16 | 9 |
| Choice gate + relationship Choice | 28 (63.6%) | 57.1% | 12 | 11 |
| Choice gate + joint/independence Noul | 5 (11.4%) | 100.0% | 0 | 22 |

These filters accept different cases and coverage, so this table does not establish superiority at matched acceptance rates. The split prototype agrees on just five English negative cases; its 100% accepted accuracy omits every genuine multi-task case. Agreement still requires correctness and coverage context.

## Latency and saved artifacts

Positive inference p50 was 52.0 ms for the original question, 48.3 ms for balanced and 292.6 ms for the first decomposition prototype. The fresh categorical-gate plus split-factor pipeline was 273.1 ms. The categorical-gate plus Choice experiment reused identical pair outputs, so its recorded time is incremental work and cannot be treated as end-to-end latency. Timing is an approximate sequential workload observation, not a rigorous benchmark.

- `results/action-decomposition-v1.json`: first prototype and fresh comparators/opposition.
- `results/action-decomposition-variants-v1.json`: categorical-gate and split-factor comparisons, exact cached-output provenance.
- `results/action-decomposition-audit-v1.json`: extraction and relationship diagnostics.
- `results/action-decomposition-comparison-v1.csv`: per-message decisions for every method.
- `datasets/action_span_audit_v1.json`: source/rubric annotations and notes.
- `results/action-decomposition-harness-v1/`: exact executed code snapshots and source report matching cached hashes. Report-only field corrections preserve predictions; saved snapshots identify executed code before later validation fixes.

```sh
HF_HUB_OFFLINE=1 uv run python -m system_one_models.action_decomposition \
  --dataset datasets/base_eval_v4.jsonl --device mps \
  --output results/action-decomposition-next.json

HF_HUB_OFFLINE=1 uv run python -m system_one_models.action_decomposition_variants \
  --source results/action-decomposition-next.json \
  --dataset datasets/base_eval_v4.jsonl --device mps \
  --output results/action-decomposition-variants-next.json

uv run python -m system_one_models.action_decomposition_audit \
  --audit datasets/action_span_audit_v1.json \
  --initial results/action-decomposition-v1.json \
  --variants results/action-decomposition-variants-v1.json \
  --output results/action-decomposition-audit-v1.json
```

## What remains

The current decomposition is not an improvement and should not replace the existing question. A further experiment should first demonstrate reliable request extraction and relationship judgments on independently reviewed boundary cases, then assess the combined policy on unseen scenarios. Repeating final multiple-action wording alone will not resolve the observed span-attribution failures. Candidate extraction and pair classification are themselves evaluation targets.
