# Independent-task evaluation and frozen candidate

The existing eight-question defaults remain unchanged. A separate opt-in task-count policy tests a cached GLiNER model and a frozen binary decoder. This experiment concerns `multiple_actions` only; its scores are not eight-label demoset accuracy.

All prompts are English for both English and French messages. Matching-language questions were not rerun. Development scenarios and the initial heldout scenarios are disjoint by template group, with translations kept together. Counts refer to translated messages, so the 48-message development set contains 24 scenarios. These reviewed, balanced synthetic cases complement the legacy v4 set, which has only six positive labels among 44 messages.

## Development comparisons

Tested eight Laya configurations: original, balanced, two-category direct, three-category direct, explicit Boolean, target-field decomposition, marked-target decomposition, and marked compact decomposition. Tested five direct GLiNER schemas together, followed by a separate single-head execution check. Positive and complementary questions were actually inferred; negative answers were not synthesized.

| Development method | English correct /24 | French correct /24 | Total /48 | False positives | False negatives |
| --- | ---: | ---: | ---: | ---: | ---: |
| Laya original | 18 | 13 | 31 | 12 | 5 |
| Laya balanced | 17 | 14 | 31 | 12 | 5 |
| Laya direct two-category | 15 | 13 | 28 | 5 | 15 |
| Laya direct three-category | 13 | 14 | 27 | 1 | 20 |
| Laya explicit Boolean | 17 | 14 | 31 | 11 | 6 |
| Laya target-field decomposition | 15 | 13 | 28 | 17 | 3 |
| Laya marked-target decomposition | 15 | 11 | 26 | 17 | 5 |
| Laya marked compact decomposition | 12 | 11 | 23 | 1 | 24 |
| GLiNER original, five-head execution | 13 | 13 | 26 | 0 | 22 |
| GLiNER balanced, five-head execution | 14 | 15 | 29 | 0 | 19 |
| GLiNER two-category, five-head execution | 16 | 17 | 33 | 15 | 0 |
| GLiNER three-category, five-head execution | 18 | 19 | 37 | 10 | 1 |
| GLiNER explicit Boolean, five-head execution | 12 | 12 | 24 | 1 | 23 |
| GLiNER three-category, single-head execution | 20 | 16 | 36 | 2 | 10 |

GLiNER renders multiple heads together. Running the selected schema alone changed 17/48 decisions; grouping is part of the frozen execution contract. This is a substantial semantic/runtime sensitivity, not evidence that an arbitrary single-head call reproduces the five-head policy. The policy preserves exact question order, companion heads, negative heads, model snapshot and package identity. Laya runs use fixed FP32 and two threads; the cached GLiNER MPS model is FP32. Token preflights prevent truncation.

## Initial candidate and regression guard

The initial categorical policy classified a message as multiple tasks when `multiple_tasks` won the three-category argmax. It improved the first routine-context heldout set, but failed the legacy v4 regression check, especially English. The first heldout set therefore does not justify adopting this policy.

| Method | Initial heldout correct /32 | Legacy v4 English /22 | Legacy v4 French /22 | Legacy v4 total /44 |
| --- | ---: | ---: | ---: | ---: |
| Laya original | 19 | 17 | 7 | 24 |
| Laya balanced | 17 | 17 | 9 | 26 |
| GLiNER categorical decoder | 25 | 11 | 9 | 20 |

For a binary task decision, multiclass argmax can select `multiple_tasks` even when the combined probability of `zero_tasks` and `one_task` is larger. A separately versioned decoder compares the actual `multiple_tasks` probability with that combined mass using a fixed equal-cost cutoff of 0.5. Exact ties remain unresolved. The raw multiclass answer is retained, and probabilities are not assumed calibrated.

This decoder was characterized using saved development and legacy v4 outputs only. It was frozen before testing a fresh mixed-context set; the previous routine heldout was not used to tune this decoder. Development and legacy scores are now development evidence rather than untouched validation.

| Frozen binary decoder | English correct | French correct | Total correct | False positives | False negatives |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development | 20/24 | 18/24 | 38/48 (79.2%) | 2 | 8 |
| Legacy v4 | 20/22 | 21/22 | 41/44 (93.2%) | 3 | 0 |

The binary decoder trades recall for fewer false positives on development. Report both error types and language-specific performance, rather than treating the higher aggregate accuracy as sufficient evidence.

## Fresh mixed-context assessment

The frozen binary policy improves accuracy on this fresh test, but fails the predeclared requirement to preserve multi-task recall in each language. It therefore remains experimental and does not replace defaults. Model, companion heads, wording and decoder were fixed before inference; no adjustment followed the result.

| Fresh mixed test | English correct /32 | French correct /32 | Total /64 | False positives | False negatives |
| --- | ---: | ---: | ---: | ---: | ---: |
| Laya balanced | 23 | 18 | 41 (64.1%) | 22 | 1 |
| Frozen GLiNER binary | 27 | 27 | 54 (84.4%) | 5 | 5 |

| Stratum | Laya balanced correct /32 | Frozen GLiNER binary correct /32 |
| --- | ---: | ---: |
| Routine | 21 | 27 |
| Suspicious | 20 | 27 |

Multi-task recall is 14/16 English and 13/16 French for the binary policy, versus 15/16 and 16/16 for balanced Laya. Fewer false alarms come with additional missed multi-task requests in both languages. The gain in aggregate accuracy does not satisfy that recall guard.

The new test contains 16 suspicious and 16 routine scenarios, each stratum balanced between eight multi-task and eight single/no-task scenarios. A separate reviewer checked all English/French translations and task labels against the annotation rubric and compared concrete scenarios with both previous datasets. Eight draft scenarios were replaced for being too close to earlier templates before inference. Risk strata describe intended textual framing, not verified fraud or authenticity. Author and independent-review artifacts are retained with hashes.

## Opposition and component diagnostics

Opposition is a separately inferred complementary decision, not the mathematical complement of the positive output. It remains recorded but is not an acceptance guarantee: the initial categorical policy's opposition accepted seven initial-heldout messages, six incorrectly. On legacy v4 it accepted 24 messages, all incorrect. Under the frozen binary decoder, development accepted only three correct messages, while legacy v4 accepted three incorrect messages. On the fresh mixed test it accepted six messages, three incorrectly (English: one accepted, one wrong; French: five accepted, two wrong). An offline confidence ranking at exactly the same per-language acceptance counts retained zero errors, without ties at those cutoffs. These six accepted cases cover only 9.4% of the test, so this is a small diagnostic rather than a tuned threshold or production guarantee. Such tiny and inconsistent opposition coverage cannot support a production filter.

The target-field decomposition gate reached 47.7% precision and 96.9% recall on development. Supplying independently reviewed request spans diagnostically improved its final message classification to 44/48 (24/24 English, 20/24 French) and its independent-pair witness precision from 15.4% to 90.9%. This identifies request attribution as an important bottleneck. Reviewed spans are diagnostic inputs only; they are unavailable in an automated deployment and do not count as end-to-end accuracy. Marked compact relationships failed separately even with correct spans.

Cached JevK5 loaded on CPU FP32 with two threads, but reference convolution/attention kernels took approximately 60 seconds for one ten-head warmup message. The 48-row suite projected approximately 48 minutes. It was stopped during French warmup before any scored rows; its report contains runtime provenance and no accuracy estimate. No downloads or package installs were performed.

## Opt-in use and artifacts

Use the frozen policy through the separate single-message CLI. It returns the task-count result; it does not replace the other seven questions or change default evaluation schemas.

```sh
HF_HUB_OFFLINE=1 .venv-gliner/bin/python -m system_one_models \
  --task-count-policy schemas/task-count-gliner-binary-v2.json \
  --device mps --channel sms --body 'MESSAGE TO EVALUATE' \
  --task-count-opposition
```

The exported experimental policy is `schemas/task-count-gliner-binary-v2.json`. Its opposition flag reports the independent check without making agreement a guarantee of correctness. The separate `.venv-gliner` environment contains the pinned evaluated GLiNER packages.

Key reports:

- `results/action-improvement-development-v1.json`: eight Laya development configurations.
- `results/action-improvement-gliner-development-v1.json`: five grouped GLiNER configurations.
- `results/action-improvement-gliner-single-development-v1.json`: single-head sensitivity comparison.
- `results/action-improvement-heldout-policy-v1.json` and `results/action-improvement-heldout-comparators-v1.json`: initial categorical heldout assessment.
- `results/action-improvement-v4-regression-v1.json`: initial categorical regression failure.
- `results/action-binary-development-v2.json` and `results/action-binary-v4-regression-v2.json`: fixed binary decoding of saved actual outputs, without new inference.
- `results/task-count-frozen-binary-v2.json`: frozen exact model, companion heads and binary decoder.
- `results/action-binary-mixed-heldout-v2.json` and `results/action-mixed-comparators-v2.json`: fresh mixed-context assessment of the frozen binary policy and fixed Laya comparators.
- `schemas/task-count-gliner-binary-v2.json`: exported opt-in experimental policy.
- `results/action-focused-component-audit-v1.json`: request-gate and supplied-span relationship diagnostics.
- `results/action-improvement-jev-development-v1.json`: runtime-aborted zero-scored JevK5 observation.

Recorded inference timing is observational and includes concurrent local experiments; it is not a rigorous hardware benchmark. Shared five-head durations cannot be divided or credited as independently measured single-head latency. These synthetic, small, translation-paired sets support an experimental comparison; further reviewed real-world evaluation is needed to establish generalization, class prevalence or reliable confidence thresholds.
