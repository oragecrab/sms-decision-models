# Decision-model guidelines and next steps

Research recorded on 2026-09-26. This document separates guidance from model
authors, findings from our experiments, and proposed work. Sources describe
specific models and versions; their guarantees do not automatically transfer to
another backend.

## Current project decisions

- Keep instructions, questions, category names and descriptions in **English**
  for both English and French messages. This is the CLI default
  (`--question-language en`). The French schema remains an optional experiment.
- Evaluate both message languages. The default `base_eval_v3.jsonl` has 44
  messages representing 22 scenarios, each in English and French with the same
  expected labels. Laya routes message text to its English or multilingual
  checkpoint. Reports retain actual routing decisions.
- Preserve ordinary predictions as the baseline. Positive/negative confirmation
  and third-request resolution remain experimental filters, not proof that an
  answer is correct.
- Keep complete-input and complete-question token checks. Reject inputs that
  cannot be evaluated without truncation.
- Treat all current datasets as synthetic development checks. They are not
  representative production data or an independent holdout. A translation pair
  supplies two language variants of one scenario, not two independent scenarios.

## Guidelines from the research

### 1. Ask one focused judgment per question

TypeSafe recommends narrow judgments that a knowledgeable person can make
quickly from the supplied context. Define the precise condition and explicitly
cover boundary cases. When several factors determine the outcome, ask about each
factor and combine answers in code. Question IDs are not a substitute for full
instructions. Choice represents one category; Noul represents whether one
proposition is true; Score represents a defined ordinal rubric.

Source: [TypeSafe question-design guidance](https://docs.typesafe.ai/primitives).

### 2. Align instructions, criteria and label wording

Give each option a clear meaning; include an other/not-stated outcome when the
answer space needs it. Avoid instructions and criteria that ask different
questions. Laya documents label sensitivity: boolean-word Choice labels can
bias decisions, and Noul can follow labels instead of the message. Its documented
Noul workaround uses neutral A/B model-facing labels while retaining true/false
semantic slots. We already use that mapping; it still needs validation on our
states and checkpoint.

Sources: [TypeSafe primitives](https://docs.typesafe.ai/primitives),
[Laya's documented limitations](https://github.com/NandhaKishorM/laya#honest-limits).

### 3. Test wording, order and negation; do not assume logical identities

Laya documents changes from option order, wording and negated requests. Test
semantically equivalent formulations, shuffled option order and boundary cases
against reviewed labels. Such tests reveal sensitivity; stability alone does not
establish correctness.

TypeSafe explicitly warns that a proposition and its negation need not have
probabilities summing to one. Choice and one Noul per category also have different
semantics: Choice compares alternatives, whereas each Noul judges a proposition.
Do not transfer thresholds between these forms without evaluation. A negative
veto of one category does not vote for a particular alternative category.

Sources: [Laya benchmark limitations](https://nandhakishorm.github.io/laya/benchmarks/),
[Jev 1.13 limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13).

### 4. Keep exact computation in code and provide relevant evidence

TypeSafe describes weaknesses in counting, arithmetic, dates and complex
indirection. Count explicit candidates in code; use the model for semantic
judgments about candidates. Supply the relevant state fields and avoid unrelated
context. Preserve the evidence needed for the decision when filtering text.

Source: [Jev 1.13 limitations and alternatives](https://docs.typesafe.ai/model-jaggedness/jev-1.13).

### 5. Calibrate confidence on the intended workflow

Laya and Decider recommend calibration on representative labeled examples from
the application before using confidence to route decisions. Laya documents both
overconfidence and task-dependent calibration behavior. There is no universal
acceptance threshold that transfers across checkpoints, question types or
languages. Select thresholds using measured error costs and tolerated coverage.

Temperature scaling adjusts probability sharpness and preserves category rank;
it generally does not fix the winning answer. It can improve confidence-based
acceptance decisions. Use separate data to fit calibration and to assess the
final policy. A temperature fallback warning does not establish calibrated
confidence. Zerank's relevance weights are not calibrated class probabilities;
our adapter already marks probability metrics unsupported for that backend.

Sources: [Laya rollout guidance](https://nandhakishorm.github.io/laya/staged-adoption/),
[Decider model card](https://huggingface.co/Mapika/decider-4b),
[Guo et al., On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).

### 6. Distinguish repeatability, accuracy and acceptance coverage

TypeSafe's self-consistency cookbooks measure repeated-answer stability and
illustrate an uncertain/review outcome. They explicitly distinguish repeatability
from correctness; their sample thresholds are illustrative. They do not establish
that our positive/negative or third-request policies improve accuracy.

Evaluate agreement with reviewed expected answers. Report accepted accuracy,
accepted and rejected errors, and the fraction accepted. Compare competing
filters at the same coverage. Review disagreements rather than automatically
treating one formulation as authoritative.

Sources: [TypeSafe Choice self-consistency cookbook](https://docs.typesafe.ai/cookbooks/consistency_choice_cookbook),
[Noul self-consistency cookbook](https://docs.typesafe.ai/cookbooks/consistency_noul_cookbook).

### 7. Validate deployment in stages

Laya recommends observing representative traffic before enabling actions,
comparing decisions with reviewed outcomes, and enabling only evaluated portions
of the workflow. Record the model and schema versions, threshold policy, errors,
review/fallback rate and latency. Re-evaluate when these change and continue
sampling accepted decisions. A fixed output type does not guarantee the chosen
value is correct.

Source: [Laya staged adoption](https://nandhakishorm.github.io/laya/staged-adoption/).

## What our experiments established

These are development-set observations, not claims about unseen messages:

| Experiment | Result |
|---|---|
| English positive/negative check | Baseline 79/88 correct. Accepted 42/45 correct, covering 45/88 labels. Rejected six errors and 37 correct answers. Three errors remained accepted. |
| English third request on disputes | Fixed no errors and broke four correct answers. Replacing disputed answers reduced total accuracy from 89.8% to 85.2%. |
| Paired bilingual messages, English schema | English 149/176 correct (84.7%); French 104/176 (59.1%). Cross-language answers agreed on 102/176 labels. |
| French schema on French messages | French accuracy fell to 86/176 (48.9%). Fourteen errors were fixed, but 32 correct answers were broken. English outputs were unchanged. |

The paired comparison changes message language and the checkpoint selected by
Laya's router. It does not isolate a pure language effect. The French-schema
experiment changes both question wording and category names. Its result supports
keeping the existing English schema for now, not a general claim that English
instructions are always better.

The first rubric-aligned English schema variant on reviewed v4 labels scored
131/176 English (74.4%) and 99/176 French (56.2%), below the saved baseline
rescored on v4 (150/176 and 101/176). It remains experimental. See
[schema-variant results](../datasets/SCHEMA_VARIANTS.md). No matching-language
questions were run.

Detailed records: [model comparison](../datasets/MODEL_COMPARISON.md),
[bilingual v2](../datasets/BILINGUAL_V2.md),
[paired v3 and French-schema results](../datasets/PAIRED_V3.md).

## Proposed implementation order

### 1. Resolve rubric ambiguities before further tuning

**Completed for the paired synthetic set.** [Annotation rubric v1](ANNOTATION_RUBRIC.md)
resolves the following boundaries and records a review of all 44 messages. The
reviewed `base_eval_v4.jsonl` corrects six values across three translation pairs;
v3 and historical scores are preserved. Baseline prompts remain unchanged pending
a separately versioned rubric-aligned schema. The resolved questions were:

- Whether urgency means any deadline or actual pressure to act soon.
- Whether optional opt-outs and implicit purchase invitations count as actions.
- Whether primary action means a navigation step or the ultimate requested task.
- How factual adverse notices are labeled for sentiment.
- How overlapping message hooks are prioritized.

Rules and examples were written before applying label corrections; both
translations were reviewed against the same rules without model outputs. Future
annotations should use that rubric and flag unresolved cases for review.

### 2. Obtain a reviewed, unseen evaluation set when available

**Pending real data.** We expanded the synthetic set and added translation pairs,
but this does not replace a representative holdout. Keep development, calibration
and final test sets separate. Group by `pair_id`, and also keep near-duplicates,
paraphrases and related message templates together. Review ambiguous labels and
allow an explicit unknown/review outcome where appropriate.

### 3. Make evaluation accept explicit question-schema variants

**Implemented for ordinary evaluation.** `--question-schema PATH` accepts
versioned English schema JSON, preserving canonical IDs, question types and
criterion keys while allowing wording and order variants. Dataset validation,
scoring and language/pair summaries receive the selected schema explicitly.
Reports retain version, an order-sensitive hash and exact model-facing schemas.
`schemas/rubric-v1.json` encodes annotation rubric v1 without overwriting the
baseline. Custom schemas now support consistency; French-schema modes remain excluded.
Controlled comparisons can use the paired v4 development set; isolated wording
and option-order experiments remain pending.

### 4. Add controlled wording and option-order experiments

**Initial broad sweep implemented and measured.** See
[254-definition English-schema sweep and grouping results](../datasets/SCHEMA_SWEEP.md).
It covers both message languages, exact schema hashes and token-budget rejections.
The selected composite remains experimental, with no held-out assessment.
Further isolated wording/order experiments and independent validation remain useful.

Original plan: Use a fixed set of paraphrases and option-order permutations. Begin
with classification, requested action and multiple actions. Include negations and
boundary cases in both languages. Measure accuracy, answer-change frequency and
per-language differences. Select variants on development data; evaluate the chosen
schema once on the untouched final test set when available.

### 5. Experiment with decomposing multiple actions

**Three initial prototypes implemented and measured; none improved overall accuracy.**
See [action-decomposition results](../datasets/ACTION_DECOMPOSITION_V1.md).
Source-span proposals, Noul/Choice request gates and categorical/split-factor pair
judgments were compared with original and balanced questions under fixed FP32.
Source/rubric diagnostics expose poor extraction precision and unsupported pairs.
Actual opposition checks and exact artifacts are retained. The subsequent
[reviewed boundary and mixed-context study](../datasets/ACTION_IMPROVEMENT_V2.md)
compared eight Laya configurations, five grouped GLiNER schemas and single-head
execution. Reviewed-span diagnostics identify request attribution as a bottleneck.
A fixed binary GLiNER decoder improved fresh-test accuracy to 54/64 versus 41/64
for balanced Laya, but missed five multi-task messages versus one. It remains
experimental and opt-in because positive recall decreased. Actual opposition
agreement and matched-coverage confidence diagnostics are retained; neither
small synthetic sample establishes production calibration.

Original plan: Identify candidate requests, judge whether they are requested
together and whether they are independent tasks, then count in code. Preserve
alternatives, steps and optional requests explicitly. Simply counting positive
action categories is insufficient: clicking a link to pay can activate two
categories while representing one task. Candidate extraction and relationship
judgments also need evaluation. Compare this approach with the existing Noul;
do not assume decomposition is an improvement.

### 6. Compare confidence filtering with agreement filtering

**Offline comparison tooling implemented and measured.** See
[confidence versus opposition at matched coverage](../datasets/FILTER_COMPARISON_V4.md).
Saved outputs supply full threshold curves, matched question/language acceptance
counts, tie diagnostics, error breakdowns and recorded latency without inference.
Representative calibration and held-out policy assessment remain pending.

Original plan: Add evaluation plumbing that sweeps thresholds and reports
accepted accuracy, coverage, accepted/rejected errors and latency. Compare filters
at the same acceptance rate, per question and language. Include false positives
and false negatives—especially missed scams and falsely flagged routine messages.
Use confidence-gate calibration and threshold selection only once sufficient
representative reviewed data is available; keep fitting separate from final
assessment. The current synthetic set can exercise the tooling, not validate
production thresholds. Keep negative/third requests as comparison policies.

### 7. Improve reproducibility and plan a measured rollout

**Partially implemented.** Complete-input checks, dataset hashes, model metadata,
package versions, timing, language summaries and Laya routing records exist.
Laya's resolved checkpoint revision and calibration settings still need fuller
recording/pinning. Retain these checks and rerun evaluations after changes to
models, schemas or calibration. Once representative evidence exists, observe
traffic first, then enable an evaluated subset with review/fallback and clear
rollback conditions.

## Immediate next work without a larger dataset

Annotation rules, the paired review and general schema-variant evaluation are
complete. The first action-decomposition and matched-coverage filter comparisons are
measured. New reviewed boundary and mixed-context cases now test improved
components and a frozen binary task-count policy. Remaining errors concentrate
on optional opt-outs, task steps/fields and alternatives. Next improve these
boundaries using development evidence, then assess on new representative reviewed
messages; preserve language-specific recall alongside accuracy. Apply the
confidence-versus-agreement comparison tooling to subsequent reports. Use the paired synthetic examples
for regression and sensitivity experiments. Keep English questions and avoid
promoting a threshold or prompt revision solely because it improves these small,
previously inspected examples.

## Retained opposition metric

Every future schema comparison must include independently inferred positive/negative
opposition agreement alongside accuracy. Record agreement counts, coverage,
accuracy among agreements and accepted/rejected errors per question and language.
Do not infer the negative answer by complementing the positive probability.
Current baseline and both selected schemas were measured on reviewed v4:
[opposition results](../datasets/OPPOSITION_V4.md). General schema sweeps now save
baseline and selected opposition reports automatically; evaluate additional
finalists with the same dataset/runtime using `--consistency --question-schema`.
