# Paired English/French development dataset

`base_eval_v3.jsonl` contains 44 messages: 22 scenarios, each in English and
French. It is now the default demo/evaluation dataset. The v1 and v2 files remain
unchanged for reproducing earlier results.

The 11 original English messages were translated into French, and the 11 newer
French messages were translated into English. The original states and expected
labels are preserved. A translation copies all eight expected labels from its
source. Bodies, subjects and descriptive sender text are translated; proper
names, email addresses, URLs and reply codes are retained. Amounts and times are
expressed in the target language without changing their values. Canadian French
`déjeuner` in the hotel offer is interpreted as breakfast. Deadlines, emotional
framing, negations, and alternatives versus sequential requests are preserved.
Translations were authored before seeing predictions and have not been reviewed
by an independent bilingual annotator. Earlier rubric ambiguities still apply.

Metadata outside `state` records:

- `pair_id`: shared scenario identity, based on its original example ID.
- `source_language`: the language in which the scenario was originally authored.
- `is_translation`: distinguishes the source message from its translation.
- `language`: language of this specific message (`en` or `fr`).

The expected labels and output category keys stay identical across each pair.
The question schema remains in English. Model inputs contain only `state`; pair
identity, provenance, expected labels and language annotations are not supplied.
Laya routes using the message text. Evaluation reports include all per-message
answers, language accuracy, source-cohort accuracy, pair agreement, the differing
questions for each scenario, and counts of English-right/French-wrong and the
reverse for each question. Agreement can include answers that are both wrong.

If these examples are split into development/calibration/test sets later, keep
both translations of a scenario together by `pair_id`. Forty-four messages
represent 22 semantic scenarios, not 44 independent scenarios. The existing
English examples were used in prompt tuning, so these remain development checks.

## Initial Laya measurement

Measured on 2026-09-26, without changing prompts or labels after seeing results.

| Source cohort | English correct labels | French correct labels |
|---|---:|---:|
| Original 11 English scenarios | 79/88 (89.8%) | 53/88 (60.2%) |
| Newer 11 French scenarios | 70/88 (79.5%) | 51/88 (58.0%) |
| All 22 paired scenarios | 149/176 (84.7%) | 104/176 (59.1%) |

Six English messages matched all eight expected answers; no French message did.
Predictions matched between English and French on 102/176 paired labels (58.0%).
None of the 22 pairs had matching predictions on all eight questions.

| Question | Same answer across translations |
|---|---:|
| Classification | 14/22 |
| Industry | 13/22 |
| Urgency | 17/22 |
| Sentiment | 16/22 |
| Requested action | 8/22 |
| Multiple actions | 10/22 |
| Message hook | 13/22 |
| Sensitive data requested | 11/22 |

For example, the bank-statement, clinic-reminder and utility-bill sources were
classified as legitimate in English but suspected scams in French. The
French government-rebate translation corrected the English classification miss.
The full pair-by-pair predictions and gold labels are in the JSON report.

All 22 English requests used the English checkpoint and all 22 French requests
used the multilingual checkpoint. This compares the deployed routing workflow:
translation, checkpoint choice and an English question schema. It does not
isolate language from checkpoint behavior or establish performance on real
traffic. The original English and French cohorts reproduced their v2 accuracy.

Run:

```sh
uv run system-one-models --eval --model laya --output results/laya-paired-v3.json
```

The completed local report is `results/laya-paired-v3.json`. To use the earlier
unpaired balanced set, pass `--dataset datasets/base_eval_v2.jsonl`.

## French question and category schema

`french_questions.py` provides French translations of all eight question
instructions, category names, category descriptions and boolean criteria.
Category order and true/false semantic slots are unchanged. Boolean A/B labels
remain neutral; English question IDs and canonical output keys remain stable in
code. French choice outputs and their distributions are mapped back to the
original category keys before scoring. Reports retain both raw French-named
answers and canonical answers, the exact French schema and its label mapping.

`--question-language match` uses the French schema on annotated French messages
and the original English schema on English messages. `--question-language fr`
selects French questions for every message; `en` remains the default. These
options currently apply to ordinary `--eval`; negative/third-request templates
remain English and combining French schema selection with consistency is rejected
rather than silently mixing prompt languages. Message routing still depends only
on the message state, not the question-language annotation.

Matched-language evaluation was measured on the same 44 messages, with unchanged
expected labels and routing:

| Message language | English questions/categories | Matched questions/categories |
|---|---:|---:|
| English | 149/176 (84.7%) | 149/176 (84.7%) |
| French | 104/176 (59.1%) | 86/176 (48.9%) |

All English outputs reproduced the English-schema run. On French messages,
French questions corrected 14 labels and broke 32 previously correct labels,
for a net loss of 18 correct labels. Urgency improved from 17/22 to 18/22,
but multiple actions fell from 7/22 to 5/22; requested action fell from 10/22
to 7/22 and classification from 15/22 to 11/22. No French message matched all
eight labels. Cross-translation agreement fell from 102/176 to 87/176.

This changes both instructions/descriptions and model-facing category names,
so it does not separate the effects of those changes. French text also has
different tokenizer representations. All complete inputs passed the existing
no-truncation guards; no token budgets, checkpoint choices or expected labels
were relaxed to obtain these results. No alternate French schemas were selected
by trying to improve these scores. The translation is an experiment, not a new
recommended default or an independent human-reviewed reference.

Reproduce:

```sh
uv run system-one-models --eval --model laya --question-language match \
  --output results/laya-paired-v3-matched-questions.json
```
