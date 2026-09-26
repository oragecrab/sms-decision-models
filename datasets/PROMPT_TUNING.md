# Laya prompt tuning report

Only `src/system_one_models/questions.py` was edited by the tuning agent. The dataset, runtime, all eight question IDs/types, category keys, `max_len=512`, `head_max_len=320`, and complete-prompt/message budget checks are unchanged.

## Measurement

All inference used the locally cached checkpoint with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and reused one Router across each evaluation. Search comprised two full prompt revisions, one small question-variant sweep, and one combined verification. The initial overlong draft was rejected by the existing token check and shortened before inference; token limits were not relaxed.

| Question | Baseline | Final verified demo |
| --- | ---: | ---: |
| classification | 6/11 | 9/11 |
| industry | 10/11 | 11/11 |
| urgent_action | 10/11 | 10/11 |
| sentiment | 9/11 | 11/11 |
| requested_action | 7/11 | 9/11 |
| multiple_actions | 9/11 | 9/11 |
| message_hook | 8/11 | 9/11 |
| sensitive_data_requested | 10/11 | 11/11 |
| All individual labels | 69/88 | 79/88 |
| Fully matching messages | 2/11 | 5/11 |

**Verified on the final source:** the parent ran the real CLI demo and confirmed
5/11 fully matching messages and 79/88 matching labels. All 14 tests passed,
including actual cached-tokenizer checks. Inference used cached checkpoint
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` offline. These prompts were tuned on
this development set; the scores are not held-out accuracy estimates.


## Changes and selection

- Classification now identifies general fraud mechanisms: fees before employment/winnings, unexpected parcel fees, code requests, identity/financial disclosure and unsolicited remote access. Category descriptions distinguish routine service messages and ordinary advertising. The selected risk wording detects 5/6 scam examples rather than 2/6, with one false positive (retail promotion) rather than zero. This safety-oriented tradeoff was selected over an equally scoring 9/11 version that missed two scams and produced no false positives.
- Industry follows the service being provided, not its payment channel. Government/public services explicitly covers utilities.
- Sender tone includes celebratory/reward framing and alarm; it is judged independently of honesty.
- The action options retain their original definitions. A short main-action question measured better (9/11) than longer consequential-action/navigation instructions (8/11 or worse). This is a semantic tradeoff: it expresses priority less explicitly; installation and prize-payment misses remain.
- Multiple-action instructions distinguish independent requests from alternatives, link steps, multiple data fields and inferred purchases. Accuracy is unchanged, with the intended boundary clearer.
- Hook definitions exclude bank statements from debts and appointments from personal emergencies. Government benefits take precedence over generic rewards; this is an explicit convention for previously overlapping hooks. The selected version correctly handles the statement and clinic, but the job message's hook regresses to promotion.
- Sensitive-data wording explicitly asks about government identity numbers, bank/account details, passwords and one-time codes; routine sign-in/payment/opt-out alone remain excluded. Both sensitive-disclosure examples are now positive.
- Original urgency wording was retained because tested clarifications worsened model results. The ordinary-future-due-date ambiguity recorded in `datasets/LABEL_REVIEW.md` remains unresolved.

## Remaining mismatches in selected components

- `sms_retail_promotion`: classification `suspected_scam` vs `marketing`; multiple_actions `true` vs `false`.
- `email_government_rebate`: classification `legitimate` vs `suspected_scam`; message_hook `prize_refund_or_gift` vs `government_or_legal_matter`. Sensitive disclosure is now detected.
- `sms_job_onboarding_fee`: message_hook `sale_or_promotion` vs `job_or_investment`.
- `email_remote_support`: requested_action `share_sensitive_info_or_documents` vs `install_or_grant_access`.
- `email_utility_bill`: urgent_action `true` vs `false`; multiple_actions `true` vs `false`.
- `sms_prize_claim`: requested_action `no_action` vs `pay_or_transfer`.

No rules inspect IDs, exact messages, sender strings, specific amounts, model answers, or expected labels to force outputs. Remaining ambiguities and failures should remain visible. Confidence is not calibrated: the checkpoint warns that its 11+-choice temperature falls outside the accepted range; the runtime is unchanged.

## Additional paraphrase smoke checks

Six fresh synthetic messages were written separately from the tuning agent and
were not used to choose the prompts. They check 21 selected expected fields,
not all eight fields in every message. Results improved from 11/21 to 15/21;
fully matching messages improved from 0/6 to 1/6. These remain small synthetic
smoke checks, not a representative benchmark.

| Case | Final remaining disagreements |
| --- | --- |
| Work-from-home offer requiring a cryptocurrency processing fee | Action predicted approve-login/transaction instead of pay/transfer; sensitive disclosure incorrectly positive. |
| Public-benefit refund requesting ID photo and bank account number | All three checked fields match: suspected scam, sensitive disclosure action, sensitive-data request. |
| Laptop prize requiring a handling payment | Action predicted no action instead of pay/transfer. Classification now matches suspected scam. |
| Municipal water bill due in six weeks, with no immediate action needed | Urgency incorrectly positive. Classification, industry, payment action and single-action count match. |
| Automatic verification-code notice saying never share it and no reply needed | Classification incorrectly suspected scam. No-action and no-sensitive-request fields match. |
| Informational monthly bank activity summary | Hook predicted account/payment problem instead of none/other. |

One sensitive-disclosure false positive was introduced on the cryptocurrency
recruitment-fee paraphrase. Better development scores do not remove this kind of
model error. No deterministic postprocessing was added to force expected answers.

## Reproduce the checks

Run `uv run pytest` for the default suite; set `LAYA_TEST_TOKENIZER` to a local
checkpoint tokenizer directory to include the optional real-tokenizer test.
Run `uv run system-one-models --demo` for full local inference on the base set.
The paraphrase checks used a separate scratch driver and the same guarded
classifier with a reused Router. Their raw records are in
`/private/tmp/system-one-models-smoke-before.json` and
`/private/tmp/system-one-models-smoke-after.json` for this session.
