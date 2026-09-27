# Base evaluation label review

The review below is historical. The five rubric boundaries are now resolved in
[annotation rubric v1](../docs/ANNOTATION_RUBRIC.md), with a full paired review
and six corrected values in `base_eval_v4.jsonl`. v1/v2/v3 labels and historical
reports remain unchanged.

Reviewed on 2026-09-26 against `src/system_one_models/questions.py`, with an
independent Astra subagent using high reasoning. All 88 expected values were
reviewed from the message text and question definitions, independently of model
predictions. This is an annotation review, not verification of sender identity
or an empirical accuracy assessment.

## Correction

`email_bank_statement.message_hook` changed from `invoice_or_debt` to
`none_or_other_unclear`. A monthly bank statement does not establish an invoice,
debt, or payment obligation. The message explicitly says no action is needed.
No other defined hook fits. The remaining 87 values were retained; the judgment
calls below should not be treated as uniquely determined ground truth.

## Assessment by example

| Example | Assessment of all eight expected values |
| --- | --- |
| `sms_delivery_fee` | Seven values supported; `sentiment=sad` is debatable because the adverse notice is worded matter-of-factly. Retained under the broad negative/distressing criterion. |
| `email_bank_statement` | Hook corrected as above; the other seven values are supported. Optional viewing does not override the explicit no-action statement. |
| `sms_retail_promotion` | Classification, industry, sentiment, hook, and sensitive-data values are supported. Urgency and the two action values depend on the boundaries below; retained. |
| `email_government_rebate` | Seven values supported; the hook overlaps government and reward framing. Two types of sensitive data submitted together are one disclosure action. |
| `sms_job_onboarding_fee` | All eight supported. Congratulations conveys positive tone. Paying is the requested action; reserving the position is its purpose. |
| `email_remote_support` | All eight supported. Critical-virus/files-at-risk wording conveys alarm. Calling and installing are separate actions; installation is the more consequential action. |
| `sms_clinic_reminder` | All eight supported if alternative actions are excluded from multiple actions. An appointment time is not a deadline to confirm. |
| `sms_bank_security_code` | All eight defensible. Sad versus neutral remains a tone judgment. Replying is the mechanism for disclosing the security code, not an additional action. |
| `email_parcel_tracking` | All eight supported. Explicitly informational delivery confirmation. |
| `email_utility_bill` | Seven values supported. Urgency depends on whether an ordinary distant due date counts. Alternative payment methods remain one action. |
| `sms_prize_claim` | All eight supported. Paying shipping is the consequential requested action; following the link and claiming the prize are steps/purpose rather than independent actions. Retail is a reasonable industry for the product giveaway. |

## Rubric boundaries to clarify

- **Tone:** `sentiment` asks what the sender expresses, but `sad` includes negative
  or distressing tone. A factual adverse notice can be neutral or alarming.
  `sms_delivery_fee` could reasonably be neutral; the bank warning can reasonably
  be sad due to its threatened immediate transfer. Choose an annotation rule
  before treating disagreements as model errors.
- **Urgency:** The instructions combine pressure to act soon with meeting a
  deadline. Retaining `email_utility_bill.urgent_action=false` assumes ordinary
  next-month due dates are not urgency. An any-deadline interpretation would
  require true. The Friday promotion's true label assumes an expiring offer
  supplies deadline pressure; a deadline alone versus explicit pressure needs
  a consistent rule.
- **Actions:** The promotion treats STOP as its only explicit action, with buying
  left implicit. If implicit purchase invitations count, `purchase_or_claim`
  could be primary and shopping plus opting out could count as multiple actions.
  Clarify whether optional opt-outs count and exclude alternative methods from
  the count explicitly. The clinic and utility labels already assume alternatives
  do not count as independent actions.
- **Hooks:** The rebate fits both `government_or_legal_matter` (claimed authority)
  and `prize_refund_or_gift` (monetary benefit). Retain government pending an
  explicit tie-break rule; refund/reward is also defensible.

The scam/legitimate labels represent the intended synthetic scenarios and
textual suspicion judgments. Message content alone cannot establish that an
apparently legitimate sender is authentic or a request is expected.
