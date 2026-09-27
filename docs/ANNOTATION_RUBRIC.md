# Annotation rubric v1

Adopted 2026-09-26. This is the annotation contract for
`datasets/base_eval_v4.jsonl`. It resolves the five boundaries recorded in
`LABEL_REVIEW.md`. Rules were chosen from message meaning before reviewing model
outputs. A separate subagent reviewed all 22 scenarios in both languages.
This is a synthetic development-set review, not independent production validation.

## Annotation procedure

Read the complete supplied sender, subject and body. Judge only supplied evidence;
do not open links or infer sender authentication. Apply the same rules to English
and French. Annotate each translation independently, then compare the pair. If
meaning differs, flag the translation instead of forcing shared gold labels.
Record unresolved ambiguity for review outside the eight-label schema; do not
invent an unknown label or silently force a result. Do not use model predictions
as annotation evidence. Retain the reason for every correction.

Classification describes textual suspicion, not verified fraud or authenticity.
Industry describes the claimed service, not its payment method. Sensitive-data
requests require explicit disclosure of a secret credential, identity document or
number, or bank/card details. Wallet recovery phrases are secret credentials;
gift-card/voucher codes used as payment are payment instruments. Routine sign-in
and payment alone do not imply sensitive disclosure. The existing category keys
remain unchanged.

## Urgency: pressure on an action, not every date

Mark `urgent_action=true` for an explicit instruction to act now, immediately,
today or before a near-term cutoff; an explicitly limited offer/benefit window;
or a stated threatened harm if action is delayed. Judge wording without computing
calendar distances or assuming the evaluation date is the message date.

A routine due date or event schedule alone is false. A promised benefit arriving
"today" is false unless the message also sets a cutoff or demands prompt action.
A stated open-ended window or explicit absence of a deadline is false in the
absence of other pressure. Where only a calendar date is given, require additional
pressure (e.g. "last chance", expiration or threatened consequences), rather than
inferring how soon that date is. Urgency can be true for an expiring advertisement
even when no explicit purchase request is counted by the action questions.

| Wording | Urgent | Reason |
| --- | --- | --- |
| "Pay now" / "Payez maintenant" | true | Explicit immediate action. |
| "Save 20% through Friday" / "jusqu'à vendredi" | true | Limited offer window. |
| "Rebate expires today" | true | Explicit imminent expiry. |
| "Bill is due next month" | false | Routine distant due date. |
| "Appointment Tuesday at 10:30" | false | Event time, not response deadline. |
| "To receive your prize today, pay the fee" / "Pour recevoir votre lot aujourd'hui" | false | Benefit timing, no payment deadline. |

## Requested actions: explicit tasks and optional requests

Count concrete directives and explicit invitations, including optional opt-outs.
"Reply STOP" is a request. "Book our package" is a purchase request. Advertising
an available discount alone is not an implicit purchase request. Mere permission
or availability ("you can view it whenever convenient"), especially alongside
"no action needed", is not a request. A link alone is not a directive to click.
Conditional requests count if their condition is presented as applicable; mutually
exclusive alternatives do not count as requests to do both.

For `multiple_actions`, count independent tasks requested together. Do not count
navigation, reply transport, multiple fields in one submission, or payment methods
as separate tasks. "Click to pay" is one payment; "reply with a code" is one
disclosure; buying and sending a gift card to settle a fine is one payment.
"Confirm OR call to reschedule" is one alternative task. "Call support, THEN
install a tool" contains two tasks even though both serve the same repair goal:
contacting support and granting device access are separate requested operations.
"Buy now AND reply STOP if you want to unsubscribe" contains two explicit tasks,
including the optional opt-out; a discount plus STOP alone contains only the latter.
Two independent requests in the same action category still count as two tasks.

## Primary action: the concrete task, not its transport or promised reward

Choose the task explicitly requested of the recipient. Prefer the named downstream
task over the way to reach it: pay over click-to-pay; disclose a code over reply;
install over opening its installer. Do not substitute the sender's promised result
(getting a job, repairing a device, receiving a prize) for the recipient's task.
A fee to release a prize is `pay_or_transfer`; "claim your prize" without another
explicit task is `purchase_or_claim`. Link-only navigation is `click_or_scan`.

Among independent requests, use the task the message explicitly identifies as
main. Otherwise apply this annotation tie-break order (a project convention, not
a claim about universal risk): install/grant access, disclose sensitive
information/documents, pay/transfer, approve login/transaction, update details,
sign in/verify, purchase/claim, open/download attachment, click/scan, reply/contact,
other/unclear. Apply this tie-break only to actual independent requests, never to
invented actions or steps. An explicit opt-out is primary only when no other
higher-priority task is requested. Use `no_action` when there are no requests.

Examples: call then install → installation, multiple=true; send passport and bank
credentials then call recruiter → disclosure, multiple=true; discount plus STOP →
contact, multiple=false; routine bill with two payment methods → payment,
multiple=false.

## Sentiment: expressed framing, not the reader's reaction

`happy` means explicit positive, celebratory, enthusiastic or reward framing.
`sad` retains its existing key but means explicit alarm, threat, worry, frustration
or distress; it does not require literal sadness. Factual adverse news alone is
`neutral`, even if inconvenient or suspicious. Urgency or scam classification
alone does not determine sentiment.

"Your parcel is on hold. Pay now" is neutral: a factual hold plus a directive,
without alarm or threat. "Critical virus; your files are at risk" is sad.
"Reply with the code to stop the transfer immediately" is sad because it frames
an imminent harmful transfer. An expiring rebate notice can remain neutral;
"Congratulations" and "Good news" supply happy framing. For mixed tone, use
explicit framing of the main premise; if equally strong positive and negative
framing cannot be resolved, flag for review rather than pretending neutral means
unknown.

## Hook: the main premise, separate from action and industry

Choose the premise used to attract attention, not the requested action or sender's
sector. Use these overlap rules before falling back to the most prominent premise:

- Explicit taxes, government benefits/rebates, fines with legal enforcement,
  police or courts → government/legal, even when money is offered or owed.
  An ordinary municipal water bill remains invoice/debt; public-service industry
  alone does not make the hook government/legal.
- Job or investment opportunity → job/investment, even when it promises money.
- Nongovernment prize/gift/refund → prize/refund/gift, even when release requires
  a shipping fee. A real parcel status/address/redelivery premise → delivery/order.
- Compromised or blocked account, failed transaction or unauthorized activity →
  account/payment problem. Device infection, repair or subscription renewal →
  tech support/subscription. Asking for remote software does not convert an
  account-compromise premise into a device-support premise.
- Ordinary sale/subscription offer → sale/promotion. Routine records, receipts,
  appointments and informational schedules → none/other unless they explicitly
  establish another hook. A bank statement is not a debt; an appointment is not
  a relationship emergency.

If multiple independent premises remain equally prominent after these rules,
flag for review. Do not derive the hook solely from the industry or action.

## Paired review and corrections

All 352 values (44 messages × eight questions) were reviewed against this rubric.
The independent review agreed on these three scenario corrections, applied to
both translations (six values). The other 346 values are retained. Neither
message text nor translations changed.

| Pair ID | Question | v3 → v4 | Reason |
| --- | --- | --- | --- |
| sms_delivery_fee | sentiment | sad → neutral | Factual hold, no explicit alarm/threat. |
| fr_sms_lottery_voucher | urgent_action | true → false | Today modifies receiving the prize, not a payment cutoff. |
| fr_email_streaming_security | message_hook | tech_support_or_subscription → account_or_payment_problem | Compromised account is the premise; software installation is the action. |

Retained scenarios (all eight labels in both languages): email_bank_statement,
sms_retail_promotion, email_government_rebate, sms_job_onboarding_fee,
email_remote_support, sms_clinic_reminder, sms_bank_security_code,
email_parcel_tracking, email_utility_bill, sms_prize_claim, fr_sms_traffic_fine,
fr_email_hotel_offer, fr_sms_library_closure, fr_email_insurance_receipt,
fr_sms_pharmacy_password, fr_email_investment_wallet, fr_email_building_inspection,
fr_sms_transit_schedule, fr_email_recruiting_identity.

## Reproducibility and prompt follow-up

v1/v2/v3 datasets and existing result files are historical and unchanged. The CLI
still defaults to v3 so previous commands remain reproducible. Select the reviewed
labels explicitly with `--dataset datasets/base_eval_v4.jsonl`. v4 remains synthetic
development data, with the same 22 paired scenarios; keep pairs together in splits.
Existing published scores do not describe v4. No inference score is claimed here.

The English and French model-facing schemas are unchanged. Their urgency wording
still broadly includes deadlines, their primary-action wording omits the tie-break,
and their sentiment/hook descriptions omit some of these boundaries. The
separately versioned `schemas/rubric-v1.json` now encodes these boundaries
for ordinary English-schema evaluation. Compare it against the recorded baseline
without overwriting that baseline. The annotation contract determines gold labels independently of either schema's predictions.
