# Balanced English/French development dataset

`base_eval_v2.jsonl` contains 22 synthetic messages: 11 English and 11 French,
with eight expected answers each (176 labels). The original v1 file is preserved.
The original English states and expected answers are unchanged; v2 adds a
`language: "en"` annotation to those rows. French rows use `language: "fr"`.
These annotations are evaluation metadata, not inputs or language hints to the
model. The shared question definitions and output label keys remain in English.

Each language has six suspected scams, four legitimate messages, and one
marketing message. Equal language weighting is a development choice; it does
not assume the same proportions in actual traffic. Since topics differ, a
French/English score difference also reflects content differences and is not a
controlled translation comparison. This dataset remains synthetic development
data, not a representative real-world holdout.

The French messages were authored and labeled against the current question
rubric before examining model predictions. They are distinct scenarios, not
translations of the English messages. The labels have not received independent
human review. Existing English ambiguity notes remain in `LABEL_REVIEW.md`.

| French ID | Scenario and annotation rationale |
|---|---|
| `fr_email_streaming_security` | Unsolicited remote-access installation to avoid threatened loss, plus a separate call to dispute purchases. Installation is primary; the two requested tasks count separately. Security/device support is the hook. |
| `fr_sms_traffic_fine` | Claimed traffic fine payable with a gift card under threat of seizure. This is a suspected government/legal impersonation scam. Buying and transmitting the voucher are steps of one payment, and the payment code is not a personal credential. |
| `fr_email_hotel_offer` | Cheerful hotel-package advertising with an explicit booking invitation and no expiry. Booking maps to purchase; no urgency or independent second action. |
| `fr_sms_library_closure` | Informational municipal-library closure. Government/public services, with no requested action or urgency. |
| `fr_email_insurance_receipt` | Completed home-insurance payment receipt. Insurance maps to finance; a completed receipt is not an outstanding invoice/debt. No action is requested. |
| `fr_sms_pharmacy_password` | Threat of losing prescription records unless a password is disclosed. Healthcare service, account-problem hook, sensitive disclosure and explicit urgency. Replying is the disclosure mechanism. |
| `fr_email_investment_wallet` | Celebratory guaranteed investment gain requiring a wallet recovery secret by tonight. The recovery phrase is treated as a credential; investment is the hook and finance the industry. |
| `fr_sms_lottery_voucher` | Console prize requiring a prepaid voucher payment today. Prize-payment scam; payment is the primary action. Voucher code is payment, not identity or login data. |
| `fr_email_building_inspection` | Routine property-maintenance scheduling. Other industry; one reply selecting between alternative visit times. Explicitly no response deadline. |
| `fr_sms_transit_schedule` | Informational public-transit timetable. Travel/transport service rather than government authority; no requested action, despite several negative statements. |
| `fr_email_recruiting_identity` | Job without an interview requesting passport and bank-login credentials, plus a separate scheduling call. Disclosure is primary; several data fields form one disclosure task and the scheduling call is another. No deadline or celebratory wording. |

Evaluation reports include aggregate and per-language accuracy. Consistency
reports additionally include per-language acceptance coverage/accuracy, and
third-request experiments include per-language resolution results. Laya's actual
routing decision is recorded per inference; routing uses the message state,
not the dataset's language annotation. A routing regression check requires every
French example to use the multilingual checkpoint and every English example to
use the English checkpoint.

Run the balanced evaluation with:

```sh
uv run system-one-models --eval --model laya --output results/laya-bilingual-v2.json
```

To reproduce the earlier English-only experiments, specify:

```sh
uv run system-one-models --eval --model laya --dataset datasets/base_eval_v1.jsonl
```

## Initial Laya measurement

Measured on 2026-09-26 with the existing English question schema and automatic
language routing; no prompts or labels were tuned after seeing these answers.

| Language | Messages | Correct labels | All eight labels correct |
|---|---:|---:|---:|
| English | 11 | 79/88 (89.8%) | 5/11 |
| French | 11 | 51/88 (58.0%) | 0/11 |
| Combined | 22 | 130/176 (73.9%) | 5/22 |

All 11 French messages used the multilingual checkpoint, and all 11 English
messages used the English checkpoint. On French, requested action, multiple
actions, and sensitive disclosure each scored 5/11; industry scored 6/11. This
exposes weaknesses to investigate rather than establishing a language-only
effect: the languages have different content and the English prompts were
previously tuned on the English examples. French question wording is a possible
future experiment, not part of this baseline.

The full initial report is `results/laya-bilingual-v2.json`. Reported latency
warms only the first example, so the first French prediction includes loading
the multilingual checkpoint; these timings are not a controlled serving-speed
comparison between languages.

The existing negative/third-request experiment also completed on v2, saved in
`results/laya-bilingual-third-v2.json`. All positive, negative, and third requests
routed to the expected checkpoint for their language, and baseline answers
reproduced the initial run's per-language accuracy. The negative filter accepted
45 English labels at 93.3% accuracy and 34 French labels at 58.8% accuracy.
Replacing disputes with third answers gave 85.2% English and 55.7% French accuracy.
No original errors were fixed; six correct answers were changed to wrong across
the two languages. These policies remain experiments rather than deployment
recommendations.
