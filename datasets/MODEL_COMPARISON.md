# Encoder competitor comparison

Measured on 2026-09-26, using the same 11 examples, eight question definitions,
and expected labels in `base_eval_v1.jsonl`. This is a small synthetic development
set. The questions were tuned with Laya, so these are development results and
not an independent ranking of the models.

| Model | Individual label accuracy | All eight labels correct | Scam category correct |
|---|---:|---:|---:|
| laya | 79/88 (89.8%) | 5/11 | 9/11 |
| fastino/GLiNER2.5-Decide | 73/88 (83.0%) | 1/11 | 6/11 |
| Contrastive-LM/CLM-v0.1-8B | 37/88 (42.0%) | 0/11 | 7/11 |

Neither new competitor improved aggregate accuracy on this setup. GLiNER matched
Laya's requested-action accuracy (9/11) and improved multiple-action accuracy
(10/11 versus 9/11), but scam classification fell from 9/11 to 6/11.

GLiNER ran on CPU, with the official classification scorer and all eight heads in
one encoder pass. CLM used its released head and Qwen3-8B encoder on Apple MPS,
with local Transformers causal attention, last-token pooling and L2 normalization.
The reference schema and head implementation are vendored unchanged from upstream.
The architecture and pooling convention were checked against upstream code;
numerical parity with an actual vLLM encoder has not been measured.

Measured p50 per-message latency was 503 ms for the earlier Laya MPS baseline,
2840 ms for GLiNER CPU, and 2318 ms for CLM MPS. The new runs executed concurrently
and CLM caches candidate embeddings, so these timings are observational rather
than a controlled comparison of model serving speed. Loading and warmup were
excluded. No latency tuning was performed.

Full answers, expected values, probabilities, snapshot paths and package versions
are saved in the locally generated reports under `results/`. The merged report
is `results/encoder-comparison.json`; these artifacts are ignored by Git.

## Separate positive/negative consistency check

Verified with Laya on 2026-09-26 using `--eval --consistency`. Each example
gets a normal positive request and a separate negative request. Choice questions
negatively confirm the nominated best label; boolean questions invert the
original condition. This is candidate confirmation, rather than exhaustive
one-versus-rest classification. The second request infers its own answer.

| Measure | Result |
|---|---:|
| Baseline label accuracy | 79/88 (89.8%) |
| Accepted label accuracy | 42/45 (93.3%) |
| Accepted label coverage | 45/88 (51.1%) |
| Errors accepted | 3 |
| Errors rejected | 6 |
| Messages accepted on all eight questions | 0/11 |

Agreement improved aggregate accuracy slightly while discarding nearly half
the predictions. It still accepted one urgent-action error and two multiple-action
errors. Multiple-action accuracy on accepted predictions was only 3/5 (60%),
compared with 9/11 (81.8%) before filtering. No complete message passed the filter,
so accuracy on fully accepted messages is undefined. These development examples
do not establish that agreement improves accuracy on unseen messages.

The full report is saved locally in `results/laya-consistency.json`, including
both outputs, the per-example negative schemas, package versions, per-question
coverage and accuracy, and separate positive/negative/combined timings. The
Laya runtime warns that some checkpoint confidence values are uncalibrated;
these results use decisions at 0.5, not a calibration claim.

## Third-request experiment

Measured with Laya on 2026-09-26. The same 11 messages were evaluated with
positive and negative requests, followed by a third request containing only the
43 disputed questions across those messages. Fixed paraphrases retained the
original rubric and definitions, without showing previous outputs or announcing
conflict. All 11 messages needed a third request. Gold labels were used only for
scoring, never for prompt construction or selecting a resolution.

| Measure | Result |
|---|---:|
| Original accuracy on disputed labels | 37/43 (86.0%) |
| Third accuracy on disputed labels | 33/43 (76.7%) |
| Original errors fixed | 0 |
| Correct original answers changed to wrong | 4 |
| Baseline accuracy across all labels | 79/88 (89.8%) |
| Accuracy after replacing disputed answers | 75/88 (85.2%) |
| Labels accepted with two supporting decisions | 88/88 (100%) |
| Errors accepted with two supporting decisions | 13 |

All disputed choice answers repeated the original candidate. The four changed
answers were boolean `multiple_actions` predictions, on `sms_delivery_fee`,
`email_government_rebate`, `sms_bank_security_code`, and `sms_prize_claim`.
Each switched a correct false answer to an incorrect true answer and agreed
with the inverted negative decision. The two-support policy therefore accepted
all labels, with the same 85.2% accuracy as replacing all disputes. No disputes
remained unresolved in this particular run. Acceptance does not establish truth.

Median third-request latency was 347 ms when needed; median combined latency
was 1158 ms per message, versus 783 ms in the previous two-request run. Timings
are observational. The third request added work while worsening accuracy here.
This does not establish how other prompts or models behave on unseen messages.
The full local report is `results/laya-third-request.json`.
