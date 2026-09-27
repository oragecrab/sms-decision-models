"""Experimental source-span request extraction and pairwise action decomposition."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
import re
import statistics
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from .backends import LayaBackend
from .benchmark import load_dataset
from .consistency import negative_questions
from .question_schemas import load_question_schema, schema_hash
from .questions import QUESTIONS

VERSION = "action-decomposition-v1"
# General English/French separators propose spans, not semantic requests.
SEPARATOR = re.compile(r"[.!?;]\s+|\b(?:and|or|then|et|ou|puis)\b", re.IGNORECASE)
GATE_RULE = ("Does the quoted source span explicitly request a concrete recipient operation? "
             "Use the full message to interpret shared verbs and conditions. Count directives, "
             "invitations and optional opt-outs. Exclude mere availability, factual statements, "
             "negated requests, promised rewards and sender operations.")
PAIR_RULE = ("Judge the relationship of the two quoted spans in the full message. "
             "Call then install are separate tasks. Click to pay, reply with data, "
             "multiple fields, and buy/send a voucher to pay are one task. "
             "Optional opt-outs count; advertising alone is not a purchase request.")
PAIR_CRITERIA = {
    "independent_together": "Two distinct recipient operations are both requested.",
    "same_task_steps": "Steps, transport, fields or methods of the same recipient task.",
    "alternatives": "Either operation may be done; both are not requested together.",
    "unsupported": "At least one operation is not actually requested or applicable.",
    "unclear": "The relationship cannot be resolved from the message.",
}
OPPOSITION = {"multiple_actions": {
    "type": "noul",
    "instructions": "Does the message request at most ONE independent recipient task? "
                    "Count explicit invitations and optional opt-outs. Ignore alternatives, "
                    "navigation, reply transport, fields and payment steps. Call then install "
                    "are two tasks; buying/sending a voucher for payment is one.",
    "criteria": {"true": "At most one independent task is requested.",
                 "false": "At least two independent tasks are requested together."},
    "labels": {"true": "A", "false": "B"}}}


def propose_spans(state: dict) -> list[dict]:
    candidates = []
    for field in ("subject", "body"):
        text = state.get(field, "")
        start = 0
        ends = [(m.start(), m.end()) for m in SEPARATOR.finditer(text)] + [(len(text), len(text))]
        for end, next_start in ends:
            left, right = start, end
            while left < right and text[left] in " \t\r\n,:;.!?":
                left += 1
            while right > left and text[right-1] in " \t\r\n,:;.!?":
                right -= 1
            if left < right:
                candidates.append(dict(id=f"c{len(candidates)}", field=field,
                                       start=left, end=right, text=text[left:right]))
            start = next_start
    return candidates


def gate_questions(candidates: list[dict]) -> dict:
    return {c["id"]: {"type": "noul", "instructions": GATE_RULE + " Source span: " + json.dumps(c["text"], ensure_ascii=False),
                       "criteria": {"true": "This span requests a recipient operation.",
                                    "false": "This span does not request a recipient operation."},
                       "labels": {"true": "A", "false": "B"}} for c in candidates}


def pair_questions(candidates: list[dict]) -> dict:
    return {f"{a['id']}__{b['id']}": {"type": "choice", "instructions": PAIR_RULE +
            " First span: " + json.dumps(a["text"], ensure_ascii=False) +
            " Second span: " + json.dumps(b["text"], ensure_ascii=False),
            "criteria": PAIR_CRITERIA} for a,b in itertools.combinations(candidates, 2)}


def decisive_choice(answer: dict) -> str | None:
    probabilities = answer["probabilities"]
    if set(probabilities) != set(PAIR_CRITERIA):
        raise ValueError("Incomplete relationship distribution")
    values = list(probabilities.values())
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values) or abs(sum(values)-1) > .005:
        raise ValueError("Invalid relationship distribution")
    if answer["choice"] not in probabilities:
        raise ValueError("Invalid relationship choice")
    if probabilities[answer["choice"]] != max(values):
        raise ValueError("Relationship choice conflicts with probabilities")
    return None if sum(p == max(values) for p in values) > 1 else answer["choice"]


def decompose(candidates: list[dict], gates: dict, pairs: dict) -> dict:
    if set(gates) != {c["id"] for c in candidates}:
        raise ValueError("Incomplete candidate judgments")
    selected = []
    gate_ties = []
    for c in candidates:
        probability = gates[c["id"]]["noul"]
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid candidate probability")
        if probability > .5:
            selected.append(c)
        elif probability == .5:
            gate_ties.append(c["id"])
    return selected_pair_result(selected, pairs, gate_ties=gate_ties)


def selected_pair_result(selected: list[dict], pairs: dict, *, gate_ties: list[str] | None = None) -> dict:
    gate_ties = [] if gate_ties is None else gate_ties
    expected_pairs = set(pair_questions(selected))
    if set(pairs) != expected_pairs:
        raise ValueError("Incomplete selected-pair judgments")
    relationships = {key: decisive_choice(value) for key,value in pairs.items()}
    witnesses = [key for key,value in relationships.items() if value == "independent_together"]
    review = bool(gate_ties) or any(value in (None, "unclear") for value in relationships.values())
    return dict(selected_candidate_ids=[c["id"] for c in selected], candidate_gate_ties=gate_ties,
                relationships=relationships, independent_pair_witnesses=witnesses,
                independent_pair_count=len(witnesses), multiple_actions=bool(witnesses),
                task_count_lower_bound=2 if witnesses else None,
                unresolved=review)


def checked_noul(answer: dict) -> float:
    p = float(answer["noul"])
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Invalid Boolean probability")
    return p


def confirm(predicted: bool, negative: dict) -> bool:
    p = checked_noul(negative)
    return p != .5 and predicted == (p < .5)


def predict_stage(backend, state: dict, questions: dict, batch_size: int = 16) -> tuple[dict,float]:
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    started = time.perf_counter()
    result = {}
    items = list(questions.items())
    for start in range(0,len(items),batch_size):
        result.update(backend.predict(state, questions=dict(items[start:start+batch_size])))
    return result, (time.perf_counter()-started)*1000


def evaluate_message(backend, example: dict, comparators: dict) -> dict:
    state = example["state"]
    candidates = propose_spans(state)
    gate_schema = gate_questions(candidates)
    gates, gate_ms = predict_stage(backend,state,gate_schema)
    # Select using only source and model outputs, never expected annotations.
    selected = [c for c in candidates if checked_noul(gates[c["id"]]) > .5]
    pair_schema = pair_questions(selected)
    pairs, pair_ms = predict_stage(backend,state,pair_schema)
    result = decompose(candidates,gates,pairs)
    opposite, opposite_ms = predict_stage(backend,state,OPPOSITION)
    policies = {"decomposed": dict(predicted=result["multiple_actions"],
                accepted=confirm(result["multiple_actions"],opposite["multiple_actions"]),
                negative=opposite, positive_ms=gate_ms+pair_ms, negative_ms=opposite_ms)}
    for name,schema in comparators.items():
        positive,positive_ms = predict_stage(backend,state,schema)
        negative_schema = negative_questions(positive,questions=schema)
        negative,negative_ms = predict_stage(backend,state,negative_schema)
        p = checked_noul(positive["multiple_actions"])
        policies[name] = dict(predicted=p >= .5, accepted=p != .5 and confirm(p >= .5,negative["multiple_actions"]),
                             positive=positive,negative=negative,negative_questions=negative_schema,
                             positive_ms=positive_ms,negative_ms=negative_ms)
    return dict(id=example["id"],pair_id=example.get("pair_id"),language=example.get("language","unspecified"),
                state=state,expected=example["expected"]["multiple_actions"],candidates=candidates,
                gate_questions=gate_schema,gate_answers=gates,gate_ms=gate_ms,
                pair_questions=pair_schema,pair_answers=pairs,pair_ms=pair_ms,
                decomposition=result,opposition_questions=OPPOSITION,policies=policies,
                routing=getattr(backend,"last_routing",None))


def summarize(rows: list[dict], policy: str) -> dict:
    accepted = [r for r in rows if r["policies"][policy]["accepted"]]
    correct = sum(r["policies"][policy]["predicted"] == r["expected"] for r in rows)
    accepted_errors = sum(r["policies"][policy]["predicted"] != r["expected"] for r in accepted)
    times = sorted(r["policies"][policy]["positive_ms"] for r in rows)
    return dict(total=len(rows),correct=correct,accuracy=correct/len(rows),
                false_positives=sum(r["policies"][policy]["predicted"] and not r["expected"] for r in rows),
                false_negatives=sum(not r["policies"][policy]["predicted"] and r["expected"] for r in rows),
                agreed=len(accepted),coverage=len(accepted)/len(rows),accepted_errors=accepted_errors,
                rejected_errors=len(rows)-correct-accepted_errors,
                selective_accuracy=(len(accepted)-accepted_errors)/len(accepted) if accepted else None,
                positive_latency_ms=dict(p50=statistics.median(times),p95=times[math.ceil(.95*len(times))-1]))


def run(*,dataset: Path,output: Path,device: str="mps",threads: int=2) -> dict:
    # Fix precision for variable numbers of candidate/relationship question rows.
    os.environ["LAYA_MPS_AMP_MIN_ROWS"] = "10000"
    import torch
    torch.set_num_threads(threads)
    examples = load_dataset(dataset)
    root = Path(__file__).resolve().parents[2]
    balanced,_ = load_question_schema(root/"schemas/sweep-selected-balanced-v1.json")
    comparators = {"baseline": {"multiple_actions": QUESTIONS["multiple_actions"]},
                   "balanced": {"multiple_actions": balanced["multiple_actions"]}}
    backend = LayaBackend(device=device)
    warmed=set()
    for example in examples:
        if example.get("language") not in warmed:
            evaluate_message(backend,example,comparators)
            warmed.add(example.get("language"))
    from .schema_sweep import model_identity
    report = dict(schema_version=1,version=VERSION,created_at=datetime.now(timezone.utc).isoformat(),
                  dataset=str(dataset.resolve()),dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  proposal_pattern=SEPARATOR.pattern,comparators=comparators,
                  comparator_hashes={k:schema_hash(v) for k,v in comparators.items()},
                  gate_rule=GATE_RULE,pair_rule=PAIR_RULE,pair_criteria=PAIR_CRITERIA,
                  packages={name:version(name) for name in ("laya","torch","transformers","numpy")},
                  model=dict(backend.metadata,checkpoints=model_identity(backend),torch_threads=threads),
                  predictions=[])
    output.parent.mkdir(parents=True,exist_ok=True)
    for i,example in enumerate(examples):
        row=evaluate_message(backend,example,comparators)
        report["predictions"].append(row)
        print(f"{i+1}/{len(examples)} {example['id']}: {len(row['candidates'])} spans, {len(row['decomposition']['selected_candidate_ids'])} selected, {row['decomposition']['independent_pair_count']} independent pairs",flush=True)
        # Preserve scored progress, but never silently resume incompatible outputs.
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    rows=report["predictions"]
    report["summary"] = {policy:dict(overall=summarize(rows,policy),
        per_language={lang:summarize([r for r in rows if r["language"]==lang],policy) for lang in sorted({r["language"] for r in rows})})
        for policy in ("baseline","balanced","decomposed")}
    report["changes"] = {policy:dict(fixed=sum(r["policies"][policy]["predicted"] != r["expected"] and r["policies"]["decomposed"]["predicted"] == r["expected"] for r in rows),
        broken=sum(r["policies"][policy]["predicted"] == r["expected"] and r["policies"]["decomposed"]["predicted"] != r["expected"] for r in rows)) for policy in comparators}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="mps",choices=("mps","cpu","auto"))
    parser.add_argument("--threads",type=int,default=2)
    args=parser.parse_args()
    run(dataset=args.dataset,output=args.output,device=args.device,threads=args.threads)

if __name__ == "__main__":
    main()
