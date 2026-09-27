"""Controlled categorical-gate and split-relationship action experiments."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version
from .action_decomposition import (
    GATE_RULE, OPPOSITION, checked_noul, confirm, selected_pair_result, pair_questions,
    predict_stage, propose_spans, summarize,
)
from .backends import LayaBackend
from .benchmark import load_dataset
from .schema_sweep import model_identity

VERSION="action-decomposition-variants-v1"
GATE_CRITERIA={
    "not_requested": "Information, absence of action, availability, sender activity or promised reward.",
    "requested": "A concrete recipient operation explicitly requested or invited, including an opt-out.",
}
JOINT_RULE=("Are BOTH quoted recipient operations actually requested by this message? "
            "Optional opt-outs count. Do not count factual statements, promised benefits, "
            "mere availability, negated requests, or mutually exclusive alternatives.")
INDEPENDENCE_RULE=("Would the two quoted recipient operations be distinct tasks, "
                   "rather than steps, navigation, reply transport, fields or payment methods "
                   "of one task? Call then install are distinct. Click to pay, reply with "
                   "a code, passport plus bank details in one submission, and buy/send "
                   "a voucher for payment are parts of one task.")


def choice_gates(candidates: list[dict]) -> dict:
    return {c["id"]:dict(type="choice",instructions="Classify ONLY the quoted source span. " + GATE_RULE +
                        " Source span: " + json.dumps(c["text"],ensure_ascii=False),criteria=GATE_CRITERIA)
            for c in candidates}


def selection(answer: dict) -> bool:
    probabilities=answer["probabilities"]
    if set(probabilities) != set(GATE_CRITERIA):
        raise ValueError("Incomplete gate distribution")
    values=list(probabilities.values())
    if any(not math.isfinite(p) or not 0<=p<=1 for p in values) or abs(sum(values)-1)>.005:
        raise ValueError("Invalid gate distribution")
    if answer["choice"] not in probabilities:
        raise ValueError("Invalid gate choice")
    if probabilities[answer["choice"]] != max(values):
        raise ValueError("Gate choice conflicts with probabilities")
    return answer["choice"]=="requested" and probabilities["requested"]>probabilities["not_requested"]


def factor_questions(candidates: list[dict]) -> dict:
    questions={}
    for a,b in itertools.combinations(candidates,2):
        suffix=" First span: "+json.dumps(a["text"],ensure_ascii=False)+" Second span: "+json.dumps(b["text"],ensure_ascii=False)
        pair=f"{a['id']}__{b['id']}"
        for factor,rule,yes,no in (
            ("joint",JOINT_RULE,"Both recipient operations are explicitly requested together.","Both are not requested together."),
            ("independent",INDEPENDENCE_RULE,"They are distinct recipient tasks.","They are parts of one task.")):
            questions[f"{pair}__{factor}"]=dict(type="noul",instructions=rule+suffix,
                criteria={"true":yes,"false":no},labels={"true":"A","false":"B"})
    return questions


def factor_result(answers: dict) -> dict:
    pairs={key.rsplit("__",1)[0] for key in answers}
    if set(answers)!={f"{pair}__{factor}" for pair in pairs for factor in ("joint","independent")}:
        raise ValueError("Missing pair factor")
    witnesses=[]
    ties=[]
    for pair in sorted(pairs):
        values=[checked_noul(answers[f"{pair}__{factor}"]) for factor in ("joint","independent")]
        if all(p>.5 for p in values): witnesses.append(pair)
        if .5 in values: ties.append(pair)
    return dict(multiple_actions=bool(witnesses),independent_pair_witnesses=witnesses,
                independent_pair_count=len(witnesses),task_count_lower_bound=2 if witnesses else None,
                unresolved=bool(ties))


def run(*,source:Path,dataset:Path,output:Path,device:str="mps",threads:int=2)->dict:
    os.environ["LAYA_MPS_AMP_MIN_ROWS"]="10000"
    import torch
    torch.set_num_threads(threads)
    raw=source.read_bytes()
    original=json.loads(raw)
    if hashlib.sha256(dataset.read_bytes()).hexdigest()!=original["dataset_sha256"]:
        raise ValueError("Source and dataset differ")
    examples=load_dataset(dataset)
    saved={r["id"]:r for r in original["predictions"]}
    if set(saved)!={r["id"] for r in examples}:
        raise ValueError("Source must contain the complete dataset")
    backend=LayaBackend(device=device)
    warmed=set()
    for example in examples:
        if example.get("language") not in warmed:
            candidates=propose_spans(example["state"])
            predict_stage(backend,example["state"],choice_gates(candidates))
            # Warm both head types regardless of the selected gate result.
            predict_stage(backend,example["state"],factor_questions(candidates[:2]))
            warmed.add(example.get("language"))
    checkpoints=model_identity(backend)
    if checkpoints!=original["model"]["checkpoints"] or original["model"]["torch_threads"]!=threads:
        raise ValueError("Cached comparator runtime does not match")
    report=dict(schema_version=1,version=VERSION,created_at=datetime.now(timezone.utc).isoformat(),
        source=str(source.resolve()),source_sha256=hashlib.sha256(raw).hexdigest(),
        dataset_sha256=original["dataset_sha256"],harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        gate_criteria=GATE_CRITERIA,joint_rule=JOINT_RULE,independence_rule=INDEPENDENCE_RULE,
        model=dict(backend.metadata,checkpoints=checkpoints,torch_threads=threads),
        packages={name:version(name) for name in ("laya","torch","transformers","numpy")},predictions=[])
    if report["packages"]!=original["packages"]: raise ValueError("Cached comparator packages differ")
    for i,example in enumerate(examples):
        base=saved[example["id"]]
        state=example["state"]
        candidates=propose_spans(state)
        if candidates!=base["candidates"] or state!=base["state"] or example["expected"]["multiple_actions"]!=base["expected"]:
            raise ValueError("Candidate/source evidence changed")
        gate_schema=choice_gates(candidates)
        gates,gate_ms=predict_stage(backend,state,gate_schema)
        selected=[c for c in candidates if selection(gates[c["id"]])]
        relations_schema=pair_questions(selected)
        # Reuse byte-identical pair schemas from the fixed-precision source run.
        cached={k:base["pair_answers"][k] for k in relations_schema if k in base["pair_answers"] and relations_schema[k]==base["pair_questions"][k]}
        missing={k:v for k,v in relations_schema.items() if k not in cached}
        added,relation_ms=predict_stage(backend,state,missing)
        relations={**cached,**added}
        gate_ties=[c["id"] for c in candidates if len(set(gates[c["id"]]["probabilities"].values()))==1]
        relation_result=selected_pair_result(selected,relations,gate_ties=gate_ties)
        factor_schema=factor_questions(selected)
        factors,factor_ms=predict_stage(backend,state,factor_schema)
        result=factor_result(factors)
        result["candidate_gate_ties"]=gate_ties
        result["unresolved"]=result["unresolved"] or bool(gate_ties)
        policies={"baseline":base["policies"]["baseline"],"balanced":base["policies"]["balanced"],"v1_decomposed":base["policies"]["decomposed"]}
        negative=base["policies"]["decomposed"]["negative"]
        if base["opposition_questions"]!=OPPOSITION: raise ValueError("Cached opposition schema differs")
        for name,decision,elapsed in (
            ("choice_gate",relation_result,gate_ms+relation_ms),
            ("choice_gate_split_pairs",result,gate_ms+factor_ms)):
            policies[name]=dict(predicted=decision["multiple_actions"],
                accepted=confirm(decision["multiple_actions"],negative["multiple_actions"]),
                negative=negative,negative_ms=base["policies"]["decomposed"]["negative_ms"],positive_ms=elapsed,
                timing_excludes_cached_relations=(name=="choice_gate" and bool(cached)))
        row=dict(id=example["id"],language=example["language"],expected=base["expected"],candidates=candidates,
            gate_questions=gate_schema,gate_answers=gates,selected_candidate_ids=[c["id"] for c in selected],
            gate_ms=gate_ms,pair_questions=relations_schema,pair_answers=relations,cached_pair_ids=list(cached),
            pair_questions_added_ms=relation_ms,choice_pair_result=relation_result,
            factor_questions=factor_schema,factor_answers=factors,factor_ms=factor_ms,
            split_pair_result=result,opposition_questions=OPPOSITION,policies=policies)
        report["predictions"].append(row)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        print(f"{i+1}/{len(examples)} {example['id']}: {len(selected)} selected; choice={relation_result['multiple_actions']} factors={result['multiple_actions']}",flush=True)
    rows=report["predictions"]
    report["summary"]={policy:dict(overall=summarize(rows,policy),per_language={language:summarize([r for r in rows if r["language"]==language],policy) for language in ("en","fr")}) for policy in policies}
    report["timing_note"]="choice_gate reuses some relationship inference: its positive_ms is incremental work, not comparable end-to-end latency. choice_gate_split_pairs timing includes fresh categorical gates and all fresh factors. Comparator/opposition timings are inherited source observations."
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="mps")
    args=parser.parse_args()
    run(source=args.source,dataset=args.dataset,output=args.output,device=args.device)

if __name__=="__main__":main()
