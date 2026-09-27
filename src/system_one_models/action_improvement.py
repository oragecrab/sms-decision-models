"""Reviewed development/heldout action experiments with focused evidence."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
import statistics
import time
from datetime import datetime,timezone
from pathlib import Path
from importlib.metadata import version
from .action_decomposition import (
    GATE_RULE, PAIR_RULE, PAIR_CRITERIA, OPPOSITION, propose_spans,
    decisive_choice, confirm, checked_noul, predict_stage, summarize,
)
from .action_decomposition_variants import GATE_CRITERIA,selection
from .backends import LayaBackend
from .consistency import negative_questions
from .question_schemas import load_question_schema,schema_hash
from .questions import QUESTIONS
from .schema_sweep import model_identity

ROOT=Path(__file__).resolve().parents[2]
RULE=("Count independent operations explicitly requested of the recipient. "
      "Include invitations and optional opt-outs. Do not count factual information, "
      "available benefits, negated requests, sender actions, or alternatives as extra tasks. "
      "Navigation, reply transport, multiple fields and payment steps belong to one task. "
      "Two separate requests of the same category can be two tasks.")
FOCUSED_GATE={"request":dict(type="choice",instructions="Classify only target_span, using the complete source to interpret shared verbs, negation and conditions. "+GATE_RULE,criteria=GATE_CRITERIA)}
FOCUSED_PAIR={"relationship":dict(type="choice",instructions="Judge only target_1 and target_2 in the complete source. "+PAIR_RULE,criteria=PAIR_CRITERIA)}
DIRECT_CHOICE={"multiple_actions":dict(type="choice",instructions=RULE,
    criteria={"single_or_none":"Zero or one independent recipient operation is requested.",
              "multiple":"At least two independent recipient operations are requested together."})}
DIRECT_SHORT={"multiple_actions":dict(type="noul",instructions="Are at least two separate recipient operations explicitly requested together? "+RULE,
    criteria={"true":"Two or more independent tasks are requested together.",
              "false":"At most one independent task is requested."},labels={"true":"A","false":"B"})}

DIRECT_THREE={"multiple_actions":dict(type="choice",instructions=RULE+
    " Either/or alternatives count as one task. Calling support then installing a tool count as two tasks.",
    criteria={"zero_tasks":"No recipient task is explicitly requested.",
              "one_task":"One recipient task is requested, including its steps or alternatives.",
              "multiple_tasks":"At least two distinct recipient tasks are requested together."})}
COMPACT_GATE={"request":dict(type="choice",instructions=
    "Classify only target_span in the message. Use surrounding text for the actor, shared verb, "
    "condition and negation. Is the recipient explicitly asked or invited to perform its operation? "
    "Optional opt-outs count. Facts, sender activity, promised results, available features and "
    "negated operations are not requests. A shared directive may apply to target_span.",
    criteria={"requested":"The target expresses an operation explicitly requested of the recipient.",
              "not_requested":"The target does not express a requested recipient operation."})}
COMPACT_PAIR={"relationship":dict(type="choice",instructions=
    "Judge only target_1 and target_2 in the full message. Both must be requested recipient operations. "
    "Call then install are distinct tasks. Click to pay, reply with data, fields in one submission "
    "and buy/send a voucher for payment are parts of one task. Either/or choices are alternatives. "
    "Optional opt-outs count as requests.",
    criteria={"unsupported":"At least one passage does not express a requested recipient operation.",
              "one_task_or_alternative":"Both are requested, but are parts of one task or alternatives.",
              "independent_together":"Both are requested and constitute distinct recipient tasks."})}


def config_hash(config:dict)->str:
    """Hash the full policy, preserving choice order and target presentation."""
    return hashlib.sha256(json.dumps(config,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def configurations()->dict:
    balanced,_=load_question_schema(ROOT/"schemas/sweep-selected-balanced-v1.json")
    return dict(original=dict(kind="direct",questions={"multiple_actions":QUESTIONS["multiple_actions"]}),
                balanced=dict(kind="direct",questions={"multiple_actions":balanced["multiple_actions"]}),
                direct_choice=dict(kind="direct",questions=DIRECT_CHOICE),
                direct_explicit=dict(kind="direct",questions=DIRECT_SHORT),
                direct_three=dict(kind="direct",questions=DIRECT_THREE),
                target_field=dict(kind="pipeline",marked=False,gate=FOCUSED_GATE,pair=FOCUSED_PAIR),
                marked_target=dict(kind="pipeline",marked=True,gate=FOCUSED_GATE,pair=FOCUSED_PAIR),
                marked_compact=dict(kind="pipeline",marked=True,gate=COMPACT_GATE,pair=COMPACT_PAIR))


def focused_state(state:dict,spans:list[dict],*,marked:bool)->dict:
    result=dict(state)
    for i,span in enumerate(spans):
        if state[span["field"]][span["start"]:span["end"]]!=span["text"]:
            raise ValueError("Candidate offsets do not match source")
        result["target_span" if len(spans)==1 else f"target_{i+1}"]=span["text"]
    if marked:
        for field in {span["field"] for span in spans}:
            targets=[(i,span) for i,span in enumerate(spans) if span["field"]==field]
            for i,span in sorted(targets,key=lambda pair:pair[1]["start"],reverse=True):
                text=result[field]
                tag="TARGET" if len(spans)==1 else f"TARGET_{i+1}"
                result[field]=text[:span["start"]]+f"[[{tag}]]"+text[span["start"]:span["end"]]+f"[[/{tag}]]"+text[span["end"]:]
    return result


def choice_decision(answer:dict,criteria:dict)->str|None:
    probs=answer["probabilities"]
    if set(probs)!=set(criteria) or any(not math.isfinite(v) or not 0<=v<=1 for v in probs.values()) or abs(sum(probs.values())-1)>.005:
        raise ValueError("Invalid Choice distribution")
    if answer["choice"] not in probs or probs[answer["choice"]]!=max(probs.values()):
        raise ValueError("Choice conflicts with distribution")
    return None if sum(v==max(probs.values()) for v in probs.values())>1 else answer["choice"]


def positive(backend,state:dict,config:dict)->dict:
    started=time.perf_counter()
    if config["kind"]=="direct":
        answers,_=predict_stage(backend,state,config["questions"])
        spec=config["questions"]["multiple_actions"];answer=answers["multiple_actions"]
        if spec["type"]=="choice":
            value=choice_decision(answer,spec["criteria"])
            predicted=value in ("multiple","multiple_tasks");unresolved=value is None
            confidence=answer["probabilities"][answer["choice"]]
        else:
            p=checked_noul(answer);predicted=p>=.5;unresolved=p==.5;confidence=max(p,1-p)
        return dict(predicted=predicted,unresolved=unresolved,answers=answers,
                    answer_confidence=confidence,positive_ms=(time.perf_counter()-started)*1000)
    candidates=propose_spans(state);gates={};gate_states={};selected=[];gate_ties=[]
    for span in candidates:
        target=focused_state(state,[span],marked=config["marked"])
        answer,_=predict_stage(backend,target,config["gate"])
        gates[span["id"]]=answer["request"];gate_states[span["id"]]=target
        if selection(answer["request"]):selected.append(span)
        if len(set(answer["request"]["probabilities"].values()))==1:gate_ties.append(span["id"])
    pairs={};pair_states={};witnesses=[];unresolved=bool(gate_ties)
    for a,b in itertools.combinations(selected,2):
        key=f"{a['id']}__{b['id']}";target=focused_state(state,[a,b],marked=config["marked"])
        answer,_=predict_stage(backend,target,config["pair"])
        pairs[key]=answer["relationship"];pair_states[key]=target
        decision=choice_decision(answer["relationship"],config["pair"]["relationship"]["criteria"])
        unresolved |= decision in (None,"unclear")
        if decision=="independent_together":witnesses.append(key)
    return dict(predicted=bool(witnesses),unresolved=unresolved,candidates=candidates,
                gate_answers=gates,gate_states=gate_states,selected_candidate_ids=[c["id"] for c in selected],
                candidate_gate_ties=gate_ties,pair_answers=pairs,pair_states=pair_states,
                independent_pair_witnesses=witnesses,answer_confidence=None,
                positive_ms=(time.perf_counter()-started)*1000)


def summaries(rows:list[dict],names:list[str])->dict:
    return {name:dict(overall=summarize(rows,name),per_language={lang:summarize([r for r in rows if r["language"]==lang],name) for lang in ("en","fr")}) for name in names}


def load_cases(path:Path,split:str)->list[dict]:
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids=set();pairs={}
    for row in rows:
        if row["id"] in ids:raise ValueError("Duplicate example id")
        ids.add(row["id"])
        if type(row["expected"]) is not bool or row["language"] not in ("en","fr") or row["split"] not in ("development","heldout"):
            raise ValueError("Invalid action-case metadata")
        pairs.setdefault(row["pair_id"],[]).append(row)
    template_splits={}
    for row in rows:
        group=row.get("template_group",row["pair_id"])
        template_splits.setdefault(group,set()).add(row["split"])
    if any(len(splits)>1 for splits in template_splits.values()):
        raise ValueError("Related templates cannot cross development/heldout splits")
    for pair,members in pairs.items():
        if len(members)!=2 or {r["language"] for r in members}!={"en","fr"} or len({r["split"] for r in members})!=1 or len({r["expected"] for r in members})!=1:
            raise ValueError("Translations must share labels and split")
    selected=[r for r in rows if r["split"]==split]
    if not selected:raise ValueError("Empty requested split")
    return selected


def run(*,dataset:Path,output:Path,split:str,selection_path:Path|None=None,device:str="mps",comparators_only:bool=False)->dict:
    os.environ["LAYA_MPS_AMP_MIN_ROWS"]="10000"
    import torch
    torch.set_num_threads(2)
    configs=configurations()
    if split=="heldout":
        if selection_path is None:raise ValueError("Freeze a development selection before heldout inference")
        frozen=json.loads(selection_path.read_text())
        if frozen["dataset_sha256"]!=hashlib.sha256(dataset.read_bytes()).hexdigest():raise ValueError("Selection dataset mismatch")
        winner=frozen["selected"]
        if not comparators_only and frozen["config_sha256"]!=config_hash(configs[winner]):raise ValueError("Selected configuration changed")
        if any(config_hash(configs[name])!=digest for name,digest in frozen["comparator_hashes"].items()):
            raise ValueError("Comparator configuration changed")
        names=["original","balanced"] if comparators_only else list(dict.fromkeys(["original","balanced",winner]))
    else:
        if comparators_only:raise ValueError("Comparator-only assessment needs a frozen heldout selection")
        names=list(configs)
    examples=load_cases(dataset,split)
    backend=LayaBackend(device=device)
    warmed=set()
    for example in examples:
        if example["language"] not in warmed:
            # Warmup outputs are discarded; heldout selection was already frozen.
            for name in names:positive(backend,example["state"],configs[name])
            predict_stage(backend,example["state"],OPPOSITION)
            warmed.add(example["language"])
    if split=="heldout":
        if model_identity(backend)!=frozen["development_model"]["checkpoints"] or frozen["development_model"]["torch_threads"]!=2:
            raise ValueError("Heldout model/runtime differs from development")
        packages={name:version(name) for name in ("laya","torch","transformers","numpy")}
        if packages!=frozen["development_packages"]:raise ValueError("Heldout packages differ from development")
    report=dict(schema_version=1,version="action-improvement-v1",created_at=datetime.now(timezone.utc).isoformat(),
                split=split,dataset=str(dataset.resolve()),dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
                harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                configs={name:configs[name] for name in names},config_hashes={name:config_hash(configs[name]) for name in names},
                frozen_selection=json.loads(selection_path.read_text()) if selection_path else None,
                model=dict(backend.metadata,checkpoints=model_identity(backend),torch_threads=2),
                packages={name:version(name) for name in ("laya","torch","transformers","numpy")},predictions=[])
    output.parent.mkdir(parents=True,exist_ok=True)
    for i,example in enumerate(examples):
        policies={}
        # Alternate policy order to reduce systematic timing bias.
        ordered=names if i%2==0 else list(reversed(names))
        common_negative,common_ms=predict_stage(backend,example["state"],OPPOSITION)
        for name in ordered:
            config=configs[name];result=positive(backend,example["state"],config)
            if config["kind"]=="direct" and config["questions"]["multiple_actions"]["type"]=="noul":
                negative_schema=negative_questions(result["answers"],questions=config["questions"])
                negative,negative_ms=predict_stage(backend,example["state"],negative_schema)
            else:negative,negative_ms,negative_schema=common_negative,common_ms,OPPOSITION
            result.update(negative=negative,negative_questions=negative_schema,negative_ms=negative_ms,
                          accepted=not result["unresolved"] and confirm(result["predicted"],negative["multiple_actions"]))
            policies[name]=result
        row=dict(id=example["id"],pair_id=example["pair_id"],language=example["language"],expected=example["expected"],
                 policies=policies,routing=getattr(backend,"last_routing",None))
        report["predictions"].append(row)
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        print(f"{split} {i+1}/{len(examples)} {example['id']}",flush=True)
    report["summary"]=summaries(report["predictions"],names)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report


def freeze_selection(report_path:Path,output:Path)->dict:
    report=json.loads(report_path.read_text())
    if report["split"]!="development":raise ValueError("Selection requires development data")
    dataset=Path(report["dataset"])
    if hashlib.sha256(dataset.read_bytes()).hexdigest()!=report["dataset_sha256"]:
        raise ValueError("Development dataset changed")
    expected_rows=load_cases(dataset,"development")
    actual=report["predictions"]
    expected_by_id={r["id"]:r for r in expected_rows}
    if len(actual)!=len(expected_rows) or len({r["id"] for r in actual})!=len(actual) or set(expected_by_id)!={r["id"] for r in actual}:
        raise ValueError("Incomplete or duplicate development predictions")
    for row in actual:
        if any(row[key]!=expected_by_id[row["id"]][key] for key in ("language","pair_id","expected")):
            raise ValueError("Development labels or metadata changed")
        if set(row["policies"])!=set(report["configs"]):raise ValueError("Incomplete policy predictions")
    for name,config in report["configs"].items():
        if config_hash(config)!=report["config_hashes"][name]:raise ValueError("Configuration hash mismatch")
    report["summary"]=summaries(actual,list(report["configs"]))
    def rank(name):
        summary=report["summary"][name];langs=summary["per_language"]
        # Balance classes within languages; do not let a majority class hide missed tasks.
        scores=[]
        for language in ("en","fr"):
            rows=[r for r in report["predictions"] if r["language"]==language]
            pos=[r for r in rows if r["expected"]];neg=[r for r in rows if not r["expected"]]
            if not pos or not neg:raise ValueError("Development needs both classes in each language")
            sensitivity=sum(r["policies"][name]["predicted"] for r in pos)/len(pos)
            specificity=sum(not r["policies"][name]["predicted"] for r in neg)/len(neg)
            scores.append((sensitivity+specificity)/2)
        return (min(scores),sum(scores),-summary["overall"]["false_negatives"],
                -summary["overall"]["positive_latency_ms"]["p50"])
    names=list(report["configs"]);winner=max(names,key=rank)
    result=dict(version="action-selection-v1",created_at=datetime.now(timezone.utc).isoformat(),
                dataset_sha256=report["dataset_sha256"],development_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                rule="Maximize weaker-language balanced accuracy, then both-language sum, then fewer missed tasks, then lower p50 latency",
                selected=winner,config=report["configs"][winner],config_sha256=report["config_hashes"][winner],
                development_ranks={name:rank(name) for name in names},heldout_used=False,
                comparator_hashes={name:report["config_hashes"][name] for name in ("original","balanced")},
                development_model=report["model"],development_packages=report["packages"])
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--split",choices=("development","heldout"))
    p.add_argument("--selection",type=Path)
    p.add_argument("--freeze",type=Path)
    p.add_argument("--comparators-only",action="store_true")
    args=p.parse_args()
    if args.freeze:freeze_selection(args.freeze,args.output)
    else:run(dataset=args.dataset,output=args.output,split=args.split,selection_path=args.selection,comparators_only=args.comparators_only)

if __name__=="__main__":main()
