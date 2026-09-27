"""Freeze an action policy using development reports only, including execution grouping."""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
from .action_improvement import config_hash,load_cases,summaries


def rank(rows:list[dict],name:str)->tuple:
    scores=[];missed=0
    for language in ("en","fr"):
        group=[r for r in rows if r["language"]==language]
        pos=[r for r in group if r["expected"]];neg=[r for r in group if not r["expected"]]
        if not pos or not neg:raise ValueError("Both classes required in each language")
        recall=sum(r["policies"][name]["predicted"] for r in pos)/len(pos)
        specificity=sum(not r["policies"][name]["predicted"] for r in neg)/len(neg)
        scores.append((recall+specificity)/2)
        missed+=sum(not r["policies"][name]["predicted"] for r in pos)
    times=sorted(r["policies"][name]["positive_ms"] for r in rows)
    import statistics
    return min(scores),sum(scores),-missed,-statistics.median(times)


def promotion(rows:list[dict],candidate:str,reference:str="balanced")->dict:
    checks={}
    for language in ("en","fr"):
        group=[r for r in rows if r["language"]==language]
        pos=[r for r in group if r["expected"]];neg=[r for r in group if not r["expected"]]
        def rates(name):
            recall=sum(r["policies"][name]["predicted"] for r in pos)/len(pos)
            specificity=sum(not r["policies"][name]["predicted"] for r in neg)/len(neg)
            return dict(recall=recall,specificity=specificity,balanced_accuracy=(recall+specificity)/2)
        selected=rates(candidate);base=rates(reference)
        checks[language]=dict(candidate=selected,reference=base,
            balanced_accuracy_preserved=selected["balanced_accuracy"]>=base["balanced_accuracy"],
            positive_recall_preserved=selected["recall"]>=base["recall"])
    improved=min(v["candidate"]["balanced_accuracy"] for v in checks.values())>min(v["reference"]["balanced_accuracy"] for v in checks.values())
    return dict(per_language=checks,weaker_language_improved=improved,
                passes=improved and all(v["balanced_accuracy_preserved"] and v["positive_recall_preserved"] for v in checks.values()))


def freeze(paths:list[Path],*,selection_output:Path,policy_output:Path)->dict:
    reports=[json.loads(path.read_text()) for path in paths]
    first=reports[0];dataset=Path(first["dataset"])
    digest=hashlib.sha256(dataset.read_bytes()).hexdigest()
    expected=load_cases(dataset,"development");ids={r["id"] for r in expected};expected_by_id={r["id"]:r for r in expected}
    merged={r["id"]:dict(id=r["id"],pair_id=r["pair_id"],language=r["language"],expected=r["expected"],policies={}) for r in expected}
    configs={};executions={};source_by_policy={};runtime_by_policy={}
    for path,report in zip(paths,reports):
        if report["split"]!="development" or report["dataset_sha256"]!=digest:raise ValueError("Mismatched development reports")
        actual=report["predictions"]
        if len(actual)!=len(expected) or len({r["id"] for r in actual})!=len(actual) or {r["id"] for r in actual}!=ids:
            raise ValueError("Incomplete or duplicate development predictions")
        for row in actual:
            if any(row[key]!=expected_by_id[row["id"]][key] for key in ("pair_id","language","expected")):
                raise ValueError("Development metadata mismatch")
            if set(row["policies"])!=set(report["configs"]):raise ValueError("Missing policy results")
            merged[row["id"]]["policies"].update(row["policies"])
        for name,config in report["configs"].items():
            if name in configs:raise ValueError("Duplicate candidate policy name")
            if config_hash(config)!=report["config_hashes"][name]:raise ValueError("Development config mismatch")
            configs[name]=config;source_by_policy[name]=str(path.resolve());runtime_by_policy[name]=dict(model=report["model"],packages=report["packages"])
            if report["model"]["backend"]=="gliner":
                if "batch_questions" in actual[0]:
                    positive=actual[0]["batch_questions"];negative=actual[0]["batch_negative_questions"];head=name
                    if any(r["batch_questions"]!=positive or r["batch_negative_questions"]!=negative for r in actual):
                        raise ValueError("GLiNER grouping changes across messages")
                else:
                    head=config["head_key"]
                    positive={head:config["questions"]["multiple_actions"]}
                    negative={head:actual[0]["policies"][name]["negative_questions"]["multiple_actions"]}
                    if any(r["policies"][name]["negative_questions"]["multiple_actions"]!=negative[head] for r in actual):
                        raise ValueError("Single-head opposition schema changes")
                executions[name]=dict(head_key=head,positive_questions=positive,negative_questions=negative,
                                      multiple_labels=[label for label in ("multiple","multiple_tasks") if label in positive[head]["criteria"]])
    rows=list(merged.values());names=list(configs)
    winner=max(names,key=lambda name:rank(rows,name))
    all_summaries=summaries(rows,names)
    if winner not in executions:raise ValueError("No deployable categorical candidate wins; retain existing question")
    selected_runtime=runtime_by_policy[winner];model=selected_runtime["model"]
    execution=executions[winner]
    if model["backend"]!="gliner":raise ValueError("Unsupported selected policy backend")
    policy=dict(version="task-count-policy-v1",created_at=datetime.now(timezone.utc).isoformat(),selected=winner,
        dataset_sha256=digest,execution=execution,execution_sha256=config_hash(execution),
        model=dict(model=model["model"],revision=Path(model["resolved_snapshot"]).name,checkpoints=model["checkpoints"],
                   tested_device=model["device"],parameter_dtype=model["parameter_dtype"]),packages=selected_runtime["packages"],
        development_promotion=promotion(rows,winner),heldout_used=False)
    result=dict(version="action-multimodel-selection-v1",created_at=datetime.now(timezone.utc).isoformat(),
        dataset_sha256=digest,selected=winner,config=configs[winner],config_sha256=config_hash(configs[winner]),
        execution_sha256=policy["execution_sha256"],rule="Maximize weaker-language balanced accuracy, then both-language sum, then fewer missed tasks, then lower recorded positive p50 latency",
        development_ranks={name:rank(rows,name) for name in names},development_summary=all_summaries,
        development_promotion=promotion(rows,winner),heldout_used=False,
        comparator_hashes={name:config_hash(configs[name]) for name in ("original","balanced")},
        development_model=runtime_by_policy["balanced"]["model"],development_packages=runtime_by_policy["balanced"]["packages"],
        selected_runtime=selected_runtime,
        sources={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        configs=configs,source_by_policy=source_by_policy)
    for path,value in ((selection_output,result),(policy_output,policy)):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports",nargs="+",type=Path)
    parser.add_argument("--selection-output",type=Path,required=True)
    parser.add_argument("--policy-output",type=Path,required=True)
    args=parser.parse_args();result=freeze(args.reports,selection_output=args.selection_output,policy_output=args.policy_output)
    print("Frozen development winner: "+result["selected"])

if __name__=="__main__":main()
