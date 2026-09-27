"""Diagnose extraction and independent-pair errors against a source-span audit."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path


def diagnose(rows:list[dict],audits:dict, *,variant:str)->dict:
    results={}
    for language in ("all","en","fr"):
        counts=dict(candidate_true_positive=0,candidate_false_positive=0,candidate_false_negative=0,
                    candidate_true_negative=0,gate_loses_all_true_independent_pairs=0,
                    relationship_false_negative_with_true_pair_available=0,
                    false_positive_with_unsupported_span=0,false_positive_with_same_task_or_alternative=0,
                    predicted_independent_pairs=0,incorrect_independent_pairs=0,
                    unresolved_messages=0)
        examples=[]
        for row in rows:
            if language!="all" and row["language"]!=language: continue
            audit=audits[row["id"]]
            if audit["multiple_actions"] != row["expected"]:
                raise ValueError("Source audit differs from reviewed message label")
            gold=set(audit["requested_candidate_ids"])
            ids={c["id"] for c in row["candidates"]}
            if not gold<=ids or set(audit["candidate_task_groups"])!=gold:
                raise ValueError("Invalid audited candidate ids")
            decision=row["decomposition"] if variant=="noul_gate_choice_pairs" else row["choice_pair_result"] if variant=="choice_gate_choice_pairs" else row["split_pair_result"]
            selected=set(decision["selected_candidate_ids"] if "selected_candidate_ids" in decision else row["selected_candidate_ids"])
            counts["candidate_true_positive"]+=len(gold&selected)
            counts["candidate_false_positive"]+=len(selected-gold)
            counts["candidate_false_negative"]+=len(gold-selected)
            counts["candidate_true_negative"]+=len(ids-gold-selected)
            counts["unresolved_messages"]+=decision["unresolved"]
            task_groups=audit["candidate_task_groups"]
            true_pairs={frozenset((a,b)) for a,b in itertools.combinations(gold,2) if task_groups[a]!=task_groups[b]}
            retained_true_pairs=[pair for pair in true_pairs if pair<=selected]
            witnesses=[frozenset(key.split("__")) for key in decision["independent_pair_witnesses"]]
            incorrect=[pair for pair in witnesses if pair not in true_pairs]
            unsupported=any(not pair<=gold for pair in incorrect)
            same_task=any(pair<=gold for pair in incorrect)
            counts["predicted_independent_pairs"]+=len(witnesses)
            counts["incorrect_independent_pairs"]+=len(incorrect)
            if row["expected"] and not decision["multiple_actions"]:
                counts["relationship_false_negative_with_true_pair_available" if retained_true_pairs else "gate_loses_all_true_independent_pairs"]+=1
            if not row["expected"] and decision["multiple_actions"]:
                counts["false_positive_with_unsupported_span"]+=unsupported
                counts["false_positive_with_same_task_or_alternative"]+=same_task
            examples.append(dict(id=row["id"],gate_false_positives=sorted(selected-gold),
                gate_false_negatives=sorted(gold-selected),retained_true_independent_pairs=len(retained_true_pairs),
                incorrect_pair_witnesses=[sorted(pair) for pair in incorrect],notes=audit["notes"]))
        tp=counts["candidate_true_positive"];fp=counts["candidate_false_positive"];fn=counts["candidate_false_negative"]
        counts["candidate_precision"]=tp/(tp+fp) if tp+fp else None
        counts["candidate_recall"]=tp/(tp+fn) if tp+fn else None
        counts["pair_precision"]=(counts["predicted_independent_pairs"]-counts["incorrect_independent_pairs"])/counts["predicted_independent_pairs"] if counts["predicted_independent_pairs"] else None
        results[language]=dict(summary=counts,examples=examples)
    return results


def run(audit:Path,initial:Path,variants:Path,output:Path)->dict:
    annotations=json.loads(audit.read_text())
    audits={r["id"]:r for r in annotations["rows"]}
    first=json.loads(initial.read_text());other=json.loads(variants.read_text())
    if first["dataset_sha256"]!=other["dataset_sha256"]:raise ValueError("Different datasets")
    for report in (first,other):
        if set(audits)!={r["id"] for r in report["predictions"]}:raise ValueError("Audit must cover every case")
    result=dict(schema_version=1,annotation_method=annotations["method"],
                sources={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (audit,initial,variants)},
                dataset_sha256=first["dataset_sha256"],
                note="False-positive cause flags may overlap. Gate recall measures audited proposed spans, not arbitrary unproposed operations. This is diagnostic rubric review, not held-out validation.",
                variants={"noul_gate_choice_pairs":diagnose(first["predictions"],audits,variant="noul_gate_choice_pairs"),
                          "choice_gate_choice_pairs":diagnose(other["predictions"],audits,variant="choice_gate_choice_pairs"),
                          "choice_gate_split_pairs":diagnose(other["predictions"],audits,variant="choice_gate_split_pairs")})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("audit","initial","variants","output"):parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();run(args.audit,args.initial,args.variants,args.output)

if __name__=="__main__":main()
