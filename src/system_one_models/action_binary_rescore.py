"""Rescore saved categorical action outputs using a frozen binary probability rule."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from .action_improvement import summaries, choice_decision
from .action_decomposition import confirm
from .action_policy import validate_policy


def validate_source(report:dict,policy:dict)->None:
    """Accept either original batched development or frozen execution provenance."""
    execution=policy["execution"]
    frozen=report.get("frozen_policy")
    if frozen is not None:
        validate_policy(frozen)
        for key in ("positive_questions","negative_questions","head_key","multiple_labels"):
            if frozen["execution"][key]!=execution[key]:raise ValueError("Frozen source execution differs: "+key)
        if frozen["model"]!=policy["model"] or frozen["packages"]!=policy["packages"]:
            raise ValueError("Frozen source runtime differs")
        if frozen["selected"]!=policy["previous_selection"]:raise ValueError("Frozen source selection differs")
    else:
        model=report.get("model",{})
        expected=policy["model"]
        if (report.get("packages")!=policy["packages"]
                or any(model.get(key)!=expected[key] for key in ("model","checkpoints","parameter_dtype"))
                or (model.get("revision") or Path(model.get("resolved_snapshot", "")).name)!=expected["revision"]):
            raise ValueError("Saved source runtime differs")
    dataset=report.get("dataset")
    if dataset is None or hashlib.sha256(Path(dataset).read_bytes()).hexdigest()!=report["dataset_sha256"]:
        raise ValueError("Saved source dataset provenance differs")
    rows=report["predictions"]
    if not rows or len({r["id"] for r in rows})!=len(rows):raise ValueError("Empty or duplicate saved predictions")
    for row in rows:
        value=row["policies"][policy["previous_selection"]]
        if frozen is not None:
            if value.get("policy_sha256")!=frozen["execution_sha256"]:
                raise ValueError("Saved prediction execution hash differs")
            model=value.get("model",{})
            if any(model.get(key)!=policy["model"][key] for key in ("model","revision")):
                raise ValueError("Saved prediction runtime differs")
        elif (row.get("batch_questions")!=execution["positive_questions"]
                or row.get("batch_negative_questions")!=execution["negative_questions"]):
            raise ValueError("Saved positive or negative grouping differs")


def rescore(source:Path,policy_path:Path,output:Path)->dict:
    report=json.loads(source.read_text());policy=json.loads(policy_path.read_text());validate_policy(policy)
    if policy["execution"].get("decision")!="binary_probability":raise ValueError("Expected frozen binary decoder")
    validate_source(report,policy)
    chosen=policy["previous_selection"];name=policy["selected"];head=policy["execution"]["head_key"]
    expected_questions=policy["execution"]["positive_questions"]
    threshold=policy["execution"]["positive_threshold"]
    rows=[]
    for original in report["predictions"]:
        value=original["policies"][chosen]
        if "batch_questions" in original and original["batch_questions"]!=expected_questions:raise ValueError("Saved positive grouping differs")
        if "frozen_policy" in report and report["frozen_policy"]["execution"]["positive_questions"]!=expected_questions:raise ValueError("Frozen source grouping differs")
        answers=value.get("positive_answers",value.get("answers"))
        answer=answers.get(head,answers.get("multiple_actions"))
        choice_decision(answer,expected_questions[head]["criteria"])
        probabilities=answer["probabilities"]
        probability=sum(probabilities[label] for label in policy["execution"]["multiple_labels"])
        negative=value["negative"].get(head,value["negative"].get("multiple_actions"))
        unresolved=probability==threshold;predicted=probability>threshold
        result=dict(value,predicted=predicted,multiple_actions=predicted if not unresolved else None,
            multiple_actions_probability=probability,probabilities=probabilities,unresolved=unresolved,
            accepted=not unresolved and confirm(predicted,negative),decision_rule="binary_probability")
        row=dict(original);row["policies"]=dict(original["policies"],**{name:result});rows.append(row)
    return_value=dict(schema_version=1,version="frozen-binary-rescore-v2",source=str(source.resolve()),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),dataset_sha256=report["dataset_sha256"],
        frozen_policy=policy,no_inference=True,note=f"Frozen threshold {threshold:g} binary category aggregation; does not manufacture or complement negative inference.",
        predictions=rows,summary=summaries(rows,list(rows[0]["policies"])))
    output.write_text(json.dumps(return_value,indent=2,allow_nan=False)+"\n")
    return return_value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("source","policy","output"):p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args();rescore(args.source,args.policy,args.output)

if __name__=="__main__":main()
