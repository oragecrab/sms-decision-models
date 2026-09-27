"""Regression assessment of a frozen task policy on an existing canonical dataset."""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
from .action_policy import TaskCountPolicy
from .action_improvement import summaries
from .benchmark import load_dataset


def run(*,policy_path:Path,dataset:Path,comparators:Path,output:Path,device:str="mps")->dict:
    examples=load_dataset(dataset)
    baseline=json.loads(comparators.read_text())
    if baseline["dataset_sha256"]!=hashlib.sha256(dataset.read_bytes()).hexdigest():raise ValueError("Comparator dataset mismatch")
    cached={r["id"]:r for r in baseline["predictions"]}
    if set(cached)!={r["id"] for r in examples}:raise ValueError("Comparator must cover every case")
    policy=TaskCountPolicy(policy_path,device=device)
    warmed=set()
    for example in examples:
        if example["language"] not in warmed:
            policy.predict(example["state"],check_opposition=True);warmed.add(example["language"])
    name=policy.policy["selected"]
    report=dict(schema_version=1,version="task-count-v4-regression-v1",created_at=datetime.now(timezone.utc).isoformat(),
        dataset=str(dataset.resolve()),dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        comparator_source=str(comparators.resolve()),comparator_sha256=hashlib.sha256(comparators.read_bytes()).hexdigest(),
        frozen_policy=policy.policy,comparator_runtime=baseline["model"],predictions=[])
    output.parent.mkdir(parents=True,exist_ok=True)
    for i,example in enumerate(examples):
        old=cached[example["id"]]
        if old["state"]!=example["state"] or old["expected"]!=example["expected"]["multiple_actions"] or old["language"]!=example["language"]:
            raise ValueError("Comparator evidence/label mismatch")
        value=policy.predict(example["state"],check_opposition=True);value["predicted"]=value["multiple_actions"] is True
        row=dict(id=example["id"],pair_id=example["pair_id"],language=example["language"],expected=old["expected"],
                 policies={"original":old["policies"]["baseline"],"balanced":old["policies"]["balanced"],name:value})
        report["predictions"].append(row)
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        if (i+1)%8==0:print(f"Frozen task-count v4 regression {i+1}/{len(examples)}",flush=True)
    report["summary"]=summaries(report["predictions"],["original","balanced",name])
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("policy","dataset","comparators","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",default="mps")
    args=p.parse_args();run(policy_path=args.policy,dataset=args.dataset,comparators=args.comparators,output=args.output,device=args.device)

if __name__=="__main__":main()
