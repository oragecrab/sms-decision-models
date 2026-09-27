"""Explicit task-count policy with intact categorical outputs and independent checks."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
from .action_improvement import choice_decision,config_hash,load_cases,summaries
from .action_decomposition import confirm,predict_stage
from .backends import create_backend


def validate_policy(policy:dict)->None:
    if policy.get("version")!="task-count-policy-v1":raise ValueError("Unsupported task-count policy version")
    execution=policy["execution"]
    if config_hash(execution)!=policy["execution_sha256"]:raise ValueError("Task-count policy hash mismatch")
    if execution.get("decision","category_argmax") not in ("category_argmax","binary_probability"):
        raise ValueError("Unsupported task-count decision rule")
    if execution.get("decision")=="binary_probability" and (type(execution.get("positive_threshold")) not in (float,int) or not 0<execution["positive_threshold"]<1):
        raise ValueError("Invalid binary task-count threshold")
    if execution["head_key"] not in execution["positive_questions"] or execution["head_key"] not in execution["negative_questions"]:
        raise ValueError("Missing task-count head")
    spec=execution["positive_questions"][execution["head_key"]]
    labels=execution["multiple_labels"]
    if (spec["type"]!="choice" or not isinstance(labels,list) or not labels
            or any(not isinstance(label,str) for label in labels)
            or len(set(labels))!=len(labels) or not set(labels)<set(spec["criteria"])):
        raise ValueError("Task-count policy needs unique nonempty categorical multiple-task labels and a negative category")
    negative=execution["negative_questions"][execution["head_key"]]
    if negative.get("type")!="noul" or set(negative.get("criteria",{}))!={"true","false"}:
        raise ValueError("Task-count opposition head must be a Boolean complementary question")


class TaskCountPolicy:
    def __init__(self,path:Path,*,device:str="mps",backend=None):
        self.path=path
        self.policy=json.loads(path.read_text());validate_policy(self.policy)
        self.execution=self.policy["execution"]
        if backend is None:
            os.environ["HF_HUB_OFFLINE"]="1"
            import torch
            torch.set_num_threads(2)
            self.backend=create_backend(self.policy["model"]["model"],device=device,revision=self.policy["model"]["revision"])
            from .action_gliner import checkpoint_identity
            actual=checkpoint_identity(self.backend)
            if actual!=self.policy["model"]["checkpoints"]:raise ValueError("Task-count checkpoint differs from evaluated policy")
            expected=self.policy["packages"]
            if any(version(name)!=value for name,value in expected.items()):raise ValueError("Task-count runtime packages differ from evaluated policy")
        else:self.backend=backend

    def predict(self,state:dict,*,check_opposition:bool=False)->dict:
        started=time.perf_counter()
        answers,_=predict_stage(self.backend,state,self.execution["positive_questions"])
        elapsed=(time.perf_counter()-started)*1000
        head=self.execution["head_key"];answer=answers[head]
        band=choice_decision(answer,self.execution["positive_questions"][head]["criteria"])
        p_multiple=sum(answer["probabilities"][label] for label in self.execution["multiple_labels"])
        if self.execution.get("decision","category_argmax")=="binary_probability":
            cutoff=self.execution["positive_threshold"]
            unresolved=p_multiple==cutoff
            predicted=None if unresolved else p_multiple>cutoff
        else:
            unresolved=band is None
            predicted=None if unresolved else band in self.execution["multiple_labels"]
        result=dict(multiple_actions=predicted,task_count_band=band,
                    probabilities=answer["probabilities"],selected_category_probability=answer["probabilities"][answer["choice"]],
                    unresolved=unresolved,multiple_actions_probability=p_multiple,
                    decision_rule=self.execution.get("decision","category_argmax"),positive_answers=answers,positive_ms=elapsed,
                    policy_sha256=self.policy["execution_sha256"],model=dict(self.backend.metadata))
        if check_opposition:
            negative,negative_ms=predict_stage(self.backend,state,self.execution["negative_questions"])
            result.update(negative=negative,negative_ms=negative_ms,
                          accepted=predicted is not None and confirm(predicted,negative[head]))
        return result


def evaluate(*,policy_path:Path,dataset:Path,output:Path,split:str,device:str="mps")->dict:
    policy=TaskCountPolicy(policy_path,device=device)
    if policy.policy["dataset_sha256"]!=hashlib.sha256(dataset.read_bytes()).hexdigest():
        raise ValueError("Frozen selection uses a different action dataset")
    if split!="heldout":raise ValueError("This assessment requires the reserved heldout split")
    examples=load_cases(dataset,split)
    warmed=set()
    for example in examples:
        if example["language"] not in warmed:
            policy.predict(example["state"],check_opposition=True);warmed.add(example["language"])
    name=policy.policy["selected"]
    report=dict(schema_version=1,version="frozen-task-count-heldout-v1",created_at=datetime.now(timezone.utc).isoformat(),
        split=split,dataset=str(dataset.resolve()),dataset_sha256=policy.policy["dataset_sha256"],
        frozen_policy=policy.policy,policy_file_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),predictions=[])
    output.parent.mkdir(parents=True,exist_ok=True)
    for i,example in enumerate(examples):
        value=policy.predict(example["state"],check_opposition=True)
        # Scoring assigns unresolved cases false consistently with development,
        # but unresolved is retained and they can never pass opposition.
        value["predicted"]=value["multiple_actions"] is True
        row=dict(id=example["id"],pair_id=example["pair_id"],language=example["language"],expected=example["expected"],policies={name:value})
        report["predictions"].append(row)
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        if (i+1)%8==0:print(f"Frozen task-count heldout {i+1}/{len(examples)}",flush=True)
    report["summary"]=summaries(report["predictions"],[name])
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy",type=Path,required=True)
    parser.add_argument("--device",default="mps")
    parser.add_argument("--body")
    parser.add_argument("--sender",default="")
    parser.add_argument("--subject",default="")
    parser.add_argument("--channel",default="sms",choices=("sms","email"))
    parser.add_argument("--opposition",action="store_true")
    parser.add_argument("--dataset",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--split",default="heldout",choices=("heldout",))
    args=parser.parse_args()
    if args.dataset:
        if args.body or args.output is None:parser.error("Dataset assessment needs --output and no --body")
        evaluate(policy_path=args.policy,dataset=args.dataset,output=args.output,split=args.split,device=args.device)
    else:
        if not args.body:parser.error("Provide --body or --dataset")
        from .classifier import build_state
        state=build_state(channel=args.channel,sender=args.sender,subject=args.subject,body=args.body)
        result=TaskCountPolicy(args.policy,device=args.device).predict(state,check_opposition=args.opposition)
        print(json.dumps(result,indent=2,allow_nan=False))

if __name__=="__main__":main()
