"""Development-only JevK5 comparison of five fixed direct action heads."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from .action_improvement import configurations, config_hash, load_cases, summaries, choice_decision
from .action_decomposition import checked_noul, confirm, predict_stage, OPPOSITION
from .backends import create_backend
from .consistency import negative_questions

NAMES = ('original','balanced','direct_choice','direct_three','direct_explicit')

def positive_schema(configs):
    return {f'jev_{name}': configs[name]['questions']['multiple_actions'] for name in NAMES}

def decode(answer, spec):
    if spec['type']=='noul':
        p=checked_noul(answer)
        return dict(predicted=p>=.5, unresolved=p==.5, answer_confidence=max(p,1-p))
    value=choice_decision(answer,spec['criteria'])
    return dict(predicted=value in ('multiple','multiple_tasks'),unresolved=value is None,
                answer_confidence=answer['probabilities'][answer['choice']])

def opposition_schema(answers,configs):
    result={}
    for name in NAMES:
        key=f'jev_{name}';spec=configs[name]['questions']['multiple_actions']
        if spec['type']=='noul':
            result[key]=negative_questions({'multiple_actions':answers[key]},questions={'multiple_actions':spec})['multiple_actions']
        else: result[key]=OPPOSITION['multiple_actions']
    return result

def evaluate(backend,state,configs):
    questions=positive_schema(configs)
    answers,positive_ms=predict_stage(backend,state,questions)
    negative_questions_=opposition_schema(answers,configs)
    negatives,negative_ms=predict_stage(backend,state,negative_questions_)
    policies={}
    for name in NAMES:
        key=f'jev_{name}'
        result=decode(answers[key],questions[key])
        result.update(answers={'multiple_actions':answers[key]},questions={'multiple_actions':questions[key]},
            negative={'multiple_actions':negatives[key]},negative_questions={'multiple_actions':negative_questions_[key]},
            positive_ms=positive_ms,negative_ms=negative_ms,
            accepted=not result['unresolved'] and confirm(result['predicted'],negatives[key]))
        policies[key]=result
    return dict(policies=policies,batch_questions=questions,batch_answers=answers,
                batch_negative_questions=negative_questions_,batch_negative_answers=negatives,
                batch_positive_ms=positive_ms,batch_negative_ms=negative_ms)

def checkpoint_identity(backend):
    snapshot=Path(backend.metadata['resolved_snapshot'])
    return dict(snapshot=str(snapshot), files={p.name:dict(resolved=str(p.resolve()),bytes=p.stat().st_size,
                **({'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} if p.suffix=='.json' else {}))
                for p in sorted(snapshot.iterdir()) if p.is_file() and p.suffix in ('.json','.safetensors','.bin')})

def run(*,dataset,output,device='cpu'):
    os.environ['HF_HUB_OFFLINE']='1'
    import torch
    torch.set_num_threads(2)
    configs={name:configurations()[name] for name in NAMES}
    examples=load_cases(dataset,'development')
    backend=create_backend('alibiserikbay/JevK5',device=device)
    warmed=set()
    for example in examples:
        if example['language'] not in warmed:
            started=time.perf_counter()
            evaluate(backend,example['state'],configs)
            elapsed=time.perf_counter()-started
            warmed.add(example['language'])
            print(f"JevK5 {example['language']} warmup success ({elapsed:.2f}s); excluded from scores/timing",flush=True)
    metadata=dict(backend.metadata,torch_threads=2,checkpoints=checkpoint_identity(backend))
    metadata['parameter_dtype']=backend.metadata['dtype']
    report=dict(schema_version=1,version='action-improvement-jev-development-v1',split='development',
        created_at=datetime.now(timezone.utc).isoformat(),dataset=str(dataset.resolve()),
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        configs={f'jev_{name}':configs[name] for name in NAMES},
        config_hashes={f'jev_{name}':config_hash(configs[name]) for name in NAMES},
        model=metadata,packages={name:version(name) for name in ('jevk5','torch','transformers','numpy')},
        timing_note='Each message evaluates five positive and five negative heads sequentially through the native adapter. Row batch durations are credited once. Repeated policy positive_ms/negative_ms refer to shared batch duration, not independent head latency. Warmup excluded.',
        predictions=[])
    output.parent.mkdir(parents=True,exist_ok=True)
    for i,example in enumerate(examples):
        result=evaluate(backend,example['state'],configs)
        row=dict(id=example['id'],pair_id=example['pair_id'],language=example['language'],expected=example['expected'],**result)
        report['predictions'].append(row)
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        if (i+1)%8==0:print(f'JevK5 development {i+1}/{len(examples)}',flush=True)
    report['summary']=summaries(report['predictions'],list(report['configs']))
    report['total_scored_inference_ms']=sum(r['batch_positive_ms']+r['batch_negative_ms'] for r in report['predictions'])
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    run(dataset=args.dataset,output=args.output,device=args.device)

if __name__=='__main__':main()
