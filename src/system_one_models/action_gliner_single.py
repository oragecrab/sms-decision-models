"""Single-head GLiNER action configuration, validated on development only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from .action_improvement import DIRECT_THREE, config_hash, load_cases, summaries
from .action_decomposition import confirm, predict_stage, OPPOSITION
from .action_gliner import checkpoint_identity, decode
from .backends import create_backend

HEAD_KEY='gliner_direct_three'
POLICY='gliner_three_single'
CONFIG=dict(kind='direct',questions=DIRECT_THREE,head_key=HEAD_KEY,execution='single_head')

def evaluate(backend,state):
    questions={HEAD_KEY:CONFIG['questions']['multiple_actions']}
    negative_questions={HEAD_KEY:OPPOSITION['multiple_actions']}
    answers,positive_ms=predict_stage(backend,state,questions)
    negatives,negative_ms=predict_stage(backend,state,negative_questions)
    result=decode(answers[HEAD_KEY],questions[HEAD_KEY])
    result.update(answers={'multiple_actions':answers[HEAD_KEY]},questions={'multiple_actions':questions[HEAD_KEY]},
                  negative={'multiple_actions':negatives[HEAD_KEY]},
                  negative_questions={'multiple_actions':negative_questions[HEAD_KEY]},
                  positive_ms=positive_ms,negative_ms=negative_ms,
                  accepted=not result['unresolved'] and confirm(result['predicted'],negatives[HEAD_KEY]))
    return dict(policies={POLICY:result},model_questions=questions,model_answers=answers,
                model_negative_questions=negative_questions,model_negative_answers=negatives)

def run(*,dataset,output,comparison,device='mps'):
    os.environ['HF_HUB_OFFLINE']='1'
    import torch
    torch.set_num_threads(2)
    examples=load_cases(dataset,'development')
    prior=json.loads(comparison.read_text())
    dataset_hash=hashlib.sha256(dataset.read_bytes()).hexdigest()
    if prior['dataset_sha256']!=dataset_hash or prior['split']!='development':
        raise ValueError('Batch comparator must use same development dataset')
    if prior['configs']['gliner_direct_three']['questions']!=CONFIG['questions']:
        raise ValueError('Batch comparator schema differs')
    backend=create_backend('fastino/GLiNER2.5-Decide',device=device)
    warmed=set()
    for example in examples:
        if example['language'] not in warmed:
            evaluate(backend,example['state']);warmed.add(example['language'])
            print(f"GLiNER single {example['language']} warmup success; excluded",flush=True)
    report=dict(schema_version=1,version='action-improvement-gliner-single-development-v1',split='development',
        created_at=datetime.now(timezone.utc).isoformat(),dataset=str(dataset.resolve()),dataset_sha256=dataset_hash,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        configs={POLICY:CONFIG},config_hashes={POLICY:config_hash(CONFIG)},
        model=dict(backend.metadata,torch_threads=2,checkpoints=checkpoint_identity(backend),parameter_dtype=str(next(backend.model.parameters()).dtype)),
        packages={name:version(name) for name in ('gliner2','torch','transformers','numpy')},
        timing_note='One single-head positive request and one actual single-head opposite request per example; warmup excluded.',
        comparator=str(comparison.resolve()),comparator_sha256=hashlib.sha256(comparison.read_bytes()).hexdigest(),predictions=[])
    output.parent.mkdir(parents=True,exist_ok=True)
    for i,example in enumerate(examples):
        result=evaluate(backend,example['state'])
        report['predictions'].append(dict(id=example['id'],pair_id=example['pair_id'],language=example['language'],expected=example['expected'],**result))
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        if (i+1)%8==0:print(f'GLiNER single development {i+1}/{len(examples)}',flush=True)
    report['summary']=summaries(report['predictions'],[POLICY])
    old={row['id']:row for row in prior['predictions']}
    changes=[]
    for row in report['predictions']:
        previous=old[row['id']]['policies']['gliner_direct_three'];current=row['policies'][POLICY]
        if previous['predicted']!=current['predicted']:
            changes.append(dict(id=row['id'],language=row['language'],expected=row['expected'],batch_predicted=previous['predicted'],single_predicted=current['predicted']))
    report['grouping_comparison']=dict(changed_labels=len(changes),total_labels=len(examples),changes=changes,
        batch_correct=prior['summary']['gliner_direct_three']['overall']['correct'],
        single_correct=report['summary'][POLICY]['overall']['correct'],
        max_confidence_delta=max(abs(row['policies'][POLICY]['answer_confidence']-old[row['id']]['policies']['gliner_direct_three']['answer_confidence']) for row in report['predictions']))
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--device',default='mps')
    args=parser.parse_args()
    run(dataset=args.dataset,output=args.output,comparison=args.comparison,device=args.device)

if __name__=='__main__':main()
