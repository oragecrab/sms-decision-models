"""Development-only request gate and supplied-span relationship diagnostics."""
from __future__ import annotations
import argparse, hashlib, itertools, json, os
from pathlib import Path
from importlib.metadata import version
from .action_improvement import load_cases, configurations, config_hash, focused_state, choice_decision
from .action_decomposition import propose_spans, predict_stage
from .backends import LayaBackend
from .schema_sweep import model_identity

NAMES=('target_field','marked_target','marked_compact')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def ratio(a,b): return a/b if b else None

def relationship(audit,key,criteria):
    ids=key.split('__')
    if not set(ids)<=set(audit['requested_candidate_ids']):return 'unsupported'
    value=audit['candidate_pair_relationships'].get(key)
    if value is None:raise ValueError('Missing reviewed requested-pair relationship: '+key)
    if 'one_task_or_alternative' in criteria and value in ('same_task_steps','alternatives'):return 'one_task_or_alternative'
    return value

def summarize(rows,name,language):
    subset=[r for r in rows if language=='all' or r['language']==language]
    tp=fp=fn=tn=0
    saved_correct=saved_total=saved_witnesses=saved_good=0
    oracle_correct=oracle_total=oracle_witnesses=oracle_good=0
    message_correct=message_fp=message_fn=unwitnessable=0
    cached=inferred=0
    for row in subset:
        r=row['variants'][name];g=set(row['requested_candidate_ids']);s=set(r['selected_candidate_ids']);ids=set(row['candidate_ids'])
        tp+=len(g&s);fp+=len(s-g);fn+=len(g-s);tn+=len(ids-g-s)
        for kind in ('saved','supplied'):
            for value in r[kind+'_pair_diagnostics'].values():
                correct=value['predicted']==value['expected'];witness=value['predicted']=='independent_together';good=witness and value['expected']=='independent_together'
                if kind=='saved':saved_correct+=correct;saved_total+=1;saved_witnesses+=witness;saved_good+=good
                else:oracle_correct+=correct;oracle_total+=1;oracle_witnesses+=witness;oracle_good+=good
        predicted=r['supplied_multiple_actions'];expected=row['expected']
        message_correct+=predicted==expected;message_fp+=predicted and not expected;message_fn+=expected and not predicted
        unwitnessable+=expected and not any(v['expected']=='independent_together' for v in r['supplied_pair_diagnostics'].values())
        cached+=len(r['cached_pair_ids']);inferred+=len(r['inferred_pair_ids'])
    return dict(messages=len(subset),gate=dict(true_positive=tp,false_positive=fp,false_negative=fn,true_negative=tn,precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn)),
        saved_pairs=dict(total=saved_total,correct=saved_correct,accuracy=ratio(saved_correct,saved_total),witnesses=saved_witnesses,correct_witnesses=saved_good,witness_precision=ratio(saved_good,saved_witnesses)),
        supplied_pairs=dict(total=oracle_total,correct=oracle_correct,accuracy=ratio(oracle_correct,oracle_total),witnesses=oracle_witnesses,correct_witnesses=oracle_good,witness_precision=ratio(oracle_good,oracle_witnesses)),
        supplied_span_messages=dict(correct=message_correct,accuracy=ratio(message_correct,len(subset)),false_positives=message_fp,false_negatives=message_fn,multiple_without_available_independent_pair=unwitnessable),cached_pairs=cached,inferred_pairs=inferred)

def run(source:Path,audit:Path,dataset:Path,output:Path):
    os.environ['LAYA_MPS_AMP_MIN_ROWS']='10000'
    import torch
    torch.set_num_threads(2)
    original=json.loads(source.read_text());annotations=json.loads(audit.read_text());examples=load_cases(dataset,'development')
    if original['split']!='development' or original['dataset_sha256']!=sha(dataset):raise ValueError('Source must match development dataset')
    saved={r['id']:r for r in original['predictions']}
    if len(saved)!=len(original['predictions']) or set(saved)!={r['id'] for r in examples}:raise ValueError('Development source incomplete or duplicated')
    reviews={r['id']:r for r in annotations['rows'] if r['id'] in saved}
    if set(reviews)!=set(saved):raise ValueError('Development annotations missing')
    configs=configurations()
    for name in NAMES:
        if config_hash(configs[name])!=original['config_hashes'][name] or configs[name]!=original['configs'][name]:raise ValueError('Configuration changed')
    packages={name:version(name) for name in ('laya','torch','transformers','numpy')}
    if packages!=original['packages']:raise ValueError('Package identity changed')
    backend=LayaBackend(device='mps');warmed=set()
    for example in examples:
        if example['language'] not in warmed:
            candidates=propose_spans(example['state'])
            for name in NAMES:
                target=focused_state(example['state'],candidates[:2],marked=configs[name]['marked'])
                predict_stage(backend,target,configs[name]['pair'])
            warmed.add(example['language'])
    identity=model_identity(backend)
    if identity!=original['model']['checkpoints'] or original['model']['torch_threads']!=2:raise ValueError('Runtime checkpoints changed')
    result=dict(schema_version=1,version='action-focused-component-audit-v1',split='development',sources={str(p.resolve()):sha(p) for p in (source,audit,dataset)},
        harness_sha256=sha(Path(__file__)),dataset_sha256=sha(dataset),configs={n:configs[n] for n in NAMES},
        model=dict(backend.metadata,checkpoints=identity,torch_threads=2),packages=packages,
        note='Supplied requested spans are reviewer-selected diagnostic evidence, not end-to-end predictions. No labels or task groups are included in model inputs. No heldout inference. Gate recall concerns mechanically proposed spans. Pair counts do not count exact tasks; a compound span may cover multiple tasks without an independent pair witness.',predictions=[])
    for example in examples:
        old=saved[example['id']];review=reviews[example['id']];candidates=propose_spans(example['state']);gold=set(review['requested_candidate_ids'])
        if old['expected']!=example['expected'] or old['language']!=example['language'] or old['pair_id']!=example['pair_id']:raise ValueError('Source metadata mismatch')
        if not gold<={c['id'] for c in candidates}:raise ValueError('Invalid reviewed candidate')
        row=dict(id=example['id'],pair_id=example['pair_id'],language=example['language'],expected=example['expected'],requested_candidate_ids=review['requested_candidate_ids'],candidate_ids=[c['id'] for c in candidates],compound_span=review.get('compound_span',False),missing_operation=review.get('missing_operation',False),variants={})
        requested=[c for c in candidates if c['id'] in gold]
        for name in NAMES:
            config=configs[name];policy=old['policies'][name]
            if policy['candidates']!=candidates:raise ValueError('Proposals changed')
            criteria=config['pair']['relationship']['criteria'];pair_answers={};pair_states={};cached=[];inferred=[];diagnostics={}
            for a,b in itertools.combinations(requested,2):
                key=f"{a['id']}__{b['id']}";state=focused_state(example['state'],[a,b],marked=config['marked']);pair_states[key]=state
                if key in policy['pair_answers']:
                    if policy['pair_states'][key]!=state:raise ValueError('Cached target evidence differs')
                    answer=policy['pair_answers'][key];cached.append(key)
                else:
                    raw,_=predict_stage(backend,state,config['pair']);answer=raw['relationship'];inferred.append(key)
                pair_answers[key]=answer
                diagnostics[key]=dict(expected=relationship(review,key,criteria),predicted=choice_decision(answer,criteria))
            saved_diag={key:dict(expected=relationship(review,key,criteria),predicted=choice_decision(answer,criteria)) for key,answer in policy['pair_answers'].items()}
            row['variants'][name]=dict(selected_candidate_ids=policy['selected_candidate_ids'],saved_pair_diagnostics=saved_diag,
                supplied_pair_answers=pair_answers,supplied_pair_states=pair_states,cached_pair_ids=cached,inferred_pair_ids=inferred,supplied_pair_diagnostics=diagnostics,
                supplied_multiple_actions=any(v['predicted']=='independent_together' for v in diagnostics.values()))
        result['predictions'].append(row);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(f"{len(result['predictions'])}/{len(examples)} {example['id']}",flush=True)
    result['summary']={name:{language:summarize(result['predictions'],name,language) for language in ('all','en','fr')} for name in NAMES}
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','audit','dataset','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.source,a.audit,a.dataset,a.output)
if __name__=='__main__':main()
