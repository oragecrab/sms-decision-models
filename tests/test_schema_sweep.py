from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from system_one_models import schema_sweep as sweep
from system_one_models.question_schemas import load_question_schema
from system_one_models.questions import QUESTIONS
from test_benchmark import answers


def fixtures():
    rubric, _ = load_question_schema(sweep.ROOT/'schemas/rubric-v1.json')
    catalog = json.loads((sweep.ROOT/'schemas/wording-catalog-v1.json').read_text())
    return catalog, rubric


def test_catalog_axes_are_unique_and_order_only_variants_keep_text():
    catalog, rubric = fixtures()
    candidates = sweep.build_candidates(catalog, rubric)
    assert len(candidates) >= 250
    assert len({(c['question_id'], c['sha256']) for c in candidates}) == len(candidates)
    for candidate in candidates:
        original = QUESTIONS[candidate['question_id']]
        spec = candidate['definition']
        if candidate['kind'] == 'option_order':
            assert spec['type'] == 'choice'
            assert spec['instructions'] == original['instructions']
            assert spec['criteria'] == original['criteria']
            assert list(spec['criteria']) != list(original['criteria'])
        if candidate['kind'] == 'boolean_labels':
            assert spec['criteria'] == original['criteria']
            assert spec['instructions'] == original['instructions']
    assert QUESTIONS == deepcopy(QUESTIONS)


def test_preflight_rejects_route_changes_and_checks_every_state(monkeypatch):
    states = [{'state': {'body': str(i)}} for i in range(4)]
    calls=[]
    backend=SimpleNamespace(router=SimpleNamespace(route=lambda state,q: {'model':'english'},load=lambda name: 'agent'))
    monkeypatch.setattr(sweep,'_check_token_budget',lambda ctx:calls.append(ctx.states[0]))
    sweep.preflight(backend,states,{'urgent_action':QUESTIONS['urgent_action']})
    assert calls == [e['state'] for e in states]
    backend.router.route=lambda state,q: {'model':'english' if len(q)==8 else 'multilingual'}
    with pytest.raises(ValueError, match='routing'):
        sweep.preflight(backend,states,{'urgent_action':QUESTIONS['urgent_action']})


def test_rejected_definition_never_runs_inference(monkeypatch):
    catalog,rubric=fixtures()
    candidate=sweep.build_candidates(catalog,rubric)[0]
    monkeypatch.setattr(sweep,'preflight',lambda *args: (_ for _ in ()).throw(ValueError('prompt budget')))
    backend=SimpleNamespace(predict=lambda *a,**kw:pytest.fail('no inference'))
    result=sweep.run_candidate(backend,[],candidate)
    assert result['status']=='rejected'
    assert result['reason']=='prompt budget'
    assert result['predictions']==[]


def fake_backend(examples, *, change_grouping=False):
    expected={e['state']['body']:e['expected'] for e in examples}
    calls=[]
    def predict(state, *, questions):
        calls.append((state['body'],list(questions)))
        result=answers(expected[state['body']])
        if change_grouping and len(questions)==1 and 'urgent_action' in questions:
            result['urgent_action']['noul']=1-result['urgent_action']['noul']
        backend.last_routing={'model':'english'}
        return {k:result[k] for k in questions}
    backend=SimpleNamespace(predict=predict,router=SimpleNamespace(loaded=[]),calls=calls)
    return backend


def test_grouping_detects_real_changes_with_identical_message_and_schemas(monkeypatch):
    examples=sweep.load_dataset(sweep.ROOT/'datasets/base_eval_v4.jsonl')[:2]
    monkeypatch.setattr(sweep,'preflight',lambda *a:None)
    baseline=fake_backend(examples)
    same=sweep.grouped_comparison(baseline,examples,deepcopy(QUESTIONS))
    assert same['summary']['changed_labels']==0
    assert same['summary']['max_probability_delta']==0
    assert same['summary']['routing_mismatches']==0
    changed=sweep.grouped_comparison(fake_backend(examples,change_grouping=True),examples,deepcopy(QUESTIONS))
    assert changed['summary']['changed_labels']==2
    assert changed['summary']['per_question_changed']['urgent_action']==2
    assert changed['summary']['policies']['all_at_once']['correct_labels']==16
    assert changed['summary']['policies']['one_by_one']['correct_labels']==14
    assert all(len(row['one_by_one']['question_inference_ms'])==8 for row in changed['predictions'])


def test_checkpoint_resume_reuses_completed_definitions_and_rejects_changed_identity(tmp_path,monkeypatch):
    dataset=sweep.ROOT/'datasets/base_eval_v4.jsonl'
    examples=sweep.load_dataset(dataset)
    backend=fake_backend(examples)
    catalog,rubric=fixtures()
    candidates=[c for c in sweep.build_candidates(catalog,rubric) if c['kind']=='baseline']
    monkeypatch.setattr(sweep,'LayaBackend',lambda **kw:backend)
    from system_one_models import consistency
    def consistency_predict(state, *, questions=None):
        positive = answers(next(e['expected'] for e in examples if e['state'] == state))
        if questions is not None and all(q['type'] == 'noul' for q in questions.values()):
            return {key: {'noul': 1-row['noul'] if 'noul' in row else .2} for key,row in positive.items()}
        return positive
    monkeypatch.setattr(consistency,'create_backend',lambda *a,**kw:SimpleNamespace(predict=consistency_predict,metadata={'model':'fake'}))
    monkeypatch.setattr(sweep,'model_identity',lambda b:{'snapshot':'fixed-test'})
    monkeypatch.setattr(sweep,'build_candidates',lambda *a:candidates)
    monkeypatch.setattr(sweep,'preflight',lambda *a:None)
    first=sweep.run_sweep(dataset=dataset,output=tmp_path/'run')
    backend.calls.clear()
    second=sweep.run_sweep(dataset=dataset,output=tmp_path/'run')
    assert second==first
    # Only the two language warmups preceding identity checks run on resume.
    assert len(backend.calls)==2
    cached_path = tmp_path/'run'/'opposition-selected.json'
    original_cache = cached_path.read_text()
    corrupted = json.loads(original_cache)
    corrupted['dataset_sha256'] = 'stale-dataset'
    cached_path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError,match='Cached opposition report'):
        sweep.run_sweep(dataset=dataset,output=tmp_path/'run')
    cached_path.write_text(original_cache)
    monkeypatch.setattr(sweep,'model_identity',lambda b:{'snapshot':'changed-test'})
    with pytest.raises(ValueError,match='different dataset/model'):
        sweep.run_sweep(dataset=dataset,output=tmp_path/'run')
