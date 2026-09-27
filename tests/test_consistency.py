from copy import deepcopy
from types import SimpleNamespace

import pytest

from system_one_models import consistency
from system_one_models.benchmark import DEFAULT_DATASET, load_dataset
from system_one_models.questions import QUESTIONS
from test_benchmark import answers


def test_negative_schema_targets_choice_and_inverts_boolean_criteria():
    expected = load_dataset(DEFAULT_DATASET)[0]["expected"]
    positive = answers(expected)
    negative = consistency.negative_questions(positive)
    assert set(negative) == set(QUESTIONS)
    assert all(q["type"] == "noul" for q in negative.values())
    assert positive["classification"]["choice"] in negative["classification"]["instructions"]
    assert "NOT" in negative["classification"]["instructions"]
    assert negative["urgent_action"]["criteria"]["true"] == QUESTIONS["urgent_action"]["criteria"]["false"]
    assert QUESTIONS["classification"]["type"] == "choice"


def test_consistency_accepts_complements_rejects_disagreement_and_ties():
    expected = load_dataset(DEFAULT_DATASET)[0]["expected"]
    positive = answers(expected)
    negative = {key: {"noul": 1 - row["noul"] if "noul" in row else .2} for key, row in positive.items()}
    assert all(consistency.agreement(positive, negative).values())
    negative["classification"]["noul"] = .8
    negative["urgent_action"]["noul"] = positive["urgent_action"]["noul"]
    negative["sentiment"]["noul"] = .5
    accepted = consistency.agreement(positive, negative)
    assert not accepted["classification"] and not accepted["urgent_action"] and not accepted["sentiment"]
    negative["industry"]["noul"] = float("nan")
    with pytest.raises(ValueError, match="Invalid negative"):
        consistency.agreement(positive, negative)


def test_empty_acceptance_has_zero_coverage_and_undefined_accuracy():
    rows = [{"accepted": {key: False for key in QUESTIONS}, "matches": {key: True for key in QUESTIONS}, "exact_match": True}]
    summary = consistency.summarize(rows)
    assert summary["coverage"] == 0
    assert summary["selective_accuracy"] is None
    assert summary["selective_exact_match"] is None


def test_evaluator_uses_separate_requests_and_counts_confidently_wrong_agreement(monkeypatch, tmp_path):
    example = load_dataset(DEFAULT_DATASET)[0]
    monkeypatch.setattr(consistency, "load_dataset", lambda path: [example])
    positive = answers(example["expected"])
    positive["classification"]["choice"] = next(label for label in QUESTIONS["classification"]["criteria"] if label != example["expected"]["classification"])
    negative = {key: {"noul": 1 - row["noul"] if "noul" in row else .2} for key, row in positive.items()}
    calls = []
    def predict(state, *, questions=None):
        calls.append(questions)
        return deepcopy(positive if questions is None else negative)
    monkeypatch.setattr(consistency, "create_backend", lambda *a, **kw: SimpleNamespace(predict=predict, metadata={"model": "fake"}))
    report = consistency.evaluate_consistency(["fake"], output=tmp_path / "report.json")
    assert len(calls) == 4  # distinct positive/negative calls in warmup and evaluation
    assert calls[0] is None and calls[1] is not None
    assert calls[2] is None and calls[3] is not None
    assert set(report["runs"][0]["latency_ms"]) == {"positive", "negative", "combined"}
    assert "packages" in report
    summary = report["runs"][0]["summary"]
    assert summary["coverage"] == 1
    assert summary["accepted_errors"] == 1
    assert summary["selective_accuracy"] == pytest.approx(7/8)


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_positive_boolean_probability_is_rejected(probability):
    positive = answers(load_dataset(DEFAULT_DATASET)[0]["expected"])
    negative = {key: {"noul": .2} for key in QUESTIONS}
    positive["urgent_action"]["noul"] = probability
    with pytest.raises(ValueError, match="Invalid positive"):
        consistency.agreement(positive, negative)


def test_empty_summary_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        consistency.summarize([])


def test_cli_dispatches_consistency_and_requires_eval(monkeypatch):
    import sys
    from system_one_models import cli
    calls = []
    monkeypatch.setattr(cli, "evaluate_consistency", lambda models, **kw: calls.append((models, kw)))
    monkeypatch.setattr(sys, "argv", ["system-one-models", "--eval", "--consistency", "--model", "laya"])
    cli.main()
    assert calls[0][0] == ["laya"]
    monkeypatch.setattr(sys, "argv", ["system-one-models", "--consistency"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_third_request_is_blind_and_only_targets_disputes():
    expected = load_dataset(DEFAULT_DATASET)[0]["expected"]
    positive = answers(expected)
    accepted = {key: True for key in QUESTIONS}
    accepted['classification'] = accepted['urgent_action'] = False
    negative = {key: {'noul': .8} for key in QUESTIONS}
    row = {'positive': positive, 'negative': negative, 'accepted': accepted,
           'expected': expected, 'matches': {key: True for key in QUESTIONS}}
    calls = []
    def predict(state, *, questions):
        calls.append(questions)
        result = {key: deepcopy(positive[key]) for key in questions}
        result['urgent_action']['noul'] = 1 - positive['urgent_action']['noul']
        return result
    consistency.resolve_disagreements(SimpleNamespace(predict=predict), {}, row)
    assert list(calls[0]) == ['classification', 'urgent_action']
    assert calls[0] == consistency.third_questions(['classification', 'urgent_action'])
    assert 'Candidate answer:' not in str(calls[0])
    assert row['resolutions']['classification']['supported']
    # Negative true implies original condition false, so set the third to false.
    row['third']['urgent_action']['noul'] = .2
    def corrected_predict(state, *, questions):
        return row['third']
    consistency.resolve_disagreements(SimpleNamespace(predict=corrected_predict), {}, row)
    assert row['resolutions']['urgent_action']['supported']
    summary = consistency.summarize_third([row])
    assert summary['disputed_labels'] == 2
    assert summary['accepted_labels_after_third'] == 8


def test_choice_veto_does_not_vote_for_third_alternative():
    expected = load_dataset(DEFAULT_DATASET)[0]['expected']
    positive = answers(expected)
    third = answers(expected)
    third['classification']['choice'] = next(k for k in QUESTIONS['classification']['criteria'] if k != expected['classification'])
    chosen = third['classification']['choice']
    third['classification']['probabilities'] = {k: .8 if k == chosen else .1 for k in QUESTIONS['classification']['criteria']}
    row = {'positive': positive, 'negative': {'classification': {'noul': .8}},
           'accepted': {k: k != 'classification' for k in QUESTIONS},
           'expected': expected, 'matches': {k: True for k in QUESTIONS}}
    backend = SimpleNamespace(predict=lambda state, *, questions: {'classification': third['classification']})
    consistency.resolve_disagreements(backend, {}, row)
    assert not row['resolutions']['classification']['supported']
    summary = consistency.summarize_third([row])
    assert summary['unresolved_disputes'] == 1
    assert summary['correct_answers_broken'] == 1
    assert summary['replace_disputes_accuracy'] == pytest.approx(7/8)


def test_no_disagreement_skips_third_request():
    row = {'accepted': {k: True for k in QUESTIONS},
           'matches': {k: True for k in QUESTIONS}}
    def predict(*args, **kwargs):
        pytest.fail('No third request should run')
    consistency.resolve_disagreements(SimpleNamespace(predict=predict), {}, row)
    assert row['third'] == {} and row['third_ms'] == 0
    summary = consistency.summarize_third([row])
    assert summary['third_requests'] == 0
    assert summary['third_accuracy_on_disputes'] is None
    assert summary['coverage_after_third'] == 1


def test_third_evaluator_records_separate_request_and_preserves_gold_outside_prompt(monkeypatch, tmp_path):
    import json
    example = load_dataset(DEFAULT_DATASET)[0]
    monkeypatch.setattr(consistency, 'load_dataset', lambda path: [example])
    positive = answers(example['expected'])
    negative = {key: {'noul': 1 - answer['noul'] if 'noul' in answer else .2}
                for key, answer in positive.items()}
    negative['classification']['noul'] = .8
    calls = []
    def predict(state, *, questions=None):
        calls.append((state, questions))
        if questions is None:
            return deepcopy(positive)
        if all(q['type'] == 'noul' for q in questions.values()):
            return deepcopy(negative)
        return {key: deepcopy(positive[key]) for key in questions}
    monkeypatch.setattr(consistency, 'create_backend', lambda *a, **kw: SimpleNamespace(predict=predict, metadata={'model': 'fake'}))
    output = tmp_path / 'third.json'
    report = consistency.evaluate_consistency(['fake'], output=output, third_request=True)
    assert len(calls) == 6  # three warmups, then positive/negative/disputed-third
    assert calls[-1] == (example['state'], consistency.third_questions(['classification']))
    assert json.loads(output.read_text()) == report
    summary = report['runs'][0]['third_summary']
    assert summary['disputed_labels'] == 1
    assert summary['accepted_labels_after_third'] == 8
    assert summary['selective_accuracy_after_third'] == 1
    run = report["runs"][0]
    assert run["per_language"]["en"]["examples"] == 1
    assert run["consistency_per_language"]["en"]["accepted_labels"] == 7
    assert run["third_per_language"]["en"]["accepted_labels_after_third"] == 8

def test_custom_schema_drives_positive_and_negative_requests_and_saved_hash(monkeypatch, tmp_path):
    import json
    from system_one_models.question_schemas import schema_hash
    example = load_dataset(DEFAULT_DATASET)[0]
    dataset = tmp_path/'data.jsonl'
    # Remove pair annotations so a single row is valid.
    example = {key:value for key,value in example.items() if key not in {'pair_id','source_language','is_translation'}}
    dataset.write_text(json.dumps(example)+'\n')
    questions = deepcopy(QUESTIONS)
    questions['classification']['instructions'] = 'Use this custom risk rubric.'
    questions['classification']['criteria']['suspected_scam'] = 'Custom suspicious behavior.'
    questions['urgent_action']['criteria'] = {'true':'Custom prompt-action condition.', 'false':'Custom absence of that condition.'}
    questions['urgent_action']['labels'] = {'true':'X','false':'Y'}
    path = tmp_path/'schema.json'
    path.write_text(json.dumps({'version':'custom-v1','questions':questions}))
    positive = answers(example['expected'])
    negative = {key:{'noul':1-row['noul'] if 'noul' in row else .2} for key,row in positive.items()}
    # Keep a confidently wrong accepted decision visible in counts.
    positive['industry']['choice'] = 'healthcare'
    positive['industry']['probabilities'] = {key:.8 if key=='healthcare' else .2/9 for key in QUESTIONS['industry']['criteria']}
    calls = []
    def predict(state, *, questions=None):
        calls.append(deepcopy(questions))
        if all(spec['type']=='noul' for spec in questions.values()):
            return deepcopy(negative)
        return deepcopy(positive)
    monkeypatch.setattr(consistency,'create_backend',lambda *a,**kw:SimpleNamespace(predict=predict,metadata={'model':'fake'}))
    output = tmp_path/'report.json'
    report = consistency.evaluate_consistency(['fake'],dataset=dataset,output=output,question_schema=path)
    assert calls[0] == calls[2] == questions
    assert 'Use this custom risk rubric.' in calls[3]['classification']['instructions']
    assert 'Custom suspicious behavior.' in calls[3]['classification']['instructions']
    assert 'Classify by risk, not claimed authority.' not in calls[3]['classification']['instructions']
    assert calls[3]['urgent_action']['criteria']['true'] == questions['urgent_action']['criteria']['false']
    assert calls[3]['urgent_action']['labels'] == {'true':'X','false':'Y'}
    assert report['question_schema']['sha256'] == schema_hash(questions)
    assert report['questions'] == questions
    assert json.loads(output.read_text()) == report
    metrics = report['runs'][0]['summary']
    assert metrics['agreement_labels'] == 8
    assert metrics['per_question']['industry']['accepted_errors'] == 1
    assert metrics['per_question']['industry']['rejected_errors'] == 0
    assert sum(m['accepted_errors'] for m in metrics['per_question'].values()) == metrics['accepted_errors'] == 1
    assert report['runs'][0]['consistency_per_language']['en']['agreement_labels'] == 8
    third = consistency.third_questions(['classification'],questions=questions)
    assert questions['classification']['instructions'] in third['classification']['instructions']


def test_cli_accepts_custom_schema_consistency(monkeypatch):
    import sys
    from pathlib import Path
    from system_one_models import cli
    calls=[]
    monkeypatch.setattr(cli,'evaluate_consistency',lambda models,**kw:calls.append(kw))
    monkeypatch.setattr(sys,'argv',['system-one-models','--eval','--consistency','--question-schema','custom.json'])
    cli.main()
    assert calls[0]['question_schema'] == Path('custom.json')


def test_opposition_summary_reconciles_rejected_correct_and_wrong_answers():
    questions = {'urgent_action': QUESTIONS['urgent_action']}
    rows = [
        {'language':'en','accepted':{'urgent_action':True},'matches':{'urgent_action':False},'exact_match':False},
        {'language':'en','accepted':{'urgent_action':False},'matches':{'urgent_action':False},'exact_match':False},
        {'language':'fr','accepted':{'urgent_action':False},'matches':{'urgent_action':True},'exact_match':True},
        {'language':'fr','accepted':{'urgent_action':True},'matches':{'urgent_action':True},'exact_match':True},
    ]
    summary = consistency.summarize(rows,questions=questions)
    assert summary['agreement_labels']==2 and summary['disagreement_labels']==2
    assert summary['accepted_errors']==1 and summary['rejected_errors']==1
    assert summary['selective_accuracy']==.5
    assert summary['per_question']['urgent_action']['rejected']==2
    for language in ['en','fr']:
        part=consistency.summarize([row for row in rows if row['language']==language],questions=questions)
        assert part['agreement_labels']==1
        assert part['accepted_errors']==(language=='en')
        assert part['rejected_errors']==(language=='en')
