from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from system_one_models import benchmark, cli
from system_one_models.question_schemas import load_question_schema, schema_hash
from system_one_models.questions import QUESTIONS
from test_benchmark import answers


def write_schema(tmp_path, questions=None):
    path = tmp_path / 'schema.json'
    path.write_text(json.dumps({'version': 'test-v1', 'questions': questions or deepcopy(QUESTIONS)}))
    return path


def test_hash_preserves_option_order_but_ignores_json_formatting(tmp_path):
    q = deepcopy(QUESTIONS)
    path = write_schema(tmp_path, q)
    first, meta = load_question_schema(path)
    path.write_text(json.dumps(json.loads(path.read_text()), indent=4))
    assert load_question_schema(path)[1]['sha256'] == meta['sha256']
    q['requested_action']['criteria'] = dict(reversed(list(q['requested_action']['criteria'].items())))
    assert schema_hash(q) != schema_hash(first)
    q['urgent_action']['labels'] = {'false': 'B', 'true': 'A'}
    before = schema_hash(q)
    q['urgent_action']['labels'] = {'true': 'A', 'false': 'B'}
    assert schema_hash(q) == before


@pytest.mark.parametrize('mutation', ['id', 'type', 'option', 'empty', 'boolean', 'extra'])
def test_invalid_schema_rejected_before_backend_loading(tmp_path, monkeypatch, mutation):
    q = deepcopy(QUESTIONS)
    if mutation == 'id': q.pop('industry')
    elif mutation == 'type': q['industry']['type'] = 'noul'
    elif mutation == 'option': q['industry']['criteria']['invented'] = 'Other'
    elif mutation == 'empty': q['industry']['instructions'] = ' '
    elif mutation == 'boolean': q['urgent_action']['labels'] = {'true': 'A', 'false': 'A'}
    else: q['industry']['unsupported'] = True
    monkeypatch.setattr(benchmark, 'create_backend', lambda *a, **kw: pytest.fail('backend must not load'))
    with pytest.raises(ValueError):
        benchmark.evaluate(['fake'], question_schema=write_schema(tmp_path, q))


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text('{"version":"one","version":"two","questions":{}}')
    with pytest.raises(ValueError, match='Duplicate'):
        load_question_schema(path)


def test_custom_schema_reaches_warmup_predictions_scoring_and_saved_report(tmp_path, monkeypatch):
    pair = benchmark.load_dataset(benchmark.DEFAULT_DATASET)[:2]
    dataset = tmp_path / 'paired.jsonl'
    dataset.write_text(''.join(json.dumps(row)+'\n' for row in pair))
    q = deepcopy(QUESTIONS)
    q['classification']['instructions'] = 'A deliberate wording variant.'
    q['requested_action']['criteria'] = dict(reversed(list(q['requested_action']['criteria'].items())))
    schema = write_schema(tmp_path, q)
    calls = []
    def predict(state, *, questions):
        calls.append(questions)
        return answers(pair[0]['expected'])
    monkeypatch.setattr(benchmark, 'create_backend', lambda *a, **kw: SimpleNamespace(predict=predict, metadata={'model': 'fake'}))
    output = tmp_path / 'report.json'
    report = benchmark.evaluate(['fake'], dataset=dataset, output=output, question_schema=schema)
    assert calls == [q, q, q]
    assert QUESTIONS['classification']['instructions'] != q['classification']['instructions']
    assert report['questions'] == q
    assert report['model_facing_schemas']['en']['questions'] == q
    assert report['question_schema']['sha256'] == schema_hash(q)
    assert report['question_schema']['version'] == 'test-v1'
    assert report['runs'][0]['per_language']['fr']['label_accuracy'] == 1
    assert report['runs'][0]['translation_pairs']['same_labels'] == 8
    assert json.loads(output.read_text()) == report


def test_helpers_use_supplied_schema_instead_of_global_questions():
    q = {'urgent_action': QUESTIONS['urgent_action']}
    score = benchmark.score_answer({'urgent_action': {'noul': .8}}, {'urgent_action': True}, questions=q)
    row = {'language': 'en', **score}
    assert benchmark.summarize_languages([row], questions=q)['en']['label_accuracy'] == 1
    assert set(score['matches']) == {'urgent_action'}


@pytest.mark.parametrize('flags', [[], ['--eval', '--question-language', 'match'], ['--eval', '--question-language', 'fr'], ['--demo']])
def test_cli_rejects_unsupported_schema_modes(monkeypatch, flags):
    monkeypatch.setattr('sys.argv', ['system-one-models', *flags, '--question-schema', 'variant.json'])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_cli_dispatches_schema(monkeypatch):
    calls = []
    monkeypatch.setattr(benchmark, 'create_backend', lambda *a, **kw: pytest.fail('no inference'))
    monkeypatch.setattr(cli, 'evaluate', lambda models, **kw: calls.append(kw))
    monkeypatch.setattr('sys.argv', ['system-one-models', '--eval', '--question-schema', 'variant.json'])
    cli.main()
    assert calls[0]['question_schema'] == Path('variant.json')


def test_rubric_variant_and_reviewed_dataset_are_valid():
    root = Path(__file__).resolve().parents[1]
    q, meta = load_question_schema(root / 'schemas/rubric-v1.json')
    assert meta['version'] == 'rubric-en-v1'
    rows = benchmark.load_dataset(root / 'datasets/base_eval_v4.jsonl', questions=q)
    assert len(rows) == 44
    assert list(q['requested_action']['criteria'])[0] == 'install_or_grant_access'
