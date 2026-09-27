from copy import deepcopy
from types import SimpleNamespace

import pytest

from system_one_models.questions import QUESTIONS
from system_one_models.french_questions import french_schema, canonical_answers
from system_one_models import benchmark
from test_benchmark import answers


def test_french_schema_preserves_types_order_and_polarity():
    before = deepcopy(QUESTIONS)
    schema, mappings = french_schema()
    assert set(schema) == set(QUESTIONS)
    for key, original in QUESTIONS.items():
        assert schema[key]['type'] == original['type']
        assert schema[key]['instructions'] != original['instructions']
        if original['type'] == 'choice':
            assert list(mappings[key].values()) == list(original['criteria'])
            assert list(mappings[key]) == list(schema[key]['criteria'])
        else:
            assert schema[key]['labels'] == {'true': 'A', 'false': 'B'}
            assert schema[key]['criteria']['true'].startswith('Oui')
            assert schema[key]['criteria']['false'].startswith('Non')
    assert QUESTIONS == before


def test_french_answers_map_to_identical_gold_keys_without_changing_probabilities():
    expected = benchmark.load_dataset(benchmark.DEFAULT_DATASET)[0]['expected']
    canonical = answers(expected)
    raw = deepcopy(canonical)
    _, mappings = french_schema()
    for key, mapping in mappings.items():
        inverse = {canonical: french for french, canonical in mapping.items()}
        raw[key]['choice'] = inverse[raw[key]['choice']]
        raw[key]['probabilities'] = {inverse[label]: p for label, p in raw[key]['probabilities'].items()}
    saved = deepcopy(raw)
    assert canonical_answers(raw, mappings) == canonical
    assert raw == saved


def test_matched_evaluation_uses_french_only_for_french_messages(monkeypatch):
    pair = benchmark.load_dataset(benchmark.DEFAULT_DATASET)[:2]
    monkeypatch.setattr(benchmark, 'load_dataset', lambda path: pair)
    expected = pair[0]['expected']
    calls = []
    schema, mappings = french_schema()
    def predict(state, *, questions=None):
        calls.append((state, questions))
        result = answers(expected)
        if questions is not None:
            for key, mapping in mappings.items():
                inverse = {canonical: french for french, canonical in mapping.items()}
                result[key]['choice'] = inverse[result[key]['choice']]
                result[key]['probabilities'] = {inverse[label]: p for label, p in result[key]['probabilities'].items()}
        return result
    monkeypatch.setattr(benchmark, 'create_backend', lambda *a, **kw: SimpleNamespace(predict=predict, metadata={'model':'fake'}))
    report = benchmark.evaluate(['fake'], question_language='match')
    assert calls[0][1] is None  # English warmup
    assert calls[1][1] is None  # English scored source
    assert calls[2][1] == schema  # French scored translation
    assert report['runs'][0]['per_language']['fr']['label_accuracy'] == 1
    assert report['runs'][0]['translation_pairs']['same_labels'] == 8
    fr = report['runs'][0]['predictions'][1]
    assert fr['raw_answers']['classification']['choice'] != fr['answers']['classification']['choice']
    assert fr['question_language'] == 'fr'


def test_cli_dispatches_question_language(monkeypatch):
    import sys
    from system_one_models import cli
    calls = []
    monkeypatch.setattr(cli, 'evaluate', lambda models, **kw: calls.append(kw))
    monkeypatch.setattr(sys, 'argv', ['system-one-models', '--eval', '--question-language', 'match'])
    cli.main()
    assert calls[0]['question_language'] == 'match'
    monkeypatch.setattr(sys, 'argv', ['system-one-models', '--eval', '--consistency', '--question-language', 'fr'])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
