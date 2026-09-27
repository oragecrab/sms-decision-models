from types import SimpleNamespace
import json
import math
import sys

import pytest

from system_one_models import benchmark
from system_one_models.backends import DeciderBackend
from system_one_models.classifier import MessageTooLongError
from system_one_models.questions import QUESTIONS


def answers(expected):
    result = {}
    for key, spec in QUESTIONS.items():
        if spec["type"] == "noul":
            result[key] = {"noul": 0.8 if expected[key] else 0.2}
        else:
            others = len(spec["criteria"]) - 1
            result[key] = {"choice": expected[key], "probabilities": {
                label: 0.8 if label == expected[key] else 0.2 / others
                for label in spec["criteria"]}}
    return result


def test_scoring_probability_metrics_and_boolean_polarity():
    expected = benchmark.load_dataset(benchmark.DEFAULT_DATASET)[0]["expected"]
    result = benchmark.score_answer(answers(expected), expected)
    assert result["exact_match"]
    assert all(p == pytest.approx(0.8) for p in result["gold_probabilities"].values())
    assert result["brier"]["urgent_action"] == pytest.approx(0.08)
    wrong = answers(expected)
    wrong["urgent_action"]["noul"] = 1 - wrong["urgent_action"]["noul"]
    result = benchmark.score_answer(wrong, expected)
    assert not result["exact_match"]
    assert result["gold_probabilities"]["urgent_action"] == pytest.approx(0.2)


def test_evaluation_saves_same_inputs_and_separate_warmup(monkeypatch, tmp_path):
    examples = benchmark.load_dataset(benchmark.DEFAULT_DATASET)
    expected = {json.dumps(e["state"], sort_keys=True): e["expected"] for e in examples}
    calls = []
    def factory(model, **kwargs):
        def predict(state):
            calls.append((model, state))
            return answers(expected[json.dumps(state, sort_keys=True)])
        return SimpleNamespace(predict=predict, metadata={"model": model})
    monkeypatch.setattr(benchmark, "create_backend", factory)
    output = tmp_path / "report.json"
    report = benchmark.evaluate(["laya", "fake-decider"], output=output)
    assert json.loads(output.read_text()) == report
    assert len(calls) == 2 * (len(examples) + 1)
    for run in report["runs"]:
        assert set(run["per_language"]) == {"en", "fr"}
        assert run["per_language"]["fr"]["examples"] == 22
        assert run["per_language"]["fr"]["label_accuracy"] == 1
        assert run["examples"] == len(examples)
        assert run["exact_match_accuracy"] == 1
        assert run["per_question"]["classification"]["nll"] == pytest.approx(-math.log(.8))
    assert [state for model, state in calls if model == "laya"] == [state for model, state in calls if model == "fake-decider"]


def test_dataset_rejects_duplicate_and_missing_labels(tmp_path):
    example = benchmark.load_dataset(benchmark.DEFAULT_DATASET)[0]
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(example) + "\n" + json.dumps(example))
    with pytest.raises(ValueError, match="Duplicate"):
        benchmark.load_dataset(path)
    example["expected"].pop("industry")
    path.write_text(json.dumps(example))
    with pytest.raises(ValueError, match="every question"):
        benchmark.load_dataset(path)


def test_decider_uses_typed_schema_and_rejects_truncated_context(monkeypatch):
    monkeypatch.setitem(sys.modules, "decider.systemone", SimpleNamespace(render_state=lambda state: state["body"]))
    calls = []
    def system_one(state, questions, **kw):
        calls.append((state, questions, kw))
        return {"answers": {"urgent_action": {"noul": .25}}}
    backend = DeciderBackend.__new__(DeciderBackend)
    backend.model = SimpleNamespace(m=SimpleNamespace(tok=SimpleNamespace(encode=lambda text, **kw: list(range(len(text))))), system_one=system_one)
    assert backend.predict({"body": "hello"})["urgent_action"]["noul"] == .25
    assert calls[0][1] is QUESTIONS
    assert calls[0][2]["layout"] == "state_first"
    with pytest.raises(MessageTooLongError):
        backend.predict({"body": "x" * 32769})
    assert len(calls) == 1


def test_cli_repeated_models_dispatches_evaluation(monkeypatch):
    from system_one_models import cli
    calls = []
    monkeypatch.setattr(cli, "evaluate", lambda models, **kw: calls.append((models, kw)))
    monkeypatch.setattr(sys, "argv", ["system-one-models", "--eval", "--model", "laya", "--model", "Mapika/decider-4b", "--revision", "v1"])
    cli.main()
    assert calls[0][0] == ["laya", "Mapika/decider-4b"]
    assert calls[0][1]["revision"] == "v1"


def test_installed_decider_schema_preserves_labels_and_boolean_polarity():
    pytest.importorskip("decider")
    from decider.systemone import render_question, plan_rows, assemble
    expected = benchmark.load_dataset(benchmark.DEFAULT_DATASET)[0]["expected"]
    rendered = {key: render_question(spec) for key, spec in QUESTIONS.items()}
    rows, index = plan_rows(rendered, False)
    probabilities = []
    for key, kind, start, count in index:
        names = rendered[key]["names"]
        gold = expected[key]
        probabilities.append([.8 if name == gold else .2 / (len(names)-1) for name in names])
    answer = assemble(rendered, index, probabilities)
    assert benchmark.score_answer(answer, expected)["exact_match"]


def test_bilingual_dataset_preserves_english_and_balances_languages():
    from collections import Counter
    original = benchmark.load_dataset(benchmark.DEFAULT_DATASET.with_name('base_eval_v1.jsonl'))
    expanded = benchmark.load_dataset(benchmark.DEFAULT_DATASET.with_name("base_eval_v2.jsonl"))
    assert len(expanded) == 22
    assert Counter(row['language'] for row in expanded) == {'en': 11, 'fr': 11}
    assert [{key: value for key, value in row.items() if key != 'language'}
            for row in expanded if row['language'] == 'en'] == original
    assert len({row['state']['body'] for row in expanded}) == 22
    for language in ('en', 'fr'):
        assert Counter(row['expected']['classification'] for row in expanded if row['language'] == language) == {
            'suspected_scam': 6, 'legitimate': 4, 'marketing': 1}


def test_french_dataset_uses_multilingual_router_without_gold_language_hint():
    from laya import Router
    router = Router()
    for row in benchmark.load_dataset(benchmark.DEFAULT_DATASET):
        route = router.route(row['state'], QUESTIONS)
        assert route['model'] == ('multilingual' if row['language'] == 'fr' else 'english')


def test_language_metrics_keep_mixed_and_legacy_results_separate():
    rows = []
    for language, correct in [('en', True), ('fr', False), ('unspecified', True)]:
        rows.append({'language': language, 'matches': {key: correct for key in QUESTIONS}, 'exact_match': correct})
    result = benchmark.summarize_languages(rows)
    assert result['en']['label_accuracy'] == 1
    assert result['fr']['label_accuracy'] == 0
    assert result['fr']['examples'] == 1
    assert result['unspecified']['exact_match_accuracy'] == 1


def test_dataset_rejects_invalid_language_annotation(tmp_path):
    row = benchmark.load_dataset(benchmark.DEFAULT_DATASET)[0]
    row['language'] = 'french'
    path = tmp_path / 'invalid-language.jsonl'
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='Invalid language'):
        benchmark.load_dataset(path)


def test_translation_pairs_preserve_sources_labels_and_message_structure():
    previous = benchmark.load_dataset(benchmark.DEFAULT_DATASET.with_name('base_eval_v2.jsonl'))
    paired = benchmark.load_dataset(benchmark.DEFAULT_DATASET)
    assert len(paired) == 44
    sources = {row['id']: row for row in paired if not row['is_translation']}
    assert len(sources) == len(previous) == 22
    for original in previous:
        source = sources[original['id']]
        assert source['state'] == original['state']
        assert source['expected'] == original['expected']
        pair = [row for row in paired if row['pair_id'] == source['pair_id']]
        assert len(pair) == 2
        assert {row['language'] for row in pair} == {'en', 'fr'}
        translated = next(row for row in pair if row['is_translation'])
        assert translated['expected'] == source['expected']
        assert translated['source_language'] == source['language']
        assert set(translated['state']) == set(source['state'])
        assert translated['state']['channel'] == source['state']['channel']


def test_dataset_rejects_incomplete_or_relabeled_translation_pair(tmp_path):
    pair = benchmark.load_dataset(benchmark.DEFAULT_DATASET)[:2]
    path = tmp_path / 'pairs.jsonl'
    path.write_text(json.dumps(pair[0]) + '\n')
    with pytest.raises(ValueError, match='one English and one French'):
        benchmark.load_dataset(path)
    pair[1]['expected']['urgent_action'] = not pair[0]['expected']['urgent_action']
    path.write_text(''.join(json.dumps(row) + '\n' for row in pair))
    with pytest.raises(ValueError, match='identical expected'):
        benchmark.load_dataset(path)


def test_pair_metrics_distinguish_translation_changes_from_correctness():
    pair = benchmark.load_dataset(benchmark.DEFAULT_DATASET)[:2]
    rows = []
    for example in pair:
        answer = answers(example['expected'])
        if example['language'] == 'fr':
            answer['urgent_action']['noul'] = 1 - answer['urgent_action']['noul']
        rows.append({**example, 'answers': answer, **benchmark.score_answer(answer, example['expected'])})
    summary = benchmark.summarize_pairs(rows)
    assert summary['pair_count'] == 1
    assert summary['same_labels'] == 7
    assert summary['all_answers_agree_pairs'] == 0
    assert summary['per_question']['urgent_action']['english_correct_french_wrong'] == 1
    assert summary['pairs'][0]['different_questions'] == ['urgent_action']
    assert summary['by_source_language']['en']['english_correct'] == 8
    assert summary['by_source_language']['en']['french_correct'] == 7
