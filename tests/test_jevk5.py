import json
import sys
from types import SimpleNamespace

import pytest

from system_one_models import backends
from system_one_models.benchmark import DEFAULT_DATASET, load_dataset, score_answer
from system_one_models.classifier import MessageTooLongError
from system_one_models.questions import QUESTIONS


def test_routes_repo_and_local_checkpoint(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(backends, "JevK5Backend", lambda model, **kw: calls.append((model, kw)))
    backends.create_backend("alibiserikbay/JevK5", revision="v0.2", device="cpu")
    backends.create_backend("jevk5:someone/custom")
    (tmp_path / "jevk5_config.json").write_text("{}")
    backends.create_backend(str(tmp_path))
    assert [c[0] for c in calls] == ["alibiserikbay/JevK5", "someone/custom", str(tmp_path)]
    assert calls[0][1] == {"device": "cpu", "revision": "v0.2"}


def test_native_constructor_keeps_calibration_and_disables_graphs(monkeypatch, tmp_path):
    (tmp_path / "jevk5_config.json").write_text(json.dumps({"temperature": 1.22}))
    calls = []
    def runtime(path, **kw):
        calls.append((path, kw))
        return SimpleNamespace(temperature=1.22, knockout_temperature=.93)
    monkeypatch.setitem(sys.modules, "jevk5", SimpleNamespace(JevK5=runtime))
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = backends.JevK5Backend(str(tmp_path))
    assert calls[0][1] == {"device": "cpu", "dtype": torch.float32, "graphs": False}
    assert model.metadata["temperature"] == 1.22
    assert model.metadata["device"] == "cpu"
    with pytest.raises(ValueError, match="local model"):
        backends.JevK5Backend(str(tmp_path), revision="v0.2")


def test_mps_is_rejected_before_loading_weights():
    with pytest.raises(ValueError, match="--device cpu"):
        backends.JevK5Backend("alibiserikbay/JevK5", device="mps")


def test_preflight_all_questions_before_any_inference(monkeypatch):
    monkeypatch.setitem(sys.modules, "jevk5.prompt", SimpleNamespace(
        decision_options=lambda question: list(question["criteria"].items())))
    events = []
    def encode(state, instructions, options):
        events.append("encode")
        return range(state["tokens"])
    def decide(state, question):
        events.append("decide")
        return {"noul": .75}
    model = backends.JevK5Backend.__new__(backends.JevK5Backend)
    model.model = SimpleNamespace(encode=encode, decide=decide)
    assert len(model.predict({"tokens": 16384})) == len(QUESTIONS)
    assert events == ["encode"] * len(QUESTIONS) + ["decide"] * len(QUESTIONS)
    events.clear()
    with pytest.raises(MessageTooLongError, match="No prediction"):
        model.predict({"tokens": 16385})
    assert "decide" not in events


def test_installed_runtime_answer_contract_and_boolean_polarity():
    pytest.importorskip("jevk5")
    from jevk5.prompt import decision_options, answer
    expected = load_dataset(DEFAULT_DATASET)[0]["expected"]
    results = {}
    for key, question in QUESTIONS.items():
        options = decision_options(question)
        gold = str(expected[key]).lower() if question["type"] == "noul" else expected[key]
        probs = {name: .8 if name == gold else .2 / (len(options) - 1) for name, text in options}
        results[key] = answer(question, probs, 100)
    assert score_answer(results, expected)["exact_match"]
