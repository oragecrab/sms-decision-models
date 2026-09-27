from types import SimpleNamespace
import math

import pytest

from system_one_models import backends, benchmark
from system_one_models.classifier import MessageTooLongError
from system_one_models.formatting import format_output
from system_one_models.questions import QUESTIONS


def fake_reranker(length=100, bad_score=False):
    events = []
    def preprocess(pairs, **kw):
        assert kw["processing_kwargs"]["text"]["truncation"] is False
        assert kw["processing_kwargs"]["chat_template"]["truncation"] is False
        assert "Message evidence:" in pairs[0][0]
        events.append("preprocess")
        return {"attention_mask": SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: length))}
    def predict(pairs, **kw):
        assert kw["batch_size"] == 1
        assert kw["processing_kwargs"]["text"]["truncation"] is False
        events.append("predict")
        # Highest relevance for the first candidate of each question, including true.
        return [float("nan") if bad_score else (10. if i == 0 or "true:" in pair[1] else -5.) for i, pair in enumerate(pairs)]
    backend = backends.ZerankBackend.__new__(backends.ZerankBackend)
    backend.model = SimpleNamespace(preprocess=preprocess, predict=predict)
    backend.metadata = {"model": "fake-zerank", "probability_metrics_supported": False}
    return backend, events


def test_scores_are_ranked_as_uncalibrated_relative_weights():
    backend, events = fake_reranker()
    result = backend.predict({"body": "message"})
    assert result["classification"]["choice"] == "legitimate"
    assert sum(result["classification"]["probabilities"].values()) == pytest.approx(1)
    assert result["urgent_action"]["noul"] == pytest.approx(1 / (1 + math.exp(-3)))
    assert result["classification"]["raw_scores"]["legitimate"] == 10
    assert events[-1] == "predict"
    assert events.count("preprocess") == sum(len(q["criteria"]) for q in QUESTIONS.values())
    rendered = format_output({"body": "message"}, result, model_name="Zerank")
    assert "relative yes weight" in rendered
    assert "P(yes)" not in rendered


def test_long_pair_rejected_before_inference():
    backend, events = fake_reranker(length=32769)
    with pytest.raises(MessageTooLongError, match="No prediction"):
        backend.predict({"body": "message"})
    assert "predict" not in events
    backend, events = fake_reranker(length=32768)
    backend.predict({"body": "message"})
    assert events[-1] == "predict"


def test_nonfinite_relevance_rejected():
    backend, _ = fake_reranker(bad_score=True)
    with pytest.raises(ValueError, match="invalid relevance"):
        backend.predict({"body": "message"})


def test_report_omits_probability_metrics(monkeypatch, tmp_path, capsys):
    backend, _ = fake_reranker()
    monkeypatch.setattr(benchmark, "create_backend", lambda *a, **kw: backend)
    report = benchmark.evaluate(["fake-zerank"], output=tmp_path / "report.json")
    run = report["runs"][0]
    assert all(m["nll"] is None and m["brier"] is None for m in run["per_question"].values())
    assert all("gold_probabilities" not in row and "brier" not in row for row in run["predictions"])
    assert "n/a" in capsys.readouterr().out


def test_repo_and_explicit_prefix_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(backends, "ZerankBackend", lambda model, **kw: calls.append((model, kw)))
    backends.create_backend("zeroentropy/zerank-2-reranker", revision="sha")
    backends.create_backend("zerank:/local/checkpoint", device="cpu")
    assert calls[0] == ("zeroentropy/zerank-2-reranker", {"device": "auto", "revision": "sha"})
    assert calls[1][0] == "/local/checkpoint"
