from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from system_one_models import encoder_backends, backends
from system_one_models.classifier import MessageTooLongError
from system_one_models.questions import QUESTIONS
from system_one_models.benchmark import score_answer, load_dataset, DEFAULT_DATASET


def test_dispatches_competitors(monkeypatch):
    calls = []
    monkeypatch.setattr(encoder_backends, "CLMBackend", lambda model, **kw: calls.append(("clm", model, kw)))
    monkeypatch.setattr(encoder_backends, "GLiNERBackend", lambda model, **kw: calls.append(("gliner", model, kw)))
    backends.create_backend("clm")
    backends.create_backend("gliner:/local/decide", device="cpu", revision=None)
    assert calls[0][1] == "Contrastive-LM/CLM-v0.1-8B"
    assert calls[1] == ("gliner", "/local/decide", {"device": "cpu", "revision": None})


def test_clm_last_token_normalization_cache_and_full_input_validation():
    calls = []
    embedder = encoder_backends.LocalCLMEmbedder.__new__(encoder_backends.LocalCLMEmbedder)
    embedder.device = "cpu"
    embedder.cache = OrderedDict()
    embedder.tok = SimpleNamespace(encode=lambda text, **kw: [3, 4] if text == "short" else list(range(2049)))
    def model(**kw):
        calls.append(kw)
        return SimpleNamespace(last_hidden_state=torch.tensor([[[100., 0.], [3., 4.]]]))
    embedder.model = model
    vectors, spent = embedder.embed(["short", "short"])
    assert vectors.shape == (2, 2)
    assert np.allclose(vectors[0], [.6, .8])
    assert spent == 2 and len(calls) == 1
    assert calls[0]["use_cache"] is False
    _, spent = embedder.embed(["short"])
    assert spent == 0 and len(calls) == 1
    with pytest.raises(MessageTooLongError, match="No prediction"):
        embedder.embed(["short", "long"])
    assert len(calls) == 1


def test_clm_reference_head_readout_and_boolean_contract(tmp_path):
    from system_one_models._clm_reference.heads import HeadPair, make_head
    from system_one_models._clm_reference.schema import build_pairs, answer_from_logits
    state = {"channel": "sms", "body": "Hello"}
    pairs = build_pairs(state, QUESTIONS)
    assert "channel: sms" in pairs["industry"][0]
    assert pairs["industry"][0].endswith(QUESTIONS["industry"]["instructions"])
    assert pairs["urgent_action"][1] == ["false", "true"]
    result = answer_from_logits(QUESTIONS["urgent_action"], ["false", "true"], [0., 2.])
    assert result["noul"] == pytest.approx(1 / (1 + np.exp(-2)))
    cfg = {"width": 3, "depth": 2, "hidden_size": 2, "projection_dim": 2}
    sh, ah = make_head(3, hidden=2, proj=2), make_head(3, hidden=2, proj=2)
    checkpoint = tmp_path / "head.pt"
    torch.save({"cfg": cfg, "state_head": sh.state_dict(), "action_head": ah.state_dict(), "logit_scale": torch.tensor(2.)}, checkpoint)
    head = HeadPair("test", str(checkpoint), "cpu").ensure()
    vectors = head.project_states(np.array([[.6, .8]], dtype=np.float32))
    assert torch.linalg.vector_norm(vectors).item() == pytest.approx(1.)
    assert head.scale == pytest.approx(np.exp(2))


def test_gliner_maps_all_head_distributions_and_boolean_polarity():
    model = encoder_backends.GLiNERBackend.__new__(encoder_backends.GLiNERBackend)
    expected = load_dataset(DEFAULT_DATASET)[0]["expected"]
    def probability(key, label):
        gold = str(expected[key]).lower() if type(expected[key]) is bool else expected[key]
        return .8 if label == gold else .2 / (len(QUESTIONS[key]["criteria"]) - 1)
    scorer_calls = []
    def score(text, compiled, **kw):
        scorer_calls.append((text, kw))
        return SimpleNamespace(probability=probability)
    model.scorer = SimpleNamespace(score=score)
    model.compiled = SimpleNamespace(build=lambda: {})
    model.model = SimpleNamespace(processor=SimpleNamespace(collate_fn_inference=lambda *a, **kw: SimpleNamespace(attention_mask=torch.ones((1, 100)))))
    result = model.predict({"body": "hello"})
    assert score_answer(result, expected)["exact_match"]
    assert scorer_calls[0][1]["max_len"] is None
    model.model.processor.collate_fn_inference = lambda *a, **kw: SimpleNamespace(attention_mask=torch.ones((1, 4097)))
    with pytest.raises(MessageTooLongError):
        model.predict({"body": "hello"})
    assert len(scorer_calls) == 1


def test_gliner_schema_preserves_instructions_and_descriptions():
    pytest.importorskip("gliner2")
    from gliner2.classification import ClassificationSchema, compile_schema
    schema = ClassificationSchema()
    for key, question in QUESTIONS.items():
        schema.single(key, question["criteria"], instruction=question["instructions"], activation="softmax")
    compiled = compile_schema(schema)
    assert compiled.task_order == tuple(QUESTIONS)
    for task in compiled.task_specs:
        assert task.instruction == QUESTIONS[task.name]["instructions"]
        assert task.activation == "softmax"
