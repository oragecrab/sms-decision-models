"""Encoder competitors: GLiNER classification and local reference CLM inference."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .classifier import MessageTooLongError
from .questions import QUESTIONS


def snapshot(model: str, revision: str | None, patterns: list[str]) -> str:
    if Path(model).is_dir():
        if revision is not None:
            raise ValueError("--revision cannot be used with a local checkpoint")
        return str(Path(model).resolve())
    from huggingface_hub import snapshot_download
    return snapshot_download(model, revision=revision, allow_patterns=patterns)


def select_device(device: str) -> str:
    import torch
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


class GLiNERBackend:
    def __init__(self, model: str, *, device: str = "auto", revision: str | None = None):
        try:
            from gliner2 import AutoExtractor
            from gliner2.classification import ClassificationSchema, compile_schema
            from gliner2.classification.scoring import ClassificationScorer
        except ImportError as error:
            raise RuntimeError("GLiNER needs its separate environment: uv run --extra gliner system-one-models ...") from error
        path = snapshot(model, revision, ["*.json", "*.safetensors", "*.bin", "*.model", "*.txt"])
        selected = select_device(device)
        self.model = AutoExtractor.from_pretrained(path).to(selected).eval()
        self.scorer = ClassificationScorer(self.model, device=selected).eval()
        schema = ClassificationSchema()
        for key, question in QUESTIONS.items():
            schema.single(key, question["criteria"], instruction=question["instructions"], activation="softmax")
        self.compiled = compile_schema(schema)
        self.metadata = {"backend": "gliner", "model": model, "revision": revision,
                         "resolved_snapshot": path, "device": selected,
                         "requested_device": device, "schema_mode": "all_heads_one_pass",
                         "temperature": 1.0, "input_token_limit": 4096,
                         "runtime_commit": "55656fbfa01d3d4a77485e1a1eeeaf682990ccdf"}

    def predict(self, state: dict[str, Any], *, questions: dict | None = None) -> dict[str, Any]:
        questions = QUESTIONS if questions is None else questions
        compiled = self.compiled
        if questions is not QUESTIONS:
            from gliner2.classification import ClassificationSchema, compile_schema
            schema = ClassificationSchema()
            for key, question in questions.items():
                schema.single(key, question["criteria"], instruction=question["instructions"], activation="softmax")
            compiled = compile_schema(schema)
        text = json.dumps(state, ensure_ascii=False)
        batch = self.model.processor.collate_fn_inference([(text, compiled.build())], max_len=None, error_policy="raise")
        if int(batch.attention_mask.sum().item()) > 4096:
            raise MessageTooLongError("GLiNER complete input exceeds the adapter's 4096-token limit. No prediction was made.")
        scores = self.scorer.score(text, compiled, max_len=None)
        answers = {}
        for key, question in questions.items():
            probabilities = {label: scores.probability(key, label) for label in question["criteria"]}
            if question["type"] == "noul":
                answers[key] = {"noul": probabilities["true"]}
            else:
                answers[key] = {"choice": max(probabilities, key=probabilities.get), "probabilities": probabilities}
        return answers


class LocalCLMEmbedder:
    """Qwen3-8B raw-text embeddings, last real token, L2 normalized.

    The reference serving engine sends strings without a chat template. Cache
    entries are keyed by exact text; full inputs are refused rather than truncated.
    """
    def __init__(self, path: str, device: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(
            path, trust_remote_code=False,
            dtype=torch.float32 if device == "cpu" else torch.float16,
            attn_implementation="sdpa",
        ).to(device).eval()
        self.device = device
        self.cache: OrderedDict[str, Any] = OrderedDict()

    def encode(self, text: str) -> list[int]:
        ids = self.tok.encode(text, add_special_tokens=True)
        if not ids:
            ids = self.tok.encode(" ", add_special_tokens=True)
        if len(ids) > 2048:
            raise MessageTooLongError("CLM text exceeds 2048 tokens. No prediction was made.")
        return ids

    def embed(self, texts: list[str]):
        import numpy as np
        import torch
        # Validate every input before running the first encoder pass.
        tokens = {text: self.encode(text) for text in dict.fromkeys(texts)}
        spent = 0
        with torch.inference_mode():
            for text, ids in tokens.items():
                if text not in self.cache:
                    tensor = torch.tensor([ids], device=self.device)
                    hidden = self.model(input_ids=tensor, attention_mask=torch.ones_like(tensor), use_cache=False).last_hidden_state
                    last = hidden[0, -1].float()
                    normalized = last / (torch.linalg.vector_norm(last) + 1e-12)
                    self.cache[text] = normalized.cpu().numpy()
                    spent += len(ids)
                self.cache.move_to_end(text)
            result = np.stack([self.cache[text] for text in texts])
            while len(self.cache) > 4096:
                self.cache.popitem(last=False)
        return result, spent


class CLMBackend:
    def __init__(self, model: str, *, device: str = "auto", revision: str | None = None):
        from ._clm_reference.heads import HeadPair
        import os
        head_path = snapshot(model, revision, ["CLM_v0.1-8B.pt"])
        checkpoint = Path(head_path) / "CLM_v0.1-8B.pt"
        if not checkpoint.is_file():
            raise ValueError("CLM checkpoint directory must contain CLM_v0.1-8B.pt")
        selected = select_device(device)
        # The released head is trained specifically against Qwen3-8B.
        encoder_revision = os.environ.get("CLM_ENCODER_REVISION")
        encoder_path = snapshot("Qwen/Qwen3-8B", encoder_revision,
                                ["*.json", "*.safetensors", "*.jinja", "*.txt"])
        self.embedder = LocalCLMEmbedder(encoder_path, selected)
        self.head = HeadPair("clm-latest", str(checkpoint), selected).ensure()
        self.metadata = {"backend": "clm", "model": model, "revision": revision,
                         "resolved_snapshot": head_path, "encoder": "Qwen/Qwen3-8B",
                         "encoder_snapshot": encoder_path, "encoder_revision": encoder_revision,
                         "device": selected, "requested_device": device,
                         "runtime": "transformers_last_token_pooling", "input_token_limit": 2048,
                         "encoder_dtype": "float32" if selected == "cpu" else "float16",
                         "attention": "causal", "embedding_normalization": "L2",
                         "embedding_cache": "4096 exact texts; states and candidates",
                         "head_scale": self.head.scale,
                         "reference_code_commit": "bb42c6c5bf914fd449bed2f6ca65be80602cb1f7"}

    def predict(self, state: dict[str, Any], *, questions: dict | None = None) -> dict[str, Any]:
        questions = QUESTIONS if questions is None else questions
        from ._clm_reference.schema import build_pairs, answer_from_logits
        pairs = build_pairs(state, questions)
        states = [p[0] for p in pairs.values()]
        candidates = [text for p in pairs.values() for text in p[2]]
        vectors, _ = self.embedder.embed(states + candidates)
        n = len(states)
        state_vectors = self.head.project_states(vectors[:n])
        action_vectors = self.head.project_actions(vectors[n:])
        answers, offset = {}, 0
        for i, (key, (_, names, texts)) in enumerate(pairs.items()):
            logits = self.head.scale * (action_vectors[offset:offset + len(texts)] @ state_vectors[i])
            answers[key] = answer_from_logits(questions[key], names, logits.tolist())
            offset += len(texts)
        return answers
