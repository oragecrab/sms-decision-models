"""Local model adapters sharing the project's typed answer contract."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .classifier import MessageTooLongError, classify
from .questions import QUESTIONS


class LayaBackend:
    def __init__(self, *, device: str = "auto", revision: str | None = None):
        if revision is not None:
            raise ValueError("--revision applies to Decider checkpoints, not Laya")
        from laya import Router
        self.router = Router(device=None if device == "auto" else device)
        self.metadata = {"backend": "laya", "model": "laya", "requested_device": device}

    def predict(self, state: dict[str, Any], *, questions: dict | None = None) -> dict[str, Any]:
        questions = QUESTIONS if questions is None else questions
        result = classify(state, self.router, questions=questions, return_details=True)
        self.last_routing = result.get("routing")
        answer = result["answers"]
        self.metadata["devices"] = {
            name: str(self.router.load(name).device) for name in self.router.loaded
        }
        return answer


class DeciderBackend:
    def __init__(self, model: str, *, device: str = "auto", revision: str | None = None):
        try:
            from decider.infer import Decider
        except ImportError as error:
            raise RuntimeError("Install Decider support with: uv sync --extra decider") from error
        from huggingface_hub import snapshot_download
        local = Path(model).is_dir()
        if local and revision is not None:
            raise ValueError("--revision cannot be used with a local model directory")
        path = str(Path(model).resolve()) if local else snapshot_download(
            model, revision=revision,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"],
        )
        self.model = Decider(path, device=None if device == "auto" else device, use_graphs=False)
        self.metadata = {
            "backend": "decider", "model": model, "revision": revision,
            "resolved_snapshot": path, "requested_device": device,
            "device": str(self.model.dev), "name": self.model.name,
        }

    def predict(self, state: dict[str, Any], *, questions: dict | None = None) -> dict[str, Any]:
        questions = QUESTIONS if questions is None else questions
        from decider.systemone import render_state
        # Decider truncates context to max_state_tokens; reject before it can do so.
        size = len(self.model.m.tok.encode("Context:\n" + render_state(state), add_special_tokens=False))
        if size > 32768:
            raise MessageTooLongError(
                f"Message needs {size} tokens; Decider allows 32768. No prediction was made."
            )
        return self.model.system_one(
            state, questions, max_state_tokens=32768, layout="state_first"
        )["answers"]


class JevK5Backend:
    """Official JevK5 prompt and temperatures, one decision per question."""
    def __init__(self, model: str, *, device: str = "auto", revision: str | None = None):
        if device == "mps":
            raise ValueError("JevK5's native adapter supports CUDA or eager CPU; use --device cpu on this Mac")
        try:
            from jevk5 import JevK5
        except ImportError as error:
            raise RuntimeError("Install JevK5 support with: uv sync --extra jevk5") from error
        import torch
        from huggingface_hub import snapshot_download
        selected = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if selected == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; select --device cpu for eager inference")
        local = Path(model).is_dir()
        if local and revision is not None:
            raise ValueError("--revision cannot be used with a local model directory")
        path = str(Path(model).resolve()) if local else snapshot_download(
            model, revision=revision,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"],
        )
        if not (Path(path) / "jevk5_config.json").is_file():
            raise ValueError("JevK5 checkpoint must contain jevk5_config.json with its calibration settings")
        dtype = torch.bfloat16 if selected == "cuda" else torch.float32
        self.model = JevK5(path, device=selected, dtype=dtype, graphs=False)
        self.metadata = {
            "backend": "jevk5", "model": model, "revision": revision,
            "resolved_snapshot": path, "requested_device": device,
            "device": selected, "dtype": str(dtype), "graphs": False,
            "temperature": self.model.temperature,
            "knockout_temperature": self.model.knockout_temperature,
        }

    def predict(self, state: dict[str, Any], *, questions: dict | None = None) -> dict[str, Any]:
        questions = QUESTIONS if questions is None else questions
        from jevk5.prompt import decision_options
        # Check all prompts before making any decision; never truncate evidence.
        for key, question in questions.items():
            options = [text for _, text in decision_options(question)]
            if len(options) > 16:
                raise ValueError("This adapter's prompt preflight supports at most 16 options per question")
            tokens = len(self.model.encode(state, question["instructions"], options))
            if tokens > 16384:
                raise MessageTooLongError(
                    f"JevK5 question {key!r} needs {tokens} input tokens; limit is 16384. No prediction was made."
                )
        return {key: self.model.decide(state, question) for key, question in questions.items()}


class ZerankBackend:
    """Experimental candidate-label ranking using the official CrossEncoder."""
    PROCESSING = {"text": {"truncation": False}, "chat_template": {"truncation": False}}

    def __init__(self, model: str, *, device: str = "auto", revision: str | None = None):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RuntimeError("Install Zerank support with: uv sync --extra zerank") from error
        import torch
        from huggingface_hub import snapshot_download
        local = Path(model).is_dir()
        if local and revision is not None:
            raise ValueError("--revision cannot be used with a local model directory")
        path = str(Path(model).resolve()) if local else snapshot_download(
            model, revision=revision,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"],
        )
        selected = device
        if device == "auto":
            selected = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.bfloat16 if selected == "cuda" else (torch.float16 if selected == "mps" else torch.float32)
        self.model = CrossEncoder(path, device=selected, max_length=32768,
                                  activation_fn=torch.nn.Identity(), trust_remote_code=False,
                                  model_kwargs={"dtype": dtype})
        self.metadata = {
            "backend": "zerank", "model": model, "revision": revision,
            "resolved_snapshot": path, "requested_device": device,
            "device": str(self.model.device), "dtype": str(dtype),
            "task_adaptation": "question_and_evidence_query_vs_label_description",
            "probability_metrics_supported": False,
            "score_semantics": "uncalibrated relative weights: softmax(raw relevance logits / 5)",
        }

    def predict(self, state: dict[str, Any], *, questions: dict | None = None) -> dict[str, Any]:
        questions = QUESTIONS if questions is None else questions
        pairs, groups = [], []
        evidence = json.dumps(state, ensure_ascii=False)
        for key, question in questions.items():
            query = f"{question['instructions']}\n\nMessage evidence:\n{evidence}\n\nWhich candidate answer best fits this message?"
            names = list(question["criteria"])
            start = len(pairs)
            pairs.extend((query, f"{name}: {question['criteria'][name]}") for name in names)
            groups.append((key, question, names, start))
        # Use the same official preprocessing as predict, with truncation disabled.
        for pair in pairs:
            features = self.model.preprocess([pair], processing_kwargs=self.PROCESSING)
            size = int(features["attention_mask"].sum().item())
            if size > 32768:
                raise MessageTooLongError(
                    f"Zerank pair needs {size} tokens; limit is 32768. No prediction was made."
                )
        scores = self.model.predict(pairs, batch_size=1, show_progress_bar=False,
                                    processing_kwargs=self.PROCESSING)
        scores = [float(score) for score in scores]
        if len(scores) != len(pairs) or any(not math.isfinite(score) for score in scores):
            raise ValueError("Zerank returned invalid relevance scores")
        answers = {}
        for key, question, names, start in groups:
            raw = scores[start:start + len(names)]
            peak = max(raw)
            weights = [math.exp((score - peak) / 5) for score in raw]
            total = sum(weights)
            probabilities = {name: weight / total for name, weight in zip(names, weights)}
            row = {"raw_scores": dict(zip(names, raw)), "score_semantics": "uncalibrated_relative_weights"}
            if question["type"] == "noul":
                row["noul"] = probabilities["true"]
            else:
                row.update(choice=max(probabilities, key=probabilities.get), probabilities=probabilities)
            answers[key] = row
        return answers


def create_backend(model: str, *, device: str = "auto", revision: str | None = None):
    if model == "clm" or model == "Contrastive-LM/CLM-v0.1-8B" or model.startswith("clm:"):
        from .encoder_backends import CLMBackend
        source = "Contrastive-LM/CLM-v0.1-8B" if model == "clm" else model.removeprefix("clm:")
        return CLMBackend(source, device=device, revision=revision)
    if model == "fastino/GLiNER2.5-Decide" or model.startswith("gliner:"):
        from .encoder_backends import GLiNERBackend
        return GLiNERBackend(model.removeprefix("gliner:"), device=device, revision=revision)
    if model == "laya":
        return LayaBackend(device=device, revision=revision)
    if model.startswith("zerank:"):
        return ZerankBackend(model.removeprefix("zerank:"), device=device, revision=revision)
    if model in {"zeroentropy/zerank-2-reranker", "zeroentropy/zerank-2"}:
        return ZerankBackend(model, device=device, revision=revision)
    if model.startswith("jevk5:"):
        return JevK5Backend(model.removeprefix("jevk5:"), device=device, revision=revision)
    if model in {"alibiserikbay/JevK5", "alibiserikbay/JevK5-2B", "alibiserikbay/JevK5-9B"}:
        return JevK5Backend(model, device=device, revision=revision)
    if Path(model).is_dir() and (Path(model) / "jevk5_config.json").is_file():
        return JevK5Backend(model, device=device, revision=revision)
    return DeciderBackend(model, device=device, revision=revision)
