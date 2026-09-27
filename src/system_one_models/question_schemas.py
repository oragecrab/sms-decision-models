"""Load reproducible question variants with stable public answer keys."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .questions import QUESTIONS


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate schema key: {key}")
        result[key] = value
    return result


def validate_questions(questions: dict) -> None:
    if not isinstance(questions, dict) or set(questions) != set(QUESTIONS):
        raise ValueError("Schema must preserve every canonical question ID")
    for key, spec in questions.items():
        original = QUESTIONS[key]
        if not isinstance(spec, dict) or set(spec) - {"type", "instructions", "criteria", "labels"}:
            raise ValueError(f"Invalid question definition for {key}")
        if spec.get("type") != original["type"]:
            raise ValueError(f"Schema must preserve the question type for {key}")
        if not isinstance(spec.get("instructions"), str) or not spec["instructions"].strip():
            raise ValueError(f"Schema needs nonempty instructions for {key}")
        criteria = spec.get("criteria")
        if not isinstance(criteria, dict) or set(criteria) != set(original["criteria"]):
            raise ValueError(f"Schema must preserve canonical labels for {key}")
        if any(not isinstance(value, str) or not value.strip() for value in criteria.values()):
            raise ValueError(f"Schema needs nonempty criteria descriptions for {key}")
        if spec["type"] == "noul":
            labels = spec.get("labels")
            if not isinstance(labels, dict) or set(labels) != {"true", "false"}:
                raise ValueError(f"Schema needs true/false model-facing labels for {key}")
            if any(not isinstance(value, str) or not value.strip() for value in labels.values()) or len(set(labels.values())) != 2:
                raise ValueError(f"Schema needs distinct nonempty boolean labels for {key}")
        elif "labels" in spec:
            raise ValueError(f"Choice question {key} must use canonical criterion labels")


def schema_hash(questions: dict) -> str:
    # Encode ordered mappings as lists: option order is part of the experiment.
    # JSON whitespace and property order inside a definition are not.
    ordered = [[key, spec["type"], spec["instructions"], list(spec["criteria"].items()),
                sorted(spec.get("labels", {}).items())] for key, spec in questions.items()]
    payload = json.dumps(ordered, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_question_schema(path: Path | None = None) -> tuple[dict, dict]:
    if path is None:
        questions = deepcopy(QUESTIONS)
        metadata = {"version": "baseline-en-v1", "source": "built-in"}
    else:
        document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        if not isinstance(document, dict) or set(document) != {"version", "questions"}:
            raise ValueError("Schema file must contain version and questions")
        if not isinstance(document["version"], str) or not document["version"].strip():
            raise ValueError("Schema version must be a nonempty string")
        questions = document["questions"]
        metadata = {"version": document["version"], "source": str(path.resolve())}
    validate_questions(questions)
    return questions, {**metadata, "sha256": schema_hash(questions)}
