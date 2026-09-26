"""Load the base evaluation cases and report local model results."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .classifier import classify
from .formatting import COMPACT_LABELS, format_output
from .questions import QUESTIONS


def load_examples() -> list[dict[str, Any]]:
    path = Path(__file__).resolve().parents[2] / "datasets" / "base_eval_v1.jsonl"
    with path.open(encoding="utf-8") as dataset:
        return [json.loads(line) for line in dataset if line.strip()]


def run_demo(*, details: bool = False) -> int:
    from laya import Router

    router = Router()
    examples = load_examples()
    scores = {question_id: 0 for question_id in QUESTIONS}
    exact_matches = 0
    for index, example in enumerate(examples, start=1):
        state = example["state"]
        started = time.perf_counter()
        answer = classify(state, router)
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(
            format_output(
                state,
                answer,
                title=f"{index:02d} · {example['id']}",
                expected=example["expected"],
                elapsed_ms=elapsed_ms,
                details=details,
                compact=True,
            )
        )
        results = {
            question_id: (
                question_answer.get("choice")
                if "choice" in question_answer
                else question_answer.get("noul", 0) >= 0.5
            )
            for question_id, question_answer in answer.items()
        }
        for question_id, expected_value in example["expected"].items():
            scores[question_id] += results.get(question_id) == expected_value
        exact_matches += all(
            results.get(question_id) == expected_value
            for question_id, expected_value in example["expected"].items()
        )
        if index < len(examples):
            print()
    print(
        f"\n{len(examples)}-message base eval · "
        f"exact matches {exact_matches}/{len(examples)}"
    )
    print("  " + "  ".join(
        f"{COMPACT_LABELS[question_id]} {scores[question_id]}/{len(examples)}"
        for question_id in QUESTIONS
    ))
    return 0
