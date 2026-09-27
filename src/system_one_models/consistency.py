"""A separate negative request checks the nominated positive answer."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import statistics
from importlib.metadata import PackageNotFoundError, version
import time
from datetime import datetime, timezone
from pathlib import Path

from .backends import create_backend
from .benchmark import DEFAULT_DATASET, load_dataset, score_answer, summarize_languages
from .questions import QUESTIONS
from .question_schemas import load_question_schema, schema_hash


def negative_questions(positive: dict, *, questions: dict | None = None) -> dict:
    schema = QUESTIONS if questions is None else questions
    negative = {}
    for key, original in schema.items():
        if original["type"] == "choice":
            label = positive[key]["choice"]
            description = original["criteria"][label]
            instructions = (
                f"Original decision rule: {original['instructions']} "
                f"Candidate answer: {label}. Definition: {description} "
                "Is this candidate NOT an appropriate best answer under that rule for the message?"
            )
            criteria = {"true": "Yes, the candidate is not an appropriate best answer.",
                        "false": "No, the candidate is an appropriate best answer."}
        else:
            instructions = (
                "Apply the original question's rubric. Answer its complementary question: "
                "is the condition in the original question FALSE for this message? "
                f"Original question: {original['instructions']}"
            )
            criteria = {"true": original["criteria"]["false"],
                        "false": original["criteria"]["true"]}
        negative[key] = {"type": "noul", "instructions": instructions, "criteria": criteria,
                          "labels": dict(original.get("labels", {"true": "A", "false": "B"}))}
    return negative


def agreement(positive: dict, negative: dict, *, questions: dict | None = None) -> dict[str, bool]:
    questions = QUESTIONS if questions is None else questions
    if set(positive) != set(questions) or set(negative) != set(questions):
        raise ValueError("Positive and negative answers must contain every selected question")
    accepted = {}
    for key, question in questions.items():
        probability = float(negative[key]["noul"])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"Invalid negative probability for {key}")
        if probability == 0.5:
            accepted[key] = False
        elif question["type"] == "choice":
            accepted[key] = probability < 0.5
        else:
            yes = float(positive[key]["noul"])
            if not math.isfinite(yes) or not 0 <= yes <= 1:
                raise ValueError(f"Invalid positive probability for {key}")
            accepted[key] = yes != 0.5 and ((yes > 0.5) != (probability > 0.5))
    return accepted


def summarize(rows: list[dict], *, questions: dict | None = None) -> dict:
    questions = QUESTIONS if questions is None else questions
    if not rows:
        raise ValueError("Consistency summary requires at least one example")
    summary = {}
    for key in questions:
        accepted = [row for row in rows if row["accepted"][key]]
        correct = sum(row["matches"][key] for row in accepted)
        baseline_correct = sum(row["matches"][key] for row in rows)
        summary[key] = {"agreed": len(accepted), "disagreed": len(rows) - len(accepted),
                        "agreement_rate": len(accepted) / len(rows),
                        "rejected": len(rows) - len(accepted),
                        "accepted_errors": len(accepted) - correct,
                        "rejected_errors": len(rows) - baseline_correct - (len(accepted) - correct),
                        "total": len(rows), "accepted": len(accepted),
                        "coverage": len(accepted) / len(rows),
                        "accepted_correct": correct,
                        "selective_accuracy": correct / len(accepted) if accepted else None,
                        "baseline_accuracy": sum(row["matches"][key] for row in rows) / len(rows)}
    n = len(rows) * len(questions)
    count = sum(item["accepted"] for item in summary.values())
    correct = sum(item["accepted_correct"] for item in summary.values())
    complete = [row for row in rows if all(row["accepted"].values())]
    return {"per_question": summary, "total_labels": n, "accepted_labels": count,
            "agreement_labels": count, "disagreement_labels": n-count, "agreement_rate": count/n,
            "coverage": count / n, "selective_accuracy": correct / count if count else None,
            "baseline_accuracy": sum(sum(row["matches"].values()) for row in rows) / n,
            "accepted_errors": count - correct,
            "rejected_errors": sum(sum(not value for value in row["matches"].values()) for row in rows) - (count - correct),
            "complete_messages": len(complete), "message_coverage": len(complete) / len(rows),
            "selective_exact_match": sum(row["exact_match"] for row in complete) / len(complete) if complete else None}



# Fixed paraphrases: no earlier predictions, disagreement notice, or gold labels.
THIRD_INSTRUCTIONS = {
    "classification": "Choose the category that best describes the message: routine legitimate communication, ordinary advertising, or suspected fraud. Apply the risk rules below.",
    "industry": "Select the sector of the service discussed or claimed in the message, regardless of the payment method or whether the sender is authentic.",
    "urgent_action": "Is the recipient pushed to act quickly through a deadline, time pressure, or a consequence for delaying?",
    "sentiment": "Select the emotional tone expressed by the sender: positive, matter-of-fact, or negative. Ignore whether the claims are true.",
    "requested_action": "Select the principal thing the sender wants the recipient to do. Use the available action definitions.",
    "multiple_actions": "Are at least two independent actions requested together? Exclude alternatives, steps within one action, and multiple fields of information.",
    "message_hook": "Select the central premise used to get the recipient's attention. Apply the premise rules below regardless of whether the claim is true.",
    "sensitive_data_requested": "Is the recipient explicitly asked to disclose credentials, security codes, financial details, or government identity information or documents?",
}


def third_questions(keys: list[str], *, questions: dict | None = None) -> dict:
    questions = QUESTIONS if questions is None else questions
    return {
        key: {**questions[key], "instructions": (
            THIRD_INSTRUCTIONS[key] + " Decision rubric: " + questions[key]["instructions"]
        )}
        for key in keys
    }


def decision(key: str, answer: dict, *, questions: dict | None = None):
    questions = QUESTIONS if questions is None else questions
    if questions[key]["type"] == "choice":
        if answer["choice"] not in questions[key]["criteria"]:
            raise ValueError(f"Invalid third choice for {key}")
        probabilities = answer["probabilities"]
        if set(probabilities) != set(questions[key]["criteria"]):
            raise ValueError(f"Incomplete third distribution for {key}")
        values = list(probabilities.values())
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values) or abs(sum(values) - 1) > .005:
            raise ValueError(f"Invalid third distribution for {key}")
        # An exact tie has no decisive third vote.
        if sum(p == max(values) for p in values) > 1:
            return None
        return answer["choice"]
    probability = float(answer["noul"])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(f"Invalid third probability for {key}")
    return None if probability == .5 else probability > .5


def resolve_disagreements(backend, state: dict, row: dict, *, questions: dict | None = None) -> None:
    questions = QUESTIONS if questions is None else questions
    keys = [key for key in questions if not row["accepted"][key]]
    schema = third_questions(keys, questions=questions)
    started = time.perf_counter()
    third = backend.predict(state, questions=schema) if keys else {}
    elapsed = (time.perf_counter() - started) * 1000 if keys else 0.0
    resolutions = {}
    for key in keys:
        original = decision(key, row["positive"][key], questions=questions)
        proposed = decision(key, third[key], questions=questions)
        negative = float(row["negative"][key]["noul"])
        # A choice-negative answer only vetoes one candidate; it never votes
        # for a particular alternative. Only the original candidate can gain
        # two independent votes. Boolean complements do identify an alternative.
        supported = proposed is not None and proposed == original
        if questions[key]["type"] == "noul" and negative != .5:
            supported = supported or (proposed is not None and proposed == (negative < .5))
        resolutions[key] = {
            "original_decision": original, "third_decision": proposed,
            "third_correct": proposed == row["expected"][key],
            "supported": supported,
        }
    row.update(third=third, third_questions=schema, third_ms=elapsed,
               third_routing=getattr(backend, "last_routing", None) if keys else None,
               resolutions=resolutions)


def summarize_third(rows: list[dict], *, questions: dict | None = None) -> dict:
    questions = QUESTIONS if questions is None else questions
    disputed = [(row, key, result) for row in rows for key, result in row["resolutions"].items()]
    n = len(disputed)
    original_correct = sum(row["matches"][key] for row, key, _ in disputed)
    third_correct = sum(result["third_correct"] for _, _, result in disputed)
    supported = [(row, key, result) for row, key, result in disputed if result["supported"]]
    initial = sum(sum(row["accepted"].values()) for row in rows)
    initial_correct = sum(row["matches"][key] for row in rows for key in questions if row["accepted"][key])
    final_count = initial + len(supported)
    final_correct = initial_correct + sum(result["third_correct"] for _, _, result in supported)
    total = len(rows) * len(questions)
    return {
        "disputed_labels": n,
        "third_requests": sum(bool(row["third"]) for row in rows),
        "original_correct_on_disputes": original_correct,
        "original_accuracy_on_disputes": original_correct / n if n else None,
        "third_correct_on_disputes": third_correct,
        "third_accuracy_on_disputes": third_correct / n if n else None,
        "errors_fixed": sum(not row["matches"][key] and result["third_correct"] for row, key, result in disputed),
        "correct_answers_broken": sum(row["matches"][key] and not result["third_correct"] for row, key, result in disputed),
        "supported_disputes": len(supported),
        "unresolved_disputes": n - len(supported),
        "supported_dispute_errors": sum(not result["third_correct"] for _, _, result in supported),
        "accepted_labels_after_third": final_count,
        "coverage_after_third": final_count / total,
        "accepted_errors_after_third": final_count - final_correct,
        "selective_accuracy_after_third": final_correct / final_count if final_count else None,
        "replace_disputes_accuracy": (initial_correct + third_correct) / total,
        "per_question": {
            key: {"disputed": sum(k == key for _, k, _ in disputed),
                  "original_correct": sum(row["matches"][k] for row, k, _ in disputed if k == key),
                  "third_correct": sum(result["third_correct"] for _, k, result in disputed if k == key)}
            for key in questions
        },
    }

def evaluate_consistency(models: list[str], *, dataset: Path = DEFAULT_DATASET,
                         output: Path | None = None, device: str = "auto",
                         revision: str | None = None, details: bool = False,
                         third_request: bool = False, question_schema: Path | None = None) -> dict:
    questions, schema_metadata = load_question_schema(question_schema)
    examples = load_dataset(dataset) if question_schema is None else load_dataset(dataset, questions=questions)

    def predict_positive(backend, state):
        return backend.predict(state) if question_schema is None else backend.predict(state, questions=questions)
    packages = {}
    for package in ("laya", "decider-ai", "jevk5", "sentence-transformers", "gliner2", "torch", "transformers", "numpy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    report = {"schema_version": 2, "mode": "independent_third_on_disagreement" if third_request else "separate_negative_confirmation",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "dataset": str(dataset.resolve()),
              "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
              "questions": questions, "question_schema": schema_metadata, "packages": packages, "runs": []}
    for model in models:
        print(f"Loading and warming up {model}…", flush=True)
        started = time.perf_counter()
        backend = create_backend(model, device=device, revision=revision if model != "laya" else None)
        warmed_languages = set()
        for example in examples:
            language = example.get("language", "unspecified")
            if language in warmed_languages:
                continue
            warm_positive = predict_positive(backend, example["state"])
            backend.predict(example["state"], questions=negative_questions(warm_positive, questions=questions))
            if third_request:
                backend.predict(example["state"], questions=third_questions(list(questions), questions=questions))
            warmed_languages.add(language)
        startup = (time.perf_counter() - started) * 1000
        rows = []
        for example in examples:
            started = time.perf_counter()
            positive = predict_positive(backend, example["state"])
            elapsed_positive = (time.perf_counter() - started) * 1000
            positive_routing = getattr(backend, "last_routing", None)
            # The second request sees only a candidate and rubric, never the first
            # answer's probability or a claim that the model chose this candidate.
            negative_schema = negative_questions(positive, questions=questions)
            started = time.perf_counter()
            negative = backend.predict(example["state"], questions=negative_schema)
            elapsed_negative = (time.perf_counter() - started) * 1000
            score = score_answer(positive, example["expected"], questions=questions)
            rows.append({"id": example["id"], "language": example.get("language", "unspecified"),
                         "positive_routing": positive_routing,
                         "negative_routing": getattr(backend, "last_routing", None),
                         "expected": example["expected"],
                         "positive": positive, "negative": negative,
                         "negative_questions": negative_schema, "negative_schema_sha256": schema_hash(negative_schema),
                         "accepted": agreement(positive, negative, questions=questions),
                         "matches": score["matches"], "exact_match": score["exact_match"],
                         "positive_ms": elapsed_positive, "negative_ms": elapsed_negative})
            if third_request:
                resolve_disagreements(backend, example["state"], rows[-1], questions=questions)
            if details:
                print(f"{example['id']}: accepted {sum(rows[-1]['accepted'].values())}/{len(questions)} labels", flush=True)
        summary = summarize(rows, questions=questions)
        latency = {}
        for name, values in {
            "positive": [row["positive_ms"] for row in rows],
            "negative": [row["negative_ms"] for row in rows],
            "combined": [row["positive_ms"] + row["negative_ms"] + row.get("third_ms", 0) for row in rows],
        }.items():
            values.sort()
            latency[name] = {"p50": statistics.median(values),
                             "p95": values[math.ceil(len(values) * .95) - 1]}
        if model == "laya" and hasattr(backend, "router"):
            from .schema_sweep import model_identity
            import torch
            backend.metadata["checkpoints"] = model_identity(backend)
            backend.metadata["torch_threads"] = torch.get_num_threads()
        report["runs"].append({"model": dict(backend.metadata),
                               "latency_ms": latency,
                               "startup_and_warmup_ms": startup, "summary": summary,
                               "per_language": summarize_languages(rows, questions=questions),
                               "consistency_per_language": {language: summarize([row for row in rows if row.get("language", "unspecified") == language], questions=questions) for language in summarize_languages(rows, questions=questions)},
                               "predictions": rows})
        if third_request:
            report["runs"][-1]["third_summary"] = summarize_third(rows, questions=questions)
            report["runs"][-1]["third_per_language"] = {language: summarize_third([row for row in rows if row.get("language", "unspecified") == language], questions=questions) for language in summarize_languages(rows, questions=questions)}
            times = sorted(row["third_ms"] for row in rows if row["third"])
            latency["third_when_requested"] = {"p50": statistics.median(times), "p95": times[math.ceil(len(times) * .95) - 1]} if times else None
            print("Third-request results: " + json.dumps(report["runs"][-1]["third_summary"]))
        accuracy = f"{summary['selective_accuracy']:.1%}" if summary['selective_accuracy'] is not None else "n/a"
        print(f"{model}: baseline {summary['baseline_accuracy']:.1%}; "
              f"accepted accuracy {accuracy}; coverage {summary['coverage']:.1%}; "
              f"accepted errors {summary['accepted_errors']}; rejected errors {summary['rejected_errors']}")
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        del backend
        gc.collect()
    return report
