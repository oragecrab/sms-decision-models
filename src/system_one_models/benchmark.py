"""Evaluate local backends sequentially on identical inputs and save comparisons."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .backends import create_backend
from .formatting import format_output
from .questions import QUESTIONS
from .question_schemas import load_question_schema, schema_hash

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "datasets/base_eval_v3.jsonl"


def load_dataset(path: Path, *, questions: dict | None = None) -> list[dict]:
    questions = QUESTIONS if questions is None else questions
    examples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not examples:
        raise ValueError("Evaluation dataset must not be empty")
    ids = set()
    for example in examples:
        if example["id"] in ids:
            raise ValueError(f"Duplicate example id: {example['id']}")
        ids.add(example["id"])
        if "language" in example and example["language"] not in {"en", "fr"}:
            raise ValueError(f"Invalid language for {example['id']}: expected en or fr")
        if not isinstance(example["state"], dict):
            raise ValueError("Each state must be an object")
        if set(example["expected"]) != set(questions):
            raise ValueError(f"Example {example['id']} must label every question")
        for key, spec in questions.items():
            value = example["expected"][key]
            valid = type(value) is bool if spec["type"] == "noul" else value in spec["criteria"]
            if not valid:
                raise ValueError(f"Invalid expected label for {example['id']}/{key}")
    pairs = {}
    for example in examples:
        if "pair_id" in example:
            pairs.setdefault(example["pair_id"], []).append(example)
    for pair_id, members in pairs.items():
        if len(members) != 2 or {row.get("language") for row in members} != {"en", "fr"}:
            raise ValueError(f"Pair {pair_id} must contain one English and one French example")
        if members[0]["expected"] != members[1]["expected"]:
            raise ValueError(f"Pair {pair_id} must have identical expected labels")
    return examples


def score_answer(answer: dict, expected: dict, *, questions: dict | None = None) -> dict:
    questions = QUESTIONS if questions is None else questions
    if set(answer) != set(questions):
        raise ValueError("Answers must contain exactly the selected question IDs")
    matches, gold_probabilities, briers = {}, {}, {}
    for key, spec in questions.items():
        row = answer[key]
        if spec["type"] == "noul":
            yes = float(row["noul"])
            probabilities = {False: 1 - yes, True: yes}
            predicted = yes >= 0.5
        else:
            probabilities = row["probabilities"]
            if set(probabilities) != set(spec["criteria"]):
                raise ValueError(f"Incomplete probability distribution for {key}")
            predicted = row["choice"]
            if predicted not in probabilities:
                raise ValueError(f"Invalid choice for {key}")
        values = list(probabilities.values())
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values) or abs(sum(values) - 1) > 0.005:
            raise ValueError(f"Invalid probability distribution for {key}")
        gold = expected[key]
        matches[key] = predicted == gold
        gold_probabilities[key] = probabilities[gold]
        briers[key] = sum((p - float(label == gold)) ** 2 for label, p in probabilities.items())
    return {"matches": matches, "exact_match": all(matches.values()),
            "gold_probabilities": gold_probabilities, "brier": briers}



def summarize_languages(rows: list[dict], *, questions: dict | None = None) -> dict:
    """Keep unspecified legacy data separate; never infer language from gold labels."""
    questions = QUESTIONS if questions is None else questions
    result = {}
    for language in sorted({row.get("language", "unspecified") for row in rows}):
        group = [row for row in rows if row.get("language", "unspecified") == language]
        result[language] = {
            "examples": len(group),
            "label_accuracy": sum(sum(row["matches"].values()) for row in group) / (len(group) * len(questions)),
            "exact_match_accuracy": statistics.mean(row["exact_match"] for row in group),
            "per_question": {key: statistics.mean(row["matches"][key] for row in group) for key in questions},
        }
    return result


def summarize_pairs(rows: list[dict], *, questions: dict | None = None) -> dict:
    questions = QUESTIONS if questions is None else questions
    groups = {}
    for row in rows:
        if row.get("pair_id") is not None:
            groups.setdefault(row["pair_id"], {})[row["language"]] = row
    pairs = []
    for pair_id, group in groups.items():
        en, fr = group["en"], group["fr"]
        # Match booleans using the same >= .5 scoring rule as the baseline.
        predictions = {
            language: {key: answer["choice"] if questions[key]["type"] == "choice" else answer["noul"] >= .5
                       for key, answer in row["answers"].items()}
            for language, row in group.items()
        }
        same = {key: predictions["en"][key] == predictions["fr"][key] for key in questions}
        pairs.append({"pair_id": pair_id, "source_language": en.get("source_language"),
                      "english_id": en["id"], "french_id": fr["id"],
                      "english_correct": sum(en["matches"].values()),
                      "french_correct": sum(fr["matches"].values()),
                      "same_answers": same,
                      "english_predictions": predictions["en"], "french_predictions": predictions["fr"],
                      "expected": en["expected"],
                      "different_questions": [key for key, value in same.items() if not value]})
    if not pairs:
        return {"pair_count": 0}
    by_question = {}
    for key in questions:
        en_right_fr_wrong = sum(groups[p["pair_id"]]["en"]["matches"][key] and not groups[p["pair_id"]]["fr"]["matches"][key] for p in pairs)
        fr_right_en_wrong = sum(groups[p["pair_id"]]["fr"]["matches"][key] and not groups[p["pair_id"]]["en"]["matches"][key] for p in pairs)
        by_question[key] = {"same": sum(p["same_answers"][key] for p in pairs),
                            "english_correct_french_wrong": en_right_fr_wrong,
                            "french_correct_english_wrong": fr_right_en_wrong}
    return {"pair_count": len(pairs),
            "same_labels": sum(sum(p["same_answers"].values()) for p in pairs),
            "total_paired_labels": len(pairs) * len(questions),
            "label_agreement": sum(sum(p["same_answers"].values()) for p in pairs) / (len(pairs) * len(questions)),
            "all_answers_agree_pairs": sum(all(p["same_answers"].values()) for p in pairs),
            "per_question": by_question,
            "by_source_language": {language: {
                "pairs": sum(p["source_language"] == language for p in pairs),
                "english_correct": sum(p["english_correct"] for p in pairs if p["source_language"] == language),
                "french_correct": sum(p["french_correct"] for p in pairs if p["source_language"] == language)}
                for language in sorted({p["source_language"] for p in pairs})},
            "pairs": pairs}

def evaluate(models: list[str], *, dataset: Path = DEFAULT_DATASET,
             output: Path | None = None, device: str = "auto",
             revision: str | None = None, details: bool = False,
             question_language: str = "en", question_schema: Path | None = None) -> dict:
    if question_language not in {"en", "fr", "match"}:
        raise ValueError("question_language must be en, fr, or match")
    if question_schema is not None and question_language != "en":
        raise ValueError("Custom question schemas require --question-language en")
    questions, schema_metadata = load_question_schema(question_schema)
    from .french_questions import french_schema, canonical_answers
    french, mappings = french_schema()
    examples = load_dataset(dataset) if question_schema is None else load_dataset(dataset, questions=questions)

    def predict(backend, example):
        language = example.get("language") if question_language == "match" else question_language
        if language == "fr":
            raw = backend.predict(example["state"], questions=french)
            return canonical_answers(raw, mappings), raw, "fr"
        raw = backend.predict(example["state"]) if question_schema is None else backend.predict(example["state"], questions=questions)
        return raw, raw, "en"
    packages = {}
    for package in ("laya", "decider-ai", "jevk5", "sentence-transformers", "gliner2", "torch", "transformers"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    report = {
        "schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset.resolve()),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "questions": questions, "question_language": question_language,
        "question_schema": schema_metadata,
        "model_facing_schemas": {
            "en": {"version": schema_metadata["version"], "sha256": schema_metadata["sha256"], "questions": questions},
            **({"fr": {"version": "baseline-fr-v1", "sha256": schema_hash(french), "questions": french}} if question_language != "en" else {}),
        },
        "model_facing_questions_fr": french if question_language != "en" else None,
        "french_to_canonical_labels": mappings if question_language != "en" else None,
        "packages": packages, "runs": [],
    }
    for model in models:
        print(f"\nLoading and warming up {model}…", flush=True)
        started = time.perf_counter()
        backend = create_backend(model, device=device, revision=revision if model != "laya" else None)
        predict(backend, examples[0])
        startup_ms = (time.perf_counter() - started) * 1000
        probability_metrics = backend.metadata.get("probability_metrics_supported", True)
        rows = []
        for example in examples:
            started = time.perf_counter()
            answer, raw, used_language = predict(backend, example)
            elapsed = (time.perf_counter() - started) * 1000
            rows.append({"id": example["id"], "language": example.get("language", "unspecified"),
                         "routing": getattr(backend, "last_routing", None),
                         "pair_id": example.get("pair_id"), "source_language": example.get("source_language"),
                         "is_translation": example.get("is_translation", False), "expected": example["expected"],
                         "answers": answer, "raw_answers": raw, "question_language": used_language, "inference_ms": elapsed,
                         **score_answer(answer, example["expected"], questions=questions)})
            if not probability_metrics:
                rows[-1].pop("gold_probabilities")
                rows[-1].pop("brier")
            if details:
                print(format_output(example["state"], answer, title=example["id"],
                                    expected=example["expected"], elapsed_ms=elapsed,
                                    details=True, model_name=model))
        times = sorted(row["inference_ms"] for row in rows)
        per_question = {}
        for key in questions:
            per_question[key] = {
                "accuracy": statistics.mean(row["matches"][key] for row in rows),
                "nll": statistics.mean(-math.log(max(row["gold_probabilities"][key], 1e-12)) for row in rows) if probability_metrics else None,
                "brier": statistics.mean(row["brier"][key] for row in rows) if probability_metrics else None,
            }
        run = {"model": dict(backend.metadata), "startup_and_warmup_ms": startup_ms,
               "examples": len(rows), "exact_match_accuracy": statistics.mean(r["exact_match"] for r in rows),
               "per_question": per_question,
               "latency_ms": {"p50": statistics.median(times), "p95": times[math.ceil(len(times)*0.95)-1]},
               "per_language": summarize_languages(rows, questions=questions), "translation_pairs": summarize_pairs(rows, questions=questions), "predictions": rows}
        report["runs"].append(run)
        print(f"{model}: exact match {run['exact_match_accuracy']:.1%}; "
              f"p50 {run['latency_ms']['p50']:.1f} ms; p95 {run['latency_ms']['p95']:.1f} ms; "
              f"load + warmup {startup_ms:.1f} ms")
        for language, metrics in run["per_language"].items():
            print(f"  {language}: {metrics['examples']} messages; label accuracy {metrics['label_accuracy']:.1%}; exact match {metrics['exact_match_accuracy']:.1%}")
        if run["translation_pairs"]["pair_count"]:
            paired = run["translation_pairs"]
            print(f"  Translation agreement: {paired['same_labels']}/{paired['total_paired_labels']} labels; {paired['all_answers_agree_pairs']}/{paired['pair_count']} entire pairs")
        if not probability_metrics:
            print("  Relevance weights are uncalibrated; NLL and Brier scores are omitted.")
        for key, metrics in per_question.items():
            suffix = f", NLL {metrics['nll']:.3f}, Brier {metrics['brier']:.3f}" if probability_metrics else ""
            print(f"  {key}: accuracy {metrics['accuracy']:.1%}{suffix}")
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        del backend
        gc.collect()
    print("\nComparison (same dataset and questions)")
    print(f"{'Model':35} {'Exact':>8} {'Mean accuracy':>14} {'Mean NLL':>10} {'p50 ms':>10}")
    for run in report["runs"]:
        metrics = run["per_question"].values()
        nll = (f"{statistics.mean(m['nll'] for m in metrics):.3f}"
               if run["model"].get("probability_metrics_supported", True) else "n/a")
        print(f"{run['model']['model']:35} {run['exact_match_accuracy']:8.1%} "
              f"{statistics.mean(m['accuracy'] for m in metrics):14.1%} "
              f"{nll:>10} "
              f"{run['latency_ms']['p50']:10.1f}")
    return report
