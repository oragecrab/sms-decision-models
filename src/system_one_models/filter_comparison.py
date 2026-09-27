"""Offline confidence/negative-confirmation comparisons; never run a model."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from .benchmark import score_answer
from .consistency import agreement
from .question_schemas import schema_hash


def labels(report: dict, run: dict) -> list[dict]:
    questions = report["questions"]
    result = []
    ids = set()
    for row in run["predictions"]:
        if row["id"] in ids:
            raise ValueError("Duplicate prediction id")
        ids.add(row["id"])
        matches = score_answer(row["positive"], row["expected"], questions=questions)["matches"]
        accepted = agreement(row["positive"], row["negative"], questions=questions)
        if matches != row["matches"] or accepted != row["accepted"]:
            raise ValueError("Saved scores or agreement do not match raw predictions")
        for key, spec in questions.items():
            answer = row["positive"][key]
            if spec["type"] == "noul":
                predicted = answer["noul"] >= .5
                confidence = max(answer["noul"], 1 - answer["noul"])
            else:
                predicted = answer["choice"]
                confidence = answer["probabilities"][predicted]
            result.append(dict(id=row["id"], language=row.get("language", "unspecified"),
                               question=key, expected=row["expected"][key], predicted=predicted,
                               confidence=confidence, native_confidence=answer.get("confidence"),
                               correct=matches[key], opposition=accepted[key]))
    if not result:
        raise ValueError("No saved predictions")
    return result


def metrics(rows: list[dict], accepted: list[dict]) -> dict:
    correct = sum(row["correct"] for row in accepted)
    errors = len(accepted) - correct
    wrong = [row for row in accepted if not row["correct"]]
    confusion = {}
    for row in wrong:
        key = f"{row['question']}: {row['expected']} -> {row['predicted']}"
        confusion[key] = confusion.get(key, 0) + 1
    return dict(total=len(rows), accepted=len(accepted), coverage=len(accepted)/len(rows),
                accepted_correct=correct, accepted_errors=errors,
                rejected_errors=sum(not row["correct"] for row in rows)-errors,
                rejected_correct=sum(row["correct"] for row in rows)-correct,
                selective_accuracy=correct/len(accepted) if accepted else None,
                accepted_false_positives=sum(type(r["expected"]) is bool and r["predicted"] is True for r in wrong),
                accepted_false_negatives=sum(type(r["expected"]) is bool and r["predicted"] is False for r in wrong),
                accepted_missed_scams=sum(r["question"] == "classification" and r["expected"] == "suspected_scam" for r in wrong),
                accepted_false_scam_flags=sum(r["question"] == "classification" and r["predicted"] == "suspected_scam" for r in wrong),
                error_confusion=confusion)


def matched(rows: list[dict], *, score: str = "confidence") -> dict:
    opposition = [r for r in rows if r["opposition"]]
    k = len(opposition)
    ranked = sorted(rows, key=lambda r: (-r[score], r["id"], r["question"]))
    selected = ranked[:k]
    boundary = None
    if 0 < k < len(rows) and ranked[k-1][score] == ranked[k][score]:
        cutoff = ranked[k-1][score]
        above = [r for r in rows if r[score] > cutoff]
        tied = [r for r in rows if r[score] == cutoff]
        slots = k - len(above)
        tied_errors = sum(not r["correct"] for r in tied)
        above_errors = sum(not r["correct"] for r in above)
        boundary = dict(confidence=cutoff, tied_labels=len(tied), selected_from_tie=slots,
                        accepted_errors_min=above_errors+max(0, slots-(len(tied)-tied_errors)),
                        accepted_errors_max=above_errors+min(slots,tied_errors))
    chosen = {(r["id"], r["question"]) for r in selected}
    overlap = [r for r in opposition if (r["id"], r["question"]) in chosen]
    confidence_only = [r for r in selected if not r["opposition"]]
    opposition_only = [r for r in opposition if (r["id"], r["question"]) not in chosen]
    return dict(opposition=metrics(rows, opposition), confidence_top_k=metrics(rows, selected),
                accepted_error_difference=sum(not r["correct"] for r in selected)-sum(not r["correct"] for r in opposition),
                overlap=len(overlap), confidence_only=len(confidence_only), opposition_only=len(opposition_only),
                confidence_only_errors=sum(not r["correct"] for r in confidence_only),
                opposition_only_errors=sum(not r["correct"] for r in opposition_only),
                boundary_tie=boundary,
                confidence_accepted_keys=[[r["id"], r["question"]] for r in selected],
                confidence_accepted_ids=[f"{r['id']}/{r['question']}" for r in selected])


def curve(rows: list[dict], *, score: str = "confidence") -> list[dict]:
    # Whole score ties enter together; these are genuine >= threshold policies.
    ranked = sorted(rows, key=lambda r: -r[score])
    points = [dict(threshold=None, rule="accept_none", **metrics(rows, []))]
    for i, row in enumerate(ranked):
        if i+1 == len(ranked) or ranked[i+1][score] != row[score]:
            points.append(dict(threshold=row[score], rule="confidence >= threshold",
                               **metrics(rows, ranked[:i+1])))
    return points


def analyze(report: dict, run: dict, *, score: str = "confidence") -> dict:
    rows = labels(report, run)
    if any(r.get(score) is None or not math.isfinite(r[score]) or not 0 <= r[score] <= 1 for r in rows):
        raise ValueError(f"Invalid or missing {score}")
    groups = {"overall": rows}
    for language in sorted({r["language"] for r in rows}):
        groups[f"language/{language}"] = [r for r in rows if r["language"] == language]
    for question in report["questions"]:
        groups[f"question/{question}"] = [r for r in rows if r["question"] == question]
        for language in sorted({r["language"] for r in rows}):
            groups[f"question_language/{question}/{language}"] = [r for r in rows if r["question"] == question and r["language"] == language]
    comparisons = {key: matched(group, score=score) for key, group in groups.items()}
    selected_ids = {tuple(label) for key, comparison in comparisons.items() if key.startswith("question_language/")
                    for label in comparison["confidence_accepted_keys"]}
    selected = [r for r in rows if (r["id"], r["question"]) in selected_ids]
    stratified = dict(opposition=metrics(rows, [r for r in rows if r["opposition"]]),
                      confidence_top_k=metrics(rows, selected),
                      definition="Match the opposition acceptance count separately in each question/language cell")
    return dict(score=score, model=run["model"], comparisons=comparisons, stratified_matched=stratified,
                threshold_curves={key: curve(group, score=score) for key, group in groups.items()},
                recorded_latency_ms=run.get("latency_ms"),
                labels=rows)


def compare_files(paths: list[Path]) -> dict:
    sources = []
    dataset_hash = None
    for path in paths:
        raw = path.read_bytes()
        report = json.loads(raw)
        if schema_hash(report["questions"]) != report["question_schema"]["sha256"]:
            raise ValueError("Schema hash does not match saved questions")
        if report["mode"] != "separate_negative_confirmation":
            raise ValueError("Expected separate negative confirmation reports")
        if dataset_hash is not None and dataset_hash != report["dataset_sha256"]:
            raise ValueError("Reports use different datasets")
        dataset_hash = report["dataset_sha256"]
        sources.append(dict(source=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                            dataset_sha256=dataset_hash, question_schema=report["question_schema"],
                            runs=[dict(primary=analyze(report, run), native_confidence_sensitivity=analyze(report, run, score="native_confidence")) for run in report["runs"]]))
    return dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
                method=dict(confidence="Probability of the selected positive answer; max(p,1-p) for Boolean questions",
                            matching="Top k with k equal to opposition accepted count; ties broken by id/question without labels",
                            thresholds="All unique scores, accepting complete ties; no gold-based threshold optimization",
                            caveat="Synthetic development-set analysis, uncalibrated confidence; exact top-k is a ranking diagnostic, not a deployable fixed threshold"),
                sources=sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = compare_files(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

if __name__ == "__main__":
    main()
