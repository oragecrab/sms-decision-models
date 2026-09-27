"""Load the base evaluation cases and report local model results."""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from .classifier import classify
from .formatting import COMPACT_LABELS, format_output
from .questions import QUESTIONS
from .simulation import simulate_requests
from .runtime import create_router, device_summary, get_cpu_threads, loaded_devices


def _timing_summary(label: str, values: list[float]) -> str:
    """Report milliseconds; p95 uses the nearest-rank percentile."""
    ordered = sorted(values)
    p95 = ordered[math.ceil(len(ordered) * 0.95) - 1]
    return (f"  {label}: p50 {statistics.median(ordered):.1f} ms · "
            f"p95 {p95:.1f} ms · max {ordered[-1]:.1f} ms")


def load_examples() -> list[dict[str, Any]]:
    path = Path(__file__).resolve().parents[2] / "datasets" / "base_eval_v3.jsonl"
    with path.open(encoding="utf-8") as dataset:
        return [json.loads(line) for line in dataset if line.strip()]


def run_demo(
    *, details: bool = False, simulate_server: bool = False, queue_size: int = 4,
    repeats: int = 1, device: str = "auto", offline: bool = False,
    cpu_threads: int | None = None
) -> int:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    base_examples = load_examples()
    examples = base_examples * repeats
    if repeats > 1:
        print(f"Run: {len(base_examples)} examples × {repeats} repeats = {len(examples)} requests")
    simulation = None
    if simulate_server:
        simulation = asyncio.run(simulate_requests(
            [example["state"] for example in examples], queue_size=queue_size,
            device=device, offline=offline, cpu_threads=cpu_threads
        ))
        print(f"Server simulation: {len(examples)} concurrent requests · 1 Router · "
              f"1 inference worker · queue capacity {queue_size}")
        print(device_summary(simulation.devices, requested=device))
        if simulation.cpu_threads is not None:
            print(f"CPU computation threads: {simulation.cpu_threads} (intra-op)")
        print(f"Startup + warmup: {simulation.startup_ms:.1f} ms\n")
    else:
        router = create_router(device=device, offline=offline, cpu_threads=cpu_threads)
    scores = {question_id: 0 for question_id in QUESTIONS}
    exact_matches = 0
    inference_times = []
    first_predictions = {}
    consistent_repeats = 0
    reported_devices = None
    language_scores = {}
    for index, example in enumerate(examples, start=1):
        state = example["state"]
        if simulation is None:
            started = time.perf_counter()
            answer = classify(state, router)
            elapsed_ms = (time.perf_counter() - started) * 1000
            devices = loaded_devices(router)
            if devices != reported_devices:
                print(device_summary(devices, requested=device))
                if "cpu" in devices.values():
                    print(f"CPU computation threads: {get_cpu_threads()} (intra-op)")
                reported_devices = devices
        else:
            response = simulation.requests[index - 1]
            answer = response.answers
            elapsed_ms = response.inference_ms
        inference_times.append(elapsed_ms)
        title = f"{index:02d} · {example['id']}"
        if repeats > 1:
            title += f" · repeat {(index - 1) // len(base_examples) + 1}/{repeats}"
        print(
            format_output(
                state,
                answer,
                title=title,
                expected=example["expected"],
                elapsed_ms=elapsed_ms,
                details=details,
                compact=True,
            )
        )
        if simulation is not None:
            print(f"  Wait: {response.wait_ms:.1f} ms · "
                  f"Total request latency: {response.total_ms:.1f} ms")
        results = {
            question_id: (
                question_answer.get("choice")
                if "choice" in question_answer
                else question_answer.get("noul", 0) >= 0.5
            )
            for question_id, question_answer in answer.items()
        }
        if example["id"] in first_predictions:
            consistent_repeats += results == first_predictions[example["id"]]
        else:
            first_predictions[example["id"]] = results
        for question_id, expected_value in example["expected"].items():
            scores[question_id] += results.get(question_id) == expected_value
        language = example.get("language", "unspecified")
        counts = language_scores.setdefault(language, {"messages": 0, "correct": 0})
        counts["messages"] += 1
        counts["correct"] += sum(results.get(key) == value for key, value in example["expected"].items())
        exact_matches += all(
            results.get(question_id) == expected_value
            for question_id, expected_value in example["expected"].items()
        )
        if index < len(examples):
            print()
    summary_label = f"{len(examples)}-message base eval"
    if repeats > 1:
        summary_label = f"{len(examples)} requests ({len(base_examples)} examples × {repeats} repeats)"
    print(
        f"\n{summary_label} · "
        f"exact matches {exact_matches}/{len(examples)}"
    )
    print("  " + "  ".join(
        f"{COMPACT_LABELS[question_id]} {scores[question_id]}/{len(examples)}"
        for question_id in QUESTIONS
    ))
    if repeats > 1:
        comparisons = len(base_examples) * (repeats - 1)
        print(f"  Repeat consistency: {consistent_repeats}/{comparisons} "
              "requests match their first-pass labels")
    if inference_times:
        print(_timing_summary("Inference", inference_times))
    if simulation is not None:
        if simulation.requests:
            print(_timing_summary("Wait (capacity + queue)",
                                  [response.wait_ms for response in simulation.requests]))
            print(_timing_summary("Request latency",
                                  [response.total_ms for response in simulation.requests]))
        print(f"  Serving time: {simulation.elapsed_ms:.1f} ms (excludes startup)")
        if simulation.elapsed_ms > 0:
            print(f"  Throughput: {len(examples) * 1000 / simulation.elapsed_ms:.2f} messages/s")
    for language, counts in sorted(language_scores.items()):
        total = counts["messages"] * len(QUESTIONS)
        print(f"  {language}: {counts['messages']} messages; {counts['correct']}/{total} labels correct")
    return 0
