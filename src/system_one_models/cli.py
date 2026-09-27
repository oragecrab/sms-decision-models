"""Command-line argument handling for message classification and evaluation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .benchmark import DEFAULT_DATASET, evaluate
from .backends import create_backend
from .consistency import evaluate_consistency

from .classifier import build_state
from .evaluation import run_demo
from .formatting import format_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify and evaluate SMS or email locally with Laya, Decider, JevK5, Zerank, GLiNER, or CLM.")
    parser.add_argument("--channel", choices=("sms", "email"), default="sms")
    parser.add_argument("--sender", default="")
    parser.add_argument("--subject", default="")
    parser.add_argument("--body", help="Message text. Use --demo to run the bundled examples.")
    parser.add_argument("--demo", action="store_true", help="Run the bundled base evaluation set.")
    parser.add_argument(
        "--details", action="store_true", help="Show every choice probability."
    )
    parser.add_argument("--eval", action="store_true", help="Compare models on the same labeled dataset.")
    parser.add_argument("--consistency", action="store_true", help="With --eval, confirm each prediction in a separate negative request.")
    parser.add_argument("--third-request", action="store_true", help="With --eval --consistency, retry disputed questions independently.")
    parser.add_argument("--question-language", choices=("en", "fr", "match"), default="en", help="With --eval, use English/French questions or match each annotated message language.")
    parser.add_argument("--question-schema", type=Path, help="With --eval, use a versioned English question-schema JSON file.")
    parser.add_argument("--model", action="append", help="Repeat to compare: laya, a Decider repo, alibiserikbay/JevK5, zeroentropy/zerank-2-reranker, fastino/GLiNER2.5-Decide, clm, or a prefixed local checkpoint.")
    parser.add_argument("--revision", help="Hugging Face revision for all non-Laya models; use a commit SHA for reproducibility.")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--dataset", type=Path, help="Evaluation JSONL with id, state, and expected labels.")
    parser.add_argument("--output", type=Path, help="Save evaluation report as JSON.")
    parser.add_argument("--task-count-policy", type=Path, help="Use a frozen task-count policy for this message instead of the eight-question workflow.")
    parser.add_argument("--task-count-opposition", action="store_true", help="With --task-count-policy, also report a separate complementary check.")
    args = parser.parse_args()
    if args.task_count_opposition and args.task_count_policy is None:
        parser.error("--task-count-opposition requires --task-count-policy")
    if args.task_count_policy is not None:
        if args.eval or args.demo or args.dataset or args.output or args.model or args.revision or args.question_schema or args.question_language != "en" or args.consistency or args.third_request:
            parser.error("--task-count-policy is a separate single-message mode; its model and schema are frozen in the policy file")
        if not args.body:
            parser.error("--task-count-policy requires --body")
        try:
            from .action_policy import TaskCountPolicy
            state = build_state(channel=args.channel,sender=args.sender,subject=args.subject,body=args.body)
            result = TaskCountPolicy(args.task_count_policy,device=args.device).predict(state,check_opposition=args.task_count_opposition)
        except (ValueError, RuntimeError, OSError) as error:
            parser.error(str(error))
        if not args.details:
            result.pop("positive_answers",None)
        print(json.dumps(result,indent=2,allow_nan=False))
        return
    models = args.model or ["laya"]
    if args.question_schema and (not args.eval or args.question_language != "en"):
        parser.error("--question-schema requires --eval with --question-language en")
    if args.question_language != "en" and (not args.eval or args.consistency):
        parser.error("--question-language fr/match requires ordinary --eval")
    if args.third_request and not (args.eval and args.consistency):
        parser.error("--third-request requires --eval --consistency")
    if args.consistency and not args.eval:
        parser.error("--consistency requires --eval")
    if args.eval and args.demo:
        parser.error("choose --eval or --demo")
    if not args.eval and not args.demo and (args.dataset or args.output):
        parser.error("--dataset and --output require --eval or --demo")
    if not args.eval and not args.demo and len(models) != 1:
        parser.error("multiple --model values require --eval or --demo")
    if args.revision and all(model == "laya" for model in models):
        parser.error("--revision requires a non-Laya model")
    if args.eval or (args.demo and (args.model or args.dataset or args.output or args.device != "auto")):
        try:
            evaluator = evaluate_consistency if args.consistency else evaluate
            extra = {"third_request": True} if args.third_request else {}
            if not args.consistency:
                extra["question_language"] = args.question_language
            if args.question_schema is not None:
                extra["question_schema"] = args.question_schema
            evaluator(models, dataset=args.dataset or DEFAULT_DATASET, output=args.output,
                     device=args.device, revision=args.revision, details=args.details, **extra)
        except (ValueError, RuntimeError, OSError) as error:
            parser.error(str(error))
        return

    if args.demo:
        raise SystemExit(run_demo(details=args.details))
    if not args.body:
        parser.error("provide --body or use --demo")

    state = build_state(channel=args.channel, sender=args.sender, subject=args.subject, body=args.body)
    started = time.perf_counter()
    try:
        backend = create_backend(models[0], device=args.device, revision=args.revision)
        answer = backend.predict(state)
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(format_output(state, answer, elapsed_ms=elapsed_ms, details=args.details, model_name=models[0]))
