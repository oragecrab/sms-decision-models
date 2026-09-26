"""Command-line argument handling for message classification and evaluation."""

from __future__ import annotations

import argparse
import time

from .classifier import MessageTooLongError, build_state, classify
from .evaluation import run_demo
from .formatting import format_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one SMS or email locally with Laya.")
    parser.add_argument("--channel", choices=("sms", "email"), default="sms")
    parser.add_argument("--sender", default="")
    parser.add_argument("--subject", default="")
    parser.add_argument("--body", help="Message text. Use --demo to run the bundled examples.")
    parser.add_argument("--demo", action="store_true", help="Run the bundled base evaluation set.")
    parser.add_argument(
        "--details", action="store_true", help="Show every choice probability."
    )
    args = parser.parse_args()

    if args.demo:
        raise SystemExit(run_demo(details=args.details))
    if not args.body:
        parser.error("provide --body or use --demo")

    state = build_state(channel=args.channel, sender=args.sender, subject=args.subject, body=args.body)
    started = time.perf_counter()
    try:
        answer = classify(state)
    except MessageTooLongError as error:
        parser.error(str(error))
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(format_output(state, answer, elapsed_ms=elapsed_ms, details=args.details))
