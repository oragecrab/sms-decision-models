"""Command-line argument handling for message classification and evaluation."""

from __future__ import annotations

import argparse
import time

from .classifier import MessageTooLongError, build_state, classify
from .evaluation import run_demo
from .formatting import format_output
from .runtime import DEVICES, OfflineCacheError, create_router, device_summary, get_cpu_threads, loaded_devices


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
    parser.add_argument(
        "--simulate-server", action="store_true",
        help="With --demo, simulate concurrent requests sharing a preloaded Router.",
    )
    parser.add_argument(
        "--queue-size", type=int, default=4,
        help="Waiting request capacity for --simulate-server (default: 4).",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Run the demo set this many times using the same Router (default: 1).",
    )
    parser.add_argument(
        "--device", choices=DEVICES, default="auto",
        help="Inference device (default: auto). Reports the actual device after fallback.",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Use cached checkpoint files only, without checking Hugging Face for updates.",
    )
    parser.add_argument(
        "--cpu-threads", type=int,
        help="PyTorch CPU computation threads; defaults to PyTorch's existing setting.",
    )
    args = parser.parse_args()
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.repeats != 1 and not args.demo:
        parser.error("--repeats requires --demo")
    if args.simulate_server and not args.demo:
        parser.error("--simulate-server requires --demo")
    if args.queue_size < 1:
        parser.error("--queue-size must be positive")

    if args.demo:
        try:
            status = run_demo(details=args.details, simulate_server=args.simulate_server,
                              queue_size=args.queue_size, repeats=args.repeats,
                              device=args.device, offline=args.offline, cpu_threads=args.cpu_threads)
        except (MessageTooLongError, OfflineCacheError) as error:
            parser.error(str(error))
        raise SystemExit(status)
    if not args.body:
        parser.error("provide --body or use --demo")

    state = build_state(channel=args.channel, sender=args.sender, subject=args.subject, body=args.body)
    started = time.perf_counter()
    try:
        router = create_router(device=args.device, offline=args.offline, cpu_threads=args.cpu_threads)
        answer = classify(state, router)
    except (MessageTooLongError, OfflineCacheError) as error:
        parser.error(str(error))
    elapsed_ms = (time.perf_counter() - started) * 1000
    devices = loaded_devices(router)
    print(device_summary(devices, requested=args.device))
    if "cpu" in devices.values():
        print(f"CPU computation threads: {get_cpu_threads()} (intra-op)")
    print(format_output(state, answer, elapsed_ms=elapsed_ms, details=args.details))
