"""Simulate concurrent requests served by one preloaded inference worker."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .classifier import classify
from .runtime import create_router, get_cpu_threads, loaded_devices


@dataclass(frozen=True)
class RequestResult:
    answers: dict[str, Any]
    wait_ms: float
    inference_ms: float
    total_ms: float


@dataclass(frozen=True)
class SimulationResult:
    requests: list[RequestResult]
    startup_ms: float
    elapsed_ms: float
    devices: dict[str, str] = field(default_factory=dict)
    cpu_threads: int | None = None


async def simulate_requests(
    states: list[dict[str, Any]], *, queue_size: int = 4,
    device: str = "auto", offline: bool = False, cpu_threads: int | None = None
) -> SimulationResult:
    """Submit a burst of requests; await capacity instead of dropping requests.

    Loading, warmup, and inference all run in the same worker thread. Waiting
    time includes both queue-capacity backpressure and time spent in the queue.
    Results retain input order. This simulates serving, without opening HTTP ports.
    """
    if queue_size < 1:
        raise ValueError("queue_size must be positive")
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya-inference")
    queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
    worker = None
    clients: list[asyncio.Task] = []

    def initialize():
        router = create_router(device=device, offline=offline, cpu_threads=cpu_threads)
        router.load("english")
        classify({"channel": "sms", "body": "Routine notification. No action is needed."}, router)
        return router

    def infer(state, submitted):
        started = time.perf_counter()
        answers = classify(state, router)
        finished = time.perf_counter()
        return RequestResult(
            answers=answers,
            wait_ms=(started - submitted) * 1000,
            inference_ms=(finished - started) * 1000,
            total_ms=(finished - submitted) * 1000,
        )

    async def serve():
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                state, future, submitted = item
                if future.cancelled():
                    continue
                try:
                    result = await loop.run_in_executor(executor, infer, state, submitted)
                except Exception as error:
                    if not future.done():
                        future.set_exception(error)
                else:
                    if not future.done():
                        future.set_result(result)
            finally:
                queue.task_done()

    async def request(state):
        submitted = time.perf_counter()
        future = loop.create_future()
        await queue.put((state, future, submitted))
        return await future

    try:
        started = time.perf_counter()
        router = await loop.run_in_executor(executor, initialize)
        startup_ms = (time.perf_counter() - started) * 1000
        worker = asyncio.create_task(serve())
        started = time.perf_counter()
        clients = [asyncio.create_task(request(state)) for state in states]
        # Let all admitted requests finish even when an individual request fails.
        results = await asyncio.gather(*clients, return_exceptions=True)
        elapsed_ms = (time.perf_counter() - started) * 1000
        for result in results:
            if isinstance(result, BaseException):
                raise result
        devices = await loop.run_in_executor(executor, loaded_devices, router)
        actual_cpu_threads = None
        if "cpu" in devices.values():
            actual_cpu_threads = await loop.run_in_executor(executor, get_cpu_threads)
        return SimulationResult(requests=results, startup_ms=startup_ms,
                                elapsed_ms=elapsed_ms, devices=devices, cpu_threads=actual_cpu_threads)
    finally:
        for client in clients:
            if not client.done():
                client.cancel()
        if clients:
            await asyncio.gather(*clients, return_exceptions=True)
        if worker is not None:
            await queue.put(None)
            await worker
        executor.shutdown(wait=True, cancel_futures=True)
