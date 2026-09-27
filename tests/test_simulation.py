from __future__ import annotations

import asyncio
import threading

import pytest

from system_one_models import simulation
from system_one_models.classifier import MessageTooLongError


@pytest.fixture
def fake_router(monkeypatch):
    import laya

    instances = []

    class FakeRouter:
        def __init__(self, *, device=None, models=None):
            self.device = device
            self.models = models
            self.loads = []
            instances.append(self)

        def load(self, name):
            self.loads.append(name)
            from types import SimpleNamespace
            return SimpleNamespace(device="cpu")

        @property
        def loaded(self):
            return list(dict.fromkeys(self.loads))

    monkeypatch.setattr(laya, "Router", FakeRouter)
    return instances


def test_shared_worker_keeps_event_loop_responsive_and_preserves_order(monkeypatch, fake_router):
    main_thread = threading.get_ident()
    inference_threads = set()
    calls = []
    release = threading.Event()

    async def scenario():
        loop = asyncio.get_running_loop()
        inference_started = asyncio.Event()

        def classify(state, router):
            assert router is fake_router[0]
            inference_threads.add(threading.get_ident())
            calls.append(state["body"])
            if state["body"] == "first":
                loop.call_soon_threadsafe(inference_started.set)
                assert release.wait(timeout=3), "event loop could not release inference"
            return {"body": state["body"]}

        monkeypatch.setattr(simulation, "classify", classify)
        task = asyncio.create_task(simulation.simulate_requests(
            [{"body": body} for body in ("first", "second", "third")], queue_size=1
        ))
        try:
            await asyncio.wait_for(inference_started.wait(), timeout=3)
            # The event loop can run this while the inference thread is blocked.
            assert not task.done()
            release.set()
            return await asyncio.wait_for(task, timeout=3)
        finally:
            release.set()

    result = asyncio.run(scenario())
    assert len(fake_router) == 1
    assert set(fake_router[0].loads) == {"english"}
    assert calls[0] == "Routine notification. No action is needed."
    assert calls[1:] == ["first", "second", "third"]
    assert len(inference_threads) == 1
    assert main_thread not in inference_threads
    assert [item.answers["body"] for item in result.requests] == ["first", "second", "third"]
    for item in result.requests:
        assert item.wait_ms >= 0
        assert item.inference_ms >= 0
        assert item.total_ms == pytest.approx(item.wait_ms + item.inference_ms)


def test_failed_request_does_not_strand_later_requests(monkeypatch, fake_router):
    completed = []

    def classify(state, router):
        if state["body"] == "bad":
            raise MessageTooLongError("needs review")
        completed.append(state["body"])
        return {}

    monkeypatch.setattr(simulation, "classify", classify)

    async def scenario():
        return await asyncio.wait_for(simulation.simulate_requests(
            [{"body": "bad"}, {"body": "good"}], queue_size=1
        ), timeout=3)

    with pytest.raises(MessageTooLongError, match="needs review"):
        asyncio.run(scenario())
    assert completed[-1] == "good"


def test_startup_failure_is_reported(monkeypatch, fake_router):
    def fail(state, router):
        raise RuntimeError("warmup failed")

    monkeypatch.setattr(simulation, "classify", fail)
    with pytest.raises(RuntimeError, match="warmup failed"):
        asyncio.run(simulation.simulate_requests([{"body": "Hello"}]))


def test_queue_capacity_must_be_positive():
    with pytest.raises(ValueError, match="queue_size must be positive"):
        asyncio.run(simulation.simulate_requests([], queue_size=0))


@pytest.mark.parametrize("arguments, message", [
    (["--simulate-server", "--body", "Hello"], "--simulate-server requires --demo"),
    (["--demo", "--simulate-server", "--queue-size", "0"], "--queue-size must be positive"),
    (["--demo", "--repeats", "0"], "--repeats must be positive"),
    (["--demo", "--repeats", "-1"], "--repeats must be positive"),
    (["--body", "Hello", "--repeats", "2"], "--repeats requires --demo"),
])
def test_cli_validates_simulation_options(monkeypatch, capsys, arguments, message):
    from system_one_models.cli import main

    monkeypatch.setattr("sys.argv", ["system-one-models", *arguments])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("repeats", [1, 3])
def test_demo_simulation_reports_wait_and_latency(monkeypatch, capsys, repeats):
    from system_one_models import evaluation

    examples = evaluation.load_examples()[:2]

    async def simulate(states, *, queue_size, device, offline, cpu_threads):
        assert cpu_threads is None
        assert device == "auto"
        assert offline is False
        assert states == [example["state"] for example in examples] * repeats
        assert queue_size == 1
        responses = []
        for example in examples * repeats:
            answers = {
                key: {"noul": float(value)} if isinstance(value, bool) else {"choice": value}
                for key, value in example["expected"].items()
            }
            responses.append(simulation.RequestResult(answers, 10, 20, 30))
        return simulation.SimulationResult(responses, startup_ms=50, elapsed_ms=60)

    monkeypatch.setattr(evaluation, "simulate_requests", simulate)
    monkeypatch.setattr(evaluation, "load_examples", lambda: examples)
    assert evaluation.run_demo(simulate_server=True, queue_size=1, repeats=repeats) == 0
    output = capsys.readouterr().out
    assert "1 Router" in output
    assert "Startup + warmup: 50.0 ms" in output
    assert "Wait: 10.0 ms" in output
    assert "Total request latency: 30.0 ms" in output
    count = len(examples) * repeats
    assert f"exact matches {count}/{count}" in output
    assert output.count("Wait: 10.0 ms") == count
    if repeats > 1:
        comparisons = len(examples) * (repeats - 1)
        assert f"Repeat consistency: {comparisons}/{comparisons}" in output
        assert "repeat 3/3" in output
    assert "Request latency: p50 30.0 ms · p95 30.0 ms · max 30.0 ms" in output
    assert "excludes startup" in output



def test_repeated_sequential_demo_reuses_router_and_detects_label_changes(monkeypatch, capsys, fake_router):
    from system_one_models import evaluation

    example = evaluation.load_examples()[0]
    calls = []

    def classify(state, router):
        assert router is fake_router[0]
        assert state == example["state"]
        calls.append(state)
        answers = {
            key: {"noul": float(value)} if isinstance(value, bool) else {"choice": value}
            for key, value in example["expected"].items()
        }
        if len(calls) == 2:
            answers["classification"] = {"choice": "legitimate"}
        return answers

    monkeypatch.setattr(evaluation, "classify", classify)
    monkeypatch.setattr(evaluation, "load_examples", lambda: [example])
    assert evaluation.run_demo(repeats=3) == 0
    assert len(fake_router) == 1
    assert len(calls) == 3
    output = capsys.readouterr().out
    assert "3 requests (1 examples × 3 repeats)" in output
    assert "exact matches 2/3" in output
    assert "Repeat consistency: 1/2" in output
    assert "repeat 3/3" in output


def test_timing_percentiles_use_median_and_nearest_rank():
    from system_one_models.evaluation import _timing_summary

    assert _timing_summary("Wait", list(range(1, 21))) == (
        "  Wait: p50 10.5 ms · p95 19.0 ms · max 20.0 ms"
    )
    assert _timing_summary("Wait", [5]) == "  Wait: p50 5.0 ms · p95 5.0 ms · max 5.0 ms"


def test_simulation_forwards_device_and_offline_once(monkeypatch, fake_router):
    import laya

    options = []

    def create(*, device, offline, cpu_threads):
        assert cpu_threads == 4
        options.append((device, offline))
        return laya.Router(device=device)

    monkeypatch.setattr(simulation, "create_router", create)
    monkeypatch.setattr(simulation, "classify", lambda state, router: {})
    result = asyncio.run(simulation.simulate_requests(
        [{"body": "Hello"}, {"body": "Again"}], device="mps", offline=True, cpu_threads=4
    ))
    assert options == [("mps", True)]
    assert len(fake_router) == 1
    assert result.devices == {"english": "cpu"}
