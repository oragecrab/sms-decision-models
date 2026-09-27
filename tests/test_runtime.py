from __future__ import annotations

from types import SimpleNamespace

import pytest

from system_one_models.runtime import (
    CHECKPOINT_FILES,
    OfflineCacheError,
    create_router,
    device_summary,
    loaded_devices,
)


@pytest.mark.parametrize("device, resolved", [("auto", None), ("cpu", "cpu"), ("mps", "mps"), ("cuda", "cuda")])
def test_device_selection_reaches_router(monkeypatch, device, resolved):
    import laya

    monkeypatch.setattr(laya, "Router", lambda **kwargs: kwargs)
    assert create_router(device=device) == {"device": resolved}


def test_offline_resolves_one_cached_bundle_without_network(monkeypatch):
    import huggingface_hub
    import laya
    from laya.router import DEFAULT_MODELS

    calls = []

    def snapshot(repository, **kwargs):
        calls.append((repository, kwargs))
        assert kwargs["local_files_only"] is True
        assert kwargs["allow_patterns"] == list(CHECKPOINT_FILES)
        return "/cached/checkpoint"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(laya, "Router", lambda **kwargs: kwargs)
    router = create_router(device="cpu", offline=True)
    assert len(calls) == 1
    assert router["device"] == "cpu"
    assert router["models"] == {
        name: ("/cached/checkpoint", subfolder)
        for name, (_, subfolder) in DEFAULT_MODELS.items()
    }


def test_missing_offline_cache_gives_actionable_error(monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    def missing(*args, **kwargs):
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    with pytest.raises(OfflineCacheError, match="Run once without --offline"):
        create_router(offline=True)


def test_reporting_uses_actual_device_after_fallback():
    router = SimpleNamespace(loaded=["english"], load=lambda name: SimpleNamespace(device="cpu"))
    assert device_summary(loaded_devices(router), requested="mps") == (
        "Inference device: english=cpu (requested: mps)"
    )


def test_cli_forwards_device_and_offline_to_demo(monkeypatch):
    from system_one_models import cli

    received = {}

    def demo(**kwargs):
        received.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "run_demo", demo)
    monkeypatch.setattr("sys.argv", ["system-one-models", "--demo", "--device", "cpu", "--offline"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 0
    assert received["device"] == "cpu"
    assert received["offline"] is True


def test_cli_single_message_uses_configured_router(monkeypatch, capsys):
    from system_one_models import cli

    router = SimpleNamespace(loaded=["english"], load=lambda name: SimpleNamespace(device="cpu"))

    def create(*, device, offline, cpu_threads):
        assert cpu_threads is None
        assert device == "mps"
        assert offline is True
        return router

    def classify(state, supplied_router):
        assert supplied_router is router
        return {}

    monkeypatch.setattr(cli, "create_router", create)
    monkeypatch.setattr(cli, "classify", classify)
    monkeypatch.setattr("sys.argv", ["system-one-models", "--body", "Hello", "--device", "mps", "--offline"])
    cli.main()
    assert "english=cpu (requested: mps)" in capsys.readouterr().out


def test_cpu_thread_setting_precedes_router_creation(monkeypatch):
    import torch
    import laya

    events = []
    monkeypatch.setattr(torch, "set_num_threads", lambda count: events.append(("threads", count)))
    monkeypatch.setattr(laya, "Router", lambda **kwargs: events.append(("router", kwargs)))
    create_router(device="cpu", cpu_threads=4)
    assert events == [("threads", 4), ("router", {"device": "cpu"})]


def test_default_preserves_existing_cpu_threads(monkeypatch):
    import torch
    import laya

    def unexpected(count):
        pytest.fail("default must not override the user's thread setting")

    monkeypatch.setattr(torch, "set_num_threads", unexpected)
    monkeypatch.setattr(laya, "Router", lambda **kwargs: kwargs)
    create_router()


@pytest.mark.parametrize("count", [0, -1])
def test_nonpositive_cpu_threads_are_rejected(count):
    with pytest.raises(ValueError, match="cpu_threads must be positive"):
        create_router(cpu_threads=count)


def test_cli_forwards_cpu_threads_to_demo(monkeypatch):
    from system_one_models import cli

    received = {}

    def demo(**kwargs):
        received.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "run_demo", demo)
    monkeypatch.setattr("sys.argv", ["system-one-models", "--demo", "--device", "cpu", "--cpu-threads", "4"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 0
    assert received["cpu_threads"] == 4
