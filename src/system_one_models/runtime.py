"""Router configuration, cache-only loading, and actual device reporting."""

from __future__ import annotations

from typing import Any

DEVICES = ("auto", "cpu", "mps", "cuda")
CHECKPOINT_FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")


class OfflineCacheError(RuntimeError):
    """Required checkpoint files are not available locally."""


def get_cpu_threads() -> int:
    import torch

    return torch.get_num_threads()


def create_router(
    *, device: str = "auto", offline: bool = False, cpu_threads: int | None = None
) -> Any:
    if device not in DEVICES:
        raise ValueError(f"device must be one of {', '.join(DEVICES)}")
    if cpu_threads is not None:
        if cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        import torch

        # Configure CPU kernels before checkpoint loading or inference. In the
        # simulation this runs inside the same thread that performs inference.
        torch.set_num_threads(cpu_threads)
    from laya import Router

    options = {}
    if offline:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import IncompleteSnapshotError, LocalEntryNotFoundError
        from laya.router import DEFAULT_MODELS

        # Laya's default checkpoints share a bundle. Resolve its cached root once,
        # and preserve the subfolder for each checkpoint. Only inference files are
        # required; uncached README/assets/sibling checkpoints are not downloads.
        snapshots = {}
        models = {}
        for name, (repository, subfolder) in DEFAULT_MODELS.items():
            if repository not in snapshots:
                try:
                    snapshots[repository] = snapshot_download(
                        repository, local_files_only=True, allow_patterns=list(CHECKPOINT_FILES)
                    )
                except (IncompleteSnapshotError, LocalEntryNotFoundError) as error:
                    raise OfflineCacheError(
                        "Required Laya checkpoint files are missing from the local cache. "
                        "Run once without --offline to download them."
                    ) from error
            models[name] = (snapshots[repository], subfolder)
        options["models"] = models
    return Router(device=None if device == "auto" else device, **options)


def loaded_devices(router: Any) -> dict[str, str]:
    """Read each loaded agent's actual device, including CPU fallback."""
    return {name: str(router.load(name).device) for name in router.loaded}


def device_summary(devices: dict[str, str], *, requested: str) -> str:
    actual = ", ".join(f"{name}={device}" for name, device in devices.items()) or "no checkpoint loaded"
    return f"Inference device: {actual} (requested: {requested})"
