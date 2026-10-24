"""Environment diagnostics, with optional heavyweight imports deferred until use."""

import platform
from importlib.metadata import PackageNotFoundError, version


def inspect_environment(device="auto"):
    import torch

    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    available = torch.cuda.is_available()
    if device == "cuda" and not available:
        raise RuntimeError("CUDA was requested but is not available to PyTorch")
    selected = "cuda" if device == "auto" and available else device
    if selected == "auto":
        selected = "cpu"
    packages = {}
    for name in ("torch", "transformers", "accelerate", "pandas", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "selected_device": selected,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": available,
        "cuda_device_count": torch.cuda.device_count() if available else 0,
        "gpu": torch.cuda.get_device_name() if available else None,
        "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        if available
        else None,
        "bf16_supported": torch.cuda.is_bf16_supported() if available else False,
    }
