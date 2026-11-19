"""General helpers: device selection, seeding and backend detection."""

from __future__ import annotations

import functools
import logging
import os
import random
import shutil

import numpy as np
import torch

logger = logging.getLogger("geosplat")


def get_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve the torch device to use.

    Args:
        device: Explicit device (e.g. ``"cuda"``, ``"cuda:1"``, ``"cpu"``). If
            ``None``, CUDA is used when available and CPU otherwise.

    Returns:
        The resolved :class:`torch.device`.
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch random number generators.

    Args:
        seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_cuda_home() -> None:
    """Point ``CUDA_HOME`` at a CUDA toolkit if it is unset.

    gsplat compiles its CUDA kernels just-in-time on first use and needs
    ``nvcc``. This looks for ``nvcc`` on ``PATH`` and in ``/usr/local/cuda*``.
    """
    if os.environ.get("CUDA_HOME"):
        return
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        for candidate in sorted(
            (p for p in os.listdir("/usr/local") if p.startswith("cuda")), reverse=True
        ) if os.path.isdir("/usr/local") else []:
            path = os.path.join("/usr/local", candidate, "bin", "nvcc")
            if os.path.exists(path):
                nvcc = path
                break
    if nvcc is not None:
        cuda_home = os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
        os.environ["CUDA_HOME"] = cuda_home
        os.environ["PATH"] = os.path.join(cuda_home, "bin") + os.pathsep + os.environ.get("PATH", "")


@functools.lru_cache(maxsize=None)
def gsplat_available(device_index: int = 0) -> bool:
    """Check whether the gsplat CUDA rasterizer can run on this machine.

    The first call may take several minutes because gsplat compiles its CUDA
    extension just-in-time. The result is cached for the process.

    Args:
        device_index: CUDA device index used for the smoke test.

    Returns:
        ``True`` if a tiny rasterization succeeds on the GPU, ``False`` otherwise.
    """
    if not torch.cuda.is_available():
        return False
    _ensure_cuda_home()
    try:
        from gsplat import rasterization

        dev = torch.device("cuda", device_index)
        means = torch.tensor([[0.0, 0.0, 3.0]], device=dev)
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev)
        scales = torch.full((1, 3), 0.1, device=dev)
        opacities = torch.ones(1, device=dev)
        colors = torch.ones(1, 3, device=dev)
        viewmats = torch.eye(4, device=dev)[None]
        Ks = torch.tensor([[[16.0, 0, 8], [0, 16.0, 8], [0, 0, 1]]], device=dev)
        logger.info("Checking gsplat CUDA backend (first run compiles kernels, may take minutes)...")
        rasterization(means, quats, scales, opacities, colors, viewmats, Ks, 16, 16)
        torch.cuda.synchronize(dev)
        return True
    except Exception as exc:  # noqa: BLE001 - any failure means "fall back"
        logger.warning("gsplat CUDA backend unavailable (%s); using the PyTorch fallback.", exc)
        return False


def resolve_backend(backend: str, device: torch.device) -> str:
    """Pick the rasterization backend.

    Args:
        backend: One of ``"auto"``, ``"gsplat"`` or ``"torch"``.
        device: Device training will run on.

    Returns:
        ``"gsplat"`` or ``"torch"``.

    Raises:
        ValueError: If ``backend`` is unknown.
        RuntimeError: If ``"gsplat"`` is requested but cannot be used.
    """
    if backend not in ("auto", "gsplat", "torch"):
        raise ValueError(f"Unknown backend {backend!r}; expected 'auto', 'gsplat' or 'torch'.")
    if backend == "torch":
        return "torch"
    ok = device.type == "cuda" and gsplat_available(device.index or 0)
    if backend == "gsplat" and not ok:
        raise RuntimeError("The gsplat backend requires a working CUDA GPU and toolkit.")
    return "gsplat" if ok else "torch"
