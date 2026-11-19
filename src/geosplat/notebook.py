"""Notebook helpers: one-call environment setup for Jupyter and Google Colab."""

from __future__ import annotations

import importlib.metadata
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = "GeomaticsCaminosUPM/geosplat"


def on_colab() -> bool:
    """Whether the code runs in Google Colab.

    Returns:
        ``True`` on Colab.
    """
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def pip_command() -> list[str]:
    """Command that installs packages into the running interpreter (``uv pip`` if available).

    Returns:
        Command prefix, e.g. ``["uv", "pip", "install", "--python", sys.executable]``.
    """
    if shutil.which("uv"):
        return ["uv", "pip", "install", "--python", sys.executable]
    return [sys.executable, "-m", "pip", "install"]


def gsplat_compiled() -> bool:
    """Whether gsplat's CUDA code is already available (prebuilt wheel or cached JIT build).

    Returns:
        ``True`` if no compilation is needed.
    """
    from torch.utils.cpp_extension import _get_build_directory

    files = importlib.metadata.files("gsplat") or []
    if any("csrc" in str(f) and str(f).endswith(".so") for f in files):
        return True
    return Path(_get_build_directory("gsplat_cuda", verbose=False), "gsplat_cuda.so").exists()


def install_prebuilt_gsplat(repo: str = REPO) -> bool:
    """Install a prebuilt gsplat CUDA wheel from the repository's GitHub releases.

    The release tag is ``gsplat-<ver>-torch<torch>-cu<cuda>-cp<py>`` (built by
    ``.github/workflows/gsplat-wheels.yml``). If none matches, prints the
    workflow inputs needed to build it.

    Args:
        repo: GitHub ``owner/name`` hosting the releases.

    Returns:
        ``True`` if a wheel was installed.
    """
    import torch

    gv, tv = importlib.metadata.version("gsplat"), torch.__version__.split("+")[0]
    cu, py = (torch.version.cuda or "").replace(".", ""), f"cp{sys.version_info.major}{sys.version_info.minor}"
    tag = f"gsplat-{gv}-torch{tv}-cu{cu}-{py}"
    wheel = f"https://github.com/{repo}/releases/download/{tag}/gsplat-{gv}-{py}-{py}-linux_x86_64.whl"
    cmd = pip_command() + ["-q", "--no-deps", "--force-reinstall", wheel]
    if subprocess.run(cmd, capture_output=True).returncode == 0:
        print("Installed prebuilt gsplat CUDA wheel:", tag)
        return True
    print(f"No prebuilt gsplat wheel for {tag}: CUDA code will be compiled once (3-5 min).\n"
          f"To prebuild it, run the 'gsplat wheels' workflow on GitHub (Actions tab) with python="
          f"{sys.version_info.major}.{sys.version_info.minor}, torch={tv}, cuda={torch.version.cuda}, gsplat={gv}.")
    return False


def setup(repo: str = REPO, prebuilt_gsplat: bool = True) -> dict[str, bool]:
    """Prepare a notebook session and check that every stage runs on the GPU.

    * Colab: enables interactive map widgets (hiding Colab's notice) and, if
      gsplat isn't compiled yet, installs a prebuilt gsplat wheel.
    * Inside a clone of the repository: makes the repository root the working
      directory, so ``examples/...`` paths work.
    * Configures logging and prints where geosplat was loaded from.

    Args:
        repo: GitHub ``owner/name`` with the prebuilt gsplat wheels.
        prebuilt_gsplat: Try the prebuilt gsplat wheel on Colab.

    Returns:
        ``{stage: runs_on_gpu}`` for torch, COLMAP and gsplat.
    """
    import pycolmap
    import torch

    import geosplat

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    if on_colab():
        from google.colab import output
        from IPython.display import clear_output

        output.enable_custom_widget_manager()      # interactive maps (ipyleaflet)
        clear_output()                             # hide Colab's "Third-party Jupyter widgets" notice
        if prebuilt_gsplat and torch.cuda.is_available() and not gsplat_compiled():
            install_prebuilt_gsplat(repo)

    pkg = Path(geosplat.__file__).resolve().parent
    root = pkg.parents[1]
    if (root / "examples").is_dir() and (root / "pyproject.toml").is_file():
        os.chdir(root)
    newest = max(f.stat().st_mtime for f in pkg.glob("*.py"))
    print(f"geosplat {geosplat.__version__} from {pkg} "
          f"(code modified {time.strftime('%Y-%m-%d %H:%M', time.localtime(newest))}) | working dir: {Path.cwd()}")

    gpu = {
        "torch (training, orthophotos)": torch.cuda.is_available(),
        "COLMAP (features, matching, BA)": bool(getattr(pycolmap, "has_cuda", False)),
        "gsplat rasterizer": geosplat.gsplat_available(),   # first time without a wheel: compiles (3-5 min)
    }
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    for stage, ok in gpu.items():
        print(f"  {'GPU' if ok else 'CPU!':4}  {stage}")
    if not all(gpu.values()):
        print("Some stages would run on the CPU (very slow). Colab: Runtime > Change runtime type > GPU. "
              "Local: NVIDIA driver + CUDA toolkit (nvcc) and a CUDA build of torch (uv sync does this).")
    return gpu
