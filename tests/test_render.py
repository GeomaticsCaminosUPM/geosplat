"""Tests for rasterization backends and PLY round-trips."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from geosplat.gaussians import init_gaussians, load_ply, save_ply
from geosplat.render import render
from geosplat.utils import gsplat_available


def _random_params(n: int = 200, device: str = "cpu") -> torch.nn.ParameterDict:
    """Create random Gaussians in front of an identity camera.

    Args:
        n: Number of Gaussians.
        device: Torch device.

    Returns:
        Gaussian parameters.
    """
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(n, 3)).astype(np.float32) * 0.5 + np.array([0, 0, 4], np.float32)
    cols = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
    params = init_gaussians(pts, cols, sh_degree=1, init_opacity=0.5, device=device)
    with torch.no_grad():
        params["shN"].normal_(0, 0.1)
        params["quats"].normal_()
    return params


def _camera(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Identity camera with a 64x48 image.

    Args:
        device: Torch device.

    Returns:
        Tuple ``(viewmats, Ks)``.
    """
    viewmats = torch.eye(4, device=device)[None]
    Ks = torch.tensor([[[60.0, 0, 32], [0, 60.0, 24], [0, 0, 1]]], device=device)
    return viewmats, Ks


def test_torch_backend_gradients() -> None:
    """The torch backend renders and propagates gradients to params and means2d."""
    params = _random_params()
    viewmats, Ks = _camera("cpu")
    rgb, alpha, info = render(params, viewmats, Ks, 64, 48, sh_degree=1, backend="torch")
    assert rgb.shape == (1, 48, 64, 3) and alpha.shape == (1, 48, 64, 1)
    info["means2d"].retain_grad()
    rgb.mean().backward()
    assert info["means2d"].grad is not None and info["means2d"].grad.abs().sum() > 0
    assert params["means"].grad is not None and torch.isfinite(params["means"].grad).all()
    assert info["radii"].shape == (1, 200, 2)


@pytest.mark.skipif(not gsplat_available(), reason="gsplat CUDA backend unavailable")
def test_backends_agree() -> None:
    """The torch fallback matches gsplat on the same scene."""
    params = _random_params(device="cuda")
    viewmats, Ks = _camera("cuda")
    with torch.no_grad():
        a, _, _ = render(params, viewmats, Ks, 64, 48, sh_degree=1, backend="gsplat")
        b, _, _ = render(params, viewmats, Ks, 64, 48, sh_degree=1, backend="torch")
    assert (a - b).abs().mean().item() < 1e-2


def test_ply_roundtrip(tmp_path: Path) -> None:
    """Saving and loading a PLY preserves every parameter.

    Args:
        tmp_path: Pytest temporary directory.
    """
    params = _random_params()
    loaded = load_ply(save_ply(params, tmp_path / "x.ply"))
    for key in params:
        ref = params[key].detach()
        if key == "quats":
            ref = torch.nn.functional.normalize(ref, dim=-1)
        assert torch.allclose(ref, loaded[key], atol=1e-6), key


def test_fast_ssim_matches_dense() -> None:
    """The separable batched SSIM equals a dense 11x11 Gaussian-window SSIM."""
    import torch.nn.functional as F

    from geosplat.trainer import _gaussian_1d, ssim

    torch.manual_seed(0)
    a, b = torch.rand(1, 3, 40, 50), torch.rand(1, 3, 40, 50)
    g = _gaussian_1d(11, 1.5, a.device, a.dtype)
    w = (g[:, None] @ g[None]).expand(3, 1, 11, 11)

    def blur(x: torch.Tensor) -> torch.Tensor:
        """Dense Gaussian blur per channel.

        Args:
            x: Image batch ``[1, 3, H, W]``.

        Returns:
            Blurred batch.
        """
        return F.conv2d(x, w, padding=5, groups=3)

    mu1, mu2 = blur(a), blur(b)
    s1, s2, s12 = blur(a * a) - mu1**2, blur(b * b) - mu2**2, blur(a * b) - mu1 * mu2
    c1, c2 = 0.01**2, 0.03**2
    ref = (((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1**2 + mu2**2 + c1) * (s1 + s2 + c2))).mean()
    assert torch.allclose(ssim(a, b), ref, atol=1e-5)
