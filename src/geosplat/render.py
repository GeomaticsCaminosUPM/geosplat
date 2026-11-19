"""Differentiable Gaussian rasterization.

Two backends are provided:

* ``"gsplat"``: fast tile-based CUDA kernels from the gsplat library (GPU).
* ``"torch"``: a pure-PyTorch reference rasterizer that runs anywhere (CPU or
  GPU). It is orders of magnitude slower and meant for small scenes, testing,
  or machines without CUDA.

Both return the same ``info`` keys required by
:class:`gsplat.strategy.DefaultStrategy` for densification.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .sh import eval_sh


def quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    """Convert (unnormalized) ``wxyz`` quaternions to rotation matrices.

    Args:
        quats: Quaternions of shape ``[N, 4]``.

    Returns:
        Rotation matrices of shape ``[N, 3, 3]``.
    """
    w, x, y, z = F.normalize(quats, dim=-1).unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


def render(
    params: torch.nn.ParameterDict | dict[str, torch.Tensor],
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    sh_degree: int,
    backend: str = "gsplat",
    background: torch.Tensor | None = None,
    absgrad: bool = False,
    camera_model: str = "pinhole",
    render_mode: str = "RGB",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Render Gaussians from one or more cameras.

    Args:
        params: Gaussian parameters (``means``, ``scales``, ``quats``,
            ``opacities``, ``sh0``, ``shN``) in unconstrained form.
        viewmats: World-to-camera matrices of shape ``[C, 4, 4]``.
        Ks: Intrinsics of shape ``[C, 3, 3]``.
        width: Image width in pixels.
        height: Image height in pixels.
        sh_degree: Active SH degree used for color evaluation.
        backend: ``"gsplat"`` or ``"torch"``.
        background: Optional background color of shape ``[3]``.
        absgrad: Ask gsplat to accumulate absolute 2D gradients.
        camera_model: ``"pinhole"`` or ``"ortho"`` (gsplat backend only). For
            ``"ortho"``, pixels are ``u = fx * x + cx`` in camera coordinates.
        render_mode: ``"RGB"``, ``"D"`` / ``"ED"`` (accumulated / expected depth)
            or ``"RGB+D"`` / ``"RGB+ED"`` (gsplat backend only).

    Returns:
        Tuple ``(image [C, H, W, 3 or 4 or 1], alpha [C, H, W, 1], info)``.
        Color channels are clamped to ``[0, 1]``; a depth channel, if requested,
        comes last.

    Raises:
        ValueError: If ``backend`` is unknown or an option is unsupported by
            the torch backend.
    """
    means = params["means"]
    quats = params["quats"]
    scales = torch.exp(params["scales"])
    opacities = torch.sigmoid(params["opacities"])
    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    C = viewmats.shape[0]
    backgrounds = None if background is None else background[None].expand(C, 3)

    if backend == "gsplat":
        from gsplat import rasterization

        rgb, alpha, info = rasterization(
            means, quats, scales, opacities, colors, viewmats, Ks, width, height,
            sh_degree=sh_degree, packed=False, absgrad=absgrad, backgrounds=backgrounds,
            camera_model=camera_model, render_mode=render_mode,
        )
        if render_mode.startswith("RGB"):
            rgb = torch.cat([rgb[..., :3].clamp(0.0, 1.0), rgb[..., 3:]], dim=-1)
        return rgb, alpha, info
    if backend == "torch":
        if camera_model != "pinhole" or render_mode != "RGB":
            raise ValueError("The torch backend only supports pinhole cameras and RGB rendering.")
        return _render_torch(
            means, quats, scales, opacities, colors, viewmats, Ks, width, height, sh_degree, backgrounds
        )
    raise ValueError(f"Unknown backend {backend!r}")


def _shade(p: torch.Tensor, mu: torch.Tensor, con: torch.Tensor, op: torch.Tensor,
           col: torch.Tensor) -> torch.Tensor:
    """Alpha-composite depth-sorted Gaussians for a set of pixels.

    Args:
        p: Pixel centers ``[P, 2]``.
        mu: Depth-sorted 2D means ``[M, 2]``.
        con: Depth-sorted conics (inverse 2D covariances) ``[M, 3]``.
        op: Depth-sorted opacities ``[M]``.
        col: Depth-sorted colors ``[M, 3]``.

    Returns:
        ``[P, 4]`` tensor with RGB and accumulated alpha.
    """
    dx = p[:, None, :] - mu[None]
    power = (
        -0.5 * (con[:, 0] * dx[..., 0] ** 2 + con[:, 2] * dx[..., 1] ** 2)
        - con[:, 1] * dx[..., 0] * dx[..., 1]
    )
    alpha = (op * torch.exp(power.clamp_max(0.0))).clamp_max(0.99)
    alpha = torch.where(alpha < 1.0 / 255.0, torch.zeros_like(alpha), alpha)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha[:, :-1]], 1), 1)
    w = alpha * trans
    return torch.cat([w @ col, w.sum(1, keepdim=True)], dim=1)


def _render_torch(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    sh_degree: int,
    backgrounds: torch.Tensor | None,
    near: float = 0.01,
    eps2d: float = 0.3,
    tile_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Pure-PyTorch tile-based EWA splatting for any device.

    Args:
        means: Gaussian centers ``[N, 3]``.
        quats: Rotations ``[N, 4]`` (``wxyz``).
        scales: Activated scales ``[N, 3]``.
        opacities: Activated opacities ``[N]``.
        colors: SH coefficients ``[N, K, 3]``.
        viewmats: World-to-camera matrices ``[C, 4, 4]``.
        Ks: Intrinsics ``[C, 3, 3]``.
        width: Image width.
        height: Image height.
        sh_degree: Active SH degree.
        backgrounds: Optional backgrounds ``[C, 3]``.
        near: Near plane distance.
        eps2d: Low-pass filter added to 2D covariances (matches gsplat).
        tile_size: Side of the square pixel tiles; each tile only composites the
            Gaussians overlapping it.

    Returns:
        Tuple ``(rgb [C, H, W, 3], alpha [C, H, W, 1], info)``.
    """
    device = means.device
    N = means.shape[0]
    C = viewmats.shape[0]
    R_g = quat_to_rotmat(quats)
    M = R_g * scales[:, None, :]
    cov3d = M @ M.transpose(1, 2)  # [N, 3, 3]

    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=means.dtype) + 0.5,
        torch.arange(width, device=device, dtype=means.dtype) + 0.5,
        indexing="ij",
    )
    pix = torch.stack([xs, ys], dim=-1)  # [H, W, 2]

    rgbs, alphas, means2d_all, radii_all = [], [], [], []
    geoms: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for c in range(C):
        R, t = viewmats[c, :3, :3], viewmats[c, :3, 3]
        fx, fy, cx, cy = Ks[c, 0, 0], Ks[c, 1, 1], Ks[c, 0, 2], Ks[c, 1, 2]
        p_cam = means @ R.T + t
        z = p_cam[:, 2]
        valid_z = z > near
        zs = z.clamp_min(near)
        # Clamp the projection used in the Jacobian as in 3DGS for stability.
        lim_x = 1.3 * 0.5 * width / fx
        lim_y = 1.3 * 0.5 * height / fy
        tx = (p_cam[:, 0] / zs).clamp(-lim_x - cx / fx, lim_x + (width - cx) / fx) * zs
        ty = (p_cam[:, 1] / zs).clamp(-lim_y - cy / fy, lim_y + (height - cy) / fy) * zs
        J = torch.zeros(N, 2, 3, device=device, dtype=means.dtype)
        J[:, 0, 0] = fx / zs
        J[:, 0, 2] = -fx * tx / zs**2
        J[:, 1, 1] = fy / zs
        J[:, 1, 2] = -fy * ty / zs**2
        T = J @ R
        cov2d = T @ cov3d @ T.transpose(1, 2)
        cov2d = cov2d + eps2d * torch.eye(2, device=device, dtype=means.dtype)
        means2d = torch.stack([fx * p_cam[:, 0] / zs + cx, fy * p_cam[:, 1] / zs + cy], dim=-1)

        a, b, d = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
        det = (a * d - b * b).clamp_min(1e-12)
        conic = torch.stack([d / det, -b / det, a / det], dim=-1)
        with torch.no_grad():
            rx = torch.ceil(3.0 * a.sqrt())
            ry = torch.ceil(3.0 * d.sqrt())
            on_screen = (
                (means2d[:, 0] + rx > 0) & (means2d[:, 0] - rx < width)
                & (means2d[:, 1] + ry > 0) & (means2d[:, 1] - ry < height)
            )
            visible = valid_z & on_screen
            radii = torch.stack([rx, ry], dim=-1) * visible[:, None]

        dirs = F.normalize(means - (-R.T @ t), dim=-1)
        rgb_g = (eval_sh(sh_degree, colors, dirs) + 0.5).clamp_min(0.0)
        geoms.append((conic, visible, z, rgb_g))
        means2d_all.append(means2d)
        radii_all.append(radii.to(torch.int32))

    # Rendering must consume the *stacked* tensor so that its .grad (used by the
    # densification strategy via retain_grad) is populated.
    means2d_stacked = torch.stack(means2d_all)
    for c, (conic, visible, z, rgb_g) in enumerate(geoms):
        idx = torch.nonzero(visible, as_tuple=False).squeeze(1)
        idx = idx[torch.argsort(z[idx])]
        mu, con, op, col = means2d_stacked[c][idx], conic[idx], opacities[idx], rgb_g[idx]
        r = radii_all[c][idx].to(means.dtype)
        mu_d = mu.detach()
        rows = []
        for y0 in range(0, height, tile_size):
            y1 = min(y0 + tile_size, height)
            row = []
            for x0 in range(0, width, tile_size):
                x1 = min(x0 + tile_size, width)
                # Gaussians whose 3-sigma box overlaps this tile (kept in depth order).
                hit = (
                    (mu_d[:, 0] + r[:, 0] > x0) & (mu_d[:, 0] - r[:, 0] < x1)
                    & (mu_d[:, 1] + r[:, 1] > y0) & (mu_d[:, 1] - r[:, 1] < y1)
                )
                sel = torch.nonzero(hit, as_tuple=False).squeeze(1)
                p = pix[y0:y1, x0:x1].reshape(-1, 2)
                if sel.numel() == 0:
                    out = torch.zeros(p.shape[0], 4, device=device, dtype=means.dtype)
                elif torch.is_grad_enabled():
                    out = checkpoint(_shade, p, mu[sel], con[sel], op[sel], col[sel], use_reentrant=False)
                else:
                    out = _shade(p, mu[sel], con[sel], op[sel], col[sel])
                row.append(out.reshape(y1 - y0, x1 - x0, 4))
            rows.append(torch.cat(row, dim=1))
        img = torch.cat(rows, dim=0)
        rgb, alpha = img[..., :3], img[..., 3:]
        if backgrounds is not None:
            rgb = rgb + (1 - alpha) * backgrounds[c]
        rgbs.append(rgb.clamp(0.0, 1.0))
        alphas.append(alpha)

    info = {
        "means2d": means2d_stacked,
        "radii": torch.stack(radii_all),
        "width": width,
        "height": height,
        "n_cameras": C,
        "gaussian_ids": None,  # only used in packed mode
    }
    return torch.stack(rgbs), torch.stack(alphas), info
