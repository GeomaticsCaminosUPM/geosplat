"""Real spherical harmonics (degree <= 3) used for view-dependent color."""

from __future__ import annotations

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def num_sh_bases(degree: int) -> int:
    """Number of SH coefficients per color channel for a given degree.

    Args:
        degree: SH degree (0 to 3).

    Returns:
        ``(degree + 1) ** 2``.
    """
    return (degree + 1) ** 2


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB colors in ``[0, 1]`` to degree-0 SH coefficients.

    Args:
        rgb: Colors of shape ``[..., 3]``.

    Returns:
        DC coefficients of shape ``[..., 3]``.
    """
    return (rgb - 0.5) / C0


def sh_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    """Convert degree-0 SH coefficients back to RGB.

    Args:
        sh: DC coefficients of shape ``[..., 3]``.

    Returns:
        RGB colors of shape ``[..., 3]``.
    """
    return sh * C0 + 0.5


def eval_sh(degree: int, coeffs: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate real spherical harmonics (same convention as 3DGS / gsplat).

    Args:
        degree: Active SH degree (0 to 3).
        coeffs: SH coefficients of shape ``[..., K, 3]`` with
            ``K >= (degree + 1) ** 2``.
        dirs: Unit viewing directions of shape ``[..., 3]``.

    Returns:
        Colors of shape ``[..., 3]`` (before the ``+0.5`` offset).
    """
    result = C0 * coeffs[..., 0, :]
    if degree < 1:
        return result
    x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
    result = result - C1 * y * coeffs[..., 1, :] + C1 * z * coeffs[..., 2, :] - C1 * x * coeffs[..., 3, :]
    if degree < 2:
        return result
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    result = (
        result
        + C2[0] * xy * coeffs[..., 4, :]
        + C2[1] * yz * coeffs[..., 5, :]
        + C2[2] * (2.0 * zz - xx - yy) * coeffs[..., 6, :]
        + C2[3] * xz * coeffs[..., 7, :]
        + C2[4] * (xx - yy) * coeffs[..., 8, :]
    )
    if degree < 3:
        return result
    result = (
        result
        + C3[0] * y * (3 * xx - yy) * coeffs[..., 9, :]
        + C3[1] * xy * z * coeffs[..., 10, :]
        + C3[2] * y * (4 * zz - xx - yy) * coeffs[..., 11, :]
        + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * coeffs[..., 12, :]
        + C3[4] * x * (4 * zz - xx - yy) * coeffs[..., 13, :]
        + C3[5] * z * (xx - yy) * coeffs[..., 14, :]
        + C3[6] * x * (xx - 3 * yy) * coeffs[..., 15, :]
    )
    return result
