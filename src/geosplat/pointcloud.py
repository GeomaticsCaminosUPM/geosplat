"""Colored point cloud export (sparse SfM points and dense points from the splat)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from .geo import GeoReference
from .scene import Scene
from .sh import sh_to_rgb

logger = logging.getLogger("geosplat")


def save_point_cloud(path: str | Path, xyz: np.ndarray, rgb: np.ndarray, double: bool = False) -> Path:
    """Write a colored point cloud as binary PLY.

    Args:
        path: Output ``.ply`` path.
        xyz: Coordinates ``[N, 3]``.
        rgb: Colors ``[N, 3]`` (``uint8``).
        double: Store coordinates as float64 (needed for projected CRS
            coordinates such as UTM, which lose precision in float32).

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ft = "f8" if double else "f4"
    v = np.empty(len(xyz), dtype=[("x", ft), ("y", ft), ("z", ft), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    v["red"], v["green"], v["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(v, "vertex")]).write(str(path))
    return path


@torch.no_grad()
def otsu_threshold(values: torch.Tensor, bins: int = 256, lo: float = 0.05, hi: float = 0.5) -> float:
    """Otsu's threshold of values in ``[0, 1]`` (maximizes between-class variance).

    Args:
        values: 1D tensor with values in ``[0, 1]`` (e.g. Gaussian opacities).
        bins: Histogram bins.
        lo: Lower clamp for the result.
        hi: Upper clamp for the result.

    Returns:
        The threshold, clamped to ``[lo, hi]``.
    """
    hist = torch.histc(values.float(), bins=bins, min=0.0, max=1.0)
    p = hist / hist.sum().clamp_min(1)
    centers = (torch.arange(bins, device=values.device, dtype=torch.float32) + 0.5) / bins
    w0 = torch.cumsum(p, 0)
    mu = torch.cumsum(p * centers, 0)
    between = (mu[-1] * w0 - mu) ** 2 / (w0 * (1 - w0)).clamp_min(1e-12)
    t = float(centers[int(torch.argmax(between[:-1]))])
    return float(min(max(t, lo), hi))


@torch.no_grad()
def dense_points_from_splat(
    params: torch.nn.ParameterDict | dict[str, torch.Tensor],
    scene: Scene,
    min_opacity: float | str = "auto",
    max_scale_quantile: float = 0.95,
    max_radius: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a dense colored point cloud from the centers of solid Gaussians.

    Args:
        params: Gaussian parameters (scene coordinates).
        scene: Scene (used for its center and scale).
        min_opacity: Keep Gaussians at least this opaque, or ``"auto"`` to pick
            the threshold with Otsu's method on the opacity histogram.
        max_scale_quantile: Drop the largest Gaussians (blurry background
            blobs) above this quantile of the largest axis.
        max_radius: Drop points farther than ``max_radius * scene_scale`` from
            the scene center (removes the background sphere).

    Returns:
        Tuple ``(xyz [N, 3] float64 in scene coordinates, rgb [N, 3] uint8)``.
    """
    means = params["means"].detach()
    opacity = torch.sigmoid(params["opacities"].detach())
    scale = torch.exp(params["scales"].detach()).max(dim=1).values
    center = torch.as_tensor(
        np.median(scene.points, axis=0) if len(scene.points) else np.zeros(3), dtype=means.dtype, device=means.device
    )
    inside = (means - center).norm(dim=1) < max_radius * scene.scene_scale
    if min_opacity == "auto":
        min_opacity = otsu_threshold(opacity[inside]) if inside.any() else 0.2
        logger.info("Dense point cloud: automatic opacity threshold %.2f", min_opacity)
    keep = (opacity >= float(min_opacity)) & inside
    if keep.any():
        keep &= scale <= torch.quantile(scale[keep].float()[:1_000_000], max_scale_quantile)
    rgb = (sh_to_rgb(params["sh0"].detach()[keep, 0]).clamp(0, 1) * 255).round().byte()
    return means[keep].double().cpu().numpy(), rgb.cpu().numpy()


def export_point_clouds(
    params: torch.nn.ParameterDict | dict[str, torch.Tensor],
    scene: Scene,
    output_dir: str | Path,
    geo: GeoReference | None = None,
    min_opacity: float | str = "auto",
) -> dict[str, Path]:
    """Write ``points_sparse.ply`` (SfM) and ``points_dense.ply`` (from the splat).

    With a georeference, coordinates are written in the CRS (float64), so the
    clouds overlay directly with the orthophoto/DSM in GIS or CloudCompare.

    Args:
        params: Trained Gaussian parameters.
        scene: Scene with the sparse points.
        output_dir: Output folder.
        geo: Optional georeference.
        min_opacity: Minimum opacity of Gaussians kept in ``points_dense.ply``
            (``"auto"``: Otsu threshold).

    Returns:
        Mapping ``{"points_sparse": path, "points_dense": path}``.
    """
    out = Path(output_dir)

    def to_out(xyz: np.ndarray) -> np.ndarray:
        """Convert scene coordinates to output coordinates.

        Args:
            xyz: Points ``[N, 3]`` in scene coordinates.

        Returns:
            Points in CRS coordinates if georeferenced, else unchanged.
        """
        return geo.scene_to_local(xyz.astype(np.float64)) + geo.origin if geo else xyz

    dense_xyz, dense_rgb = dense_points_from_splat(params, scene, min_opacity=min_opacity)
    return {
        "points_sparse": save_point_cloud(
            out / "points_sparse.ply", to_out(scene.points), scene.colors, double=geo is not None
        ),
        "points_dense": save_point_cloud(out / "points_dense.ply", to_out(dense_xyz), dense_rgb, double=geo is not None),
    }
