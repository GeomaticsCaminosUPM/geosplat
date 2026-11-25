"""Colored point cloud export (sparse SfM points and dense points from the splat)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
@torch.no_grad()
def processed_region(
    params: torch.nn.ParameterDict | dict[str, torch.Tensor], scene: Scene, max_radius: float = 1.5,
    max_scale_quantile: float = 0.95,
) -> torch.Tensor:
    """Gaussians of the reconstructed area (not the far background or horizon).

    Args:
        params: Gaussian parameters (scene coordinates).
        scene: Scene (used for its center and scale).
        max_radius: Keep Gaussians closer than ``max_radius * scene_scale`` to
            the center of the SfM points (drops the background sphere and horizon).
        max_scale_quantile: Drop the largest Gaussians (blurry background
            blobs) above this quantile of their largest axis.

    Returns:
        Boolean mask ``[N]`` on the parameters' device.
    """
    means = params["means"].detach()
    scale = torch.exp(params["scales"].detach()).max(dim=1).values
    center = torch.as_tensor(
        np.median(scene.points, axis=0) if len(scene.points) else np.zeros(3), dtype=means.dtype, device=means.device
    )
    inside = (means - center).norm(dim=1) < max_radius * scene.scene_scale
    if inside.any():
        inside &= scale <= torch.quantile(scale[inside].float()[:1_000_000], max_scale_quantile)
    return inside


def dense_points_from_splat(
    params: torch.nn.ParameterDict | dict[str, torch.Tensor],
    scene: Scene,
    min_opacity: float | str = "auto",
    max_scale_quantile: float = 0.95,
    max_radius: float = 1.5,
    return_index: bool = False,
) -> tuple[np.ndarray, ...]:
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
        return_index: Also return the index of each point's Gaussian.

    Returns:
        Tuple ``(xyz [N, 3] float64 in scene coordinates, rgb [N, 3] uint8)``,
        plus ``index [N]`` with ``return_index``.
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
    if return_index:
        return means[keep].double().cpu().numpy(), rgb.cpu().numpy(), torch.nonzero(keep).reshape(-1).cpu().numpy()
    return means[keep].double().cpu().numpy(), rgb.cpu().numpy()


@torch.no_grad()
def supported_by_sfm(xyz: np.ndarray, scene: Scene, radius: float) -> np.ndarray:
    """Whether an SfM point lies within about ``radius`` of each point.

    SfM points are put in voxels of size ``radius`` (dilated by one voxel);
    a point is supported when its voxel is occupied. Floaters of the splat in
    empty air have no SfM point near them.

    Args:
        xyz: Points ``[N, 3]`` (scene coordinates).
        scene: Scene with the SfM points.
        radius: Voxel size (scene units).

    Returns:
        Boolean ``[N]``.
    """
    if not len(scene.points) or not len(xyz):
        return np.ones(len(xyz), bool)
    sp = np.floor(scene.points / radius).astype(np.int64)
    lo = sp.min(0) - 2
    span = sp.max(0) - lo + 3
    code = lambda v: ((v[:, 0] - lo[0]) * span[1] + (v[:, 1] - lo[1])) * span[2] + (v[:, 2] - lo[2])  # noqa: E731
    occ = set()
    base = np.unique(sp, axis=0)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                occ.update(code(base + np.array([dx, dy, dz])).tolist())
    q = np.floor(xyz / radius).astype(np.int64)
    inside = np.all((q >= lo) & (q < lo + span), axis=1)
    out = np.zeros(len(xyz), bool)
    keys = np.array(sorted(occ), dtype=np.int64)
    qc = code(q[inside])
    pos = np.clip(np.searchsorted(keys, qc), 0, len(keys) - 1)
    out[np.flatnonzero(inside)] = keys[pos] == qc
    return out


@torch.no_grad()
def colors_from_photos(
    params: torch.nn.ParameterDict | dict[str, torch.Tensor],
    xyz: np.ndarray,
    cameras: list,
    sh_degree: int,
    fallback: np.ndarray,
    depth_scale: float = 0.5,
    occlusion_tolerance: float = 0.02,
    blend_power: float = 24.0,
) -> np.ndarray:
    """Color points from the photo that sees each one sharpest (as the orthophoto does).

    Each camera renders a splat depth map (occlusion test); a visible point is
    weighted by ``(pixels per unit at its distance x (1 - r)) ** blend_power``
    with ``r`` the distance to the image center and the measured sharpness of
    the photo as an extra factor, so the closest, most central, sharpest view
    wins and other views only blend near the switch. Points no photo sees
    keep ``fallback``.

    Args:
        params: Gaussian parameters (scene coordinates, CUDA).
        xyz: Points ``[N, 3]`` (scene coordinates).
        cameras: Cameras with their (undistorted) images.
        sh_degree: SH degree of the model.
        fallback: Colors ``[N, 3]`` (uint8) for points no photo sees.
        depth_scale: Resolution factor of the depth maps.
        occlusion_tolerance: Relative depth margin of the visibility test.
        blend_power: Sharpness of the best-view selection.

    Returns:
        Colors ``[N, 3]`` (uint8).
    """
    from .ortho import _load_image, _lse_update, sharpness_factors
    from .render import render

    device = params["means"].device
    pts = torch.as_tensor(xyz, dtype=torch.float32, device=device)
    n = len(pts)
    acc = torch.zeros(n, 3, device=device)
    wsum = torch.zeros(n, device=device)
    ref = torch.full((n,), -float("inf"), device=device)
    quality = sharpness_factors(cameras)
    for ci, cam in enumerate(cameras):
        img = _load_image(cam.image_path, (cam.width, cam.height), device)
        dcam = cam.scaled(depth_scale)
        vm = torch.tensor(dcam.world_to_cam, dtype=torch.float32, device=device)
        K = torch.tensor(dcam.K, dtype=torch.float32, device=device)
        depth, alpha, _ = render(params, vm[None], K[None], dcam.width, dcam.height, sh_degree, "gsplat",
                                 render_mode="ED")
        depth = torch.where(alpha[0, ..., 0] > 0.5, depth[0, ..., 0], torch.full_like(depth[0, ..., 0], 1e10))
        Kf = torch.tensor(cam.K, dtype=torch.float32, device=device)
        pc = pts @ vm[:3, :3].T + vm[:3, 3]
        z = pc[:, 2]
        uv = (pc @ Kf.T)[:, :2] / z.clamp_min(1e-6)[:, None]
        inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
        g = torch.stack([uv[:, 0] / cam.width * 2 - 1, uv[:, 1] / cam.height * 2 - 1], -1)[None, None]
        d = F.grid_sample(depth[None, None], g, mode="nearest", align_corners=False)[0, 0, 0]
        visible = inside & (z <= d * (1 + occlusion_tolerance))
        r = torch.linalg.norm(uv - Kf[:2, 2], dim=1) / (0.5 * float(np.hypot(cam.width, cam.height)))
        logw = blend_power * ((float(Kf[0, 0]) / z.clamp_min(1e-6)).log() + (1 - r).clamp_min(1e-6).log()
                              + float(np.log(quality[ci])))
        logw = torch.where(visible, logw, torch.full_like(logw, -float("inf")))
        w = _lse_update(ref, acc, wsum, 0, logw)
        col = F.grid_sample(img[None], g, mode="bilinear", align_corners=False)[0, :, 0].T
        acc += col * w[:, None]
        wsum += w
    seen = wsum > 0
    rgb = torch.where(seen[:, None], acc / wsum.clamp_min(1e-30)[:, None], torch.zeros_like(acc))
    out = fallback.copy()
    s_np = seen.cpu().numpy()
    out[s_np] = (rgb[seen] * 255).round().clamp(0, 255).byte().cpu().numpy()
    return out


@torch.no_grad()
def dense_point_cloud(
    params: torch.nn.ParameterDict | dict[str, torch.Tensor],
    scene: Scene,
    min_opacity: float | str = "auto",
    geo: GeoReference | None = None,
    sh_degree: int = 0,
    photo_colors: bool = True,
    support_radius: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The dense point cloud: solid Gaussian centers, checked against SfM, colored from the photos.

    * Points: centers of solid Gaussians (:func:`dense_points_from_splat`).
    * Support: only points with an SfM point within about ``support_radius``
      meters (scene units without ``geo``: that fraction x 1 % of the scene
      radius) are kept; splat floaters in empty air go.
    * Colors: from the photo that sees each point sharpest
      (:func:`colors_from_photos`, needs the CUDA backend); otherwise the
      splat color.

    Args:
        params: Gaussian parameters.
        scene: Scene (SfM points, cameras with images).
        min_opacity: Opacity threshold (``"auto"``: Otsu).
        geo: Georeference (meters per scene unit for ``support_radius``).
        sh_degree: SH degree of the model (depth rendering).
        photo_colors: Color the points from the photos.
        support_radius: Distance to the nearest SfM point (meters); ``0`` keeps all.

    Returns:
        Tuple ``(xyz [N, 3] scene coordinates, rgb [N, 3] uint8, index [N] of the Gaussians)``.
    """
    xyz, rgb, idx = dense_points_from_splat(params, scene, min_opacity, return_index=True)
    if support_radius > 0 and len(xyz):
        radius = support_radius / geo.scale if geo is not None else support_radius * 0.01 * scene.scene_scale
        keep = supported_by_sfm(xyz, scene, radius)
        logger.info("Dense point cloud: %d of %d points supported by SfM points", int(keep.sum()), len(keep))
        xyz, rgb, idx = xyz[keep], rgb[keep], idx[keep]
    if photo_colors and len(xyz) and params["means"].is_cuda:
        try:
            rgb = colors_from_photos(params, xyz, scene.cameras, sh_degree, rgb)
        except (ImportError, RuntimeError) as e:            # no gsplat CUDA backend: splat colors
            logger.warning("Point colors from the photos skipped (%s)", e)
    return xyz, rgb, idx


def export_point_clouds(
    params: torch.nn.ParameterDict | dict[str, torch.Tensor],
    scene: Scene,
    output_dir: str | Path,
    geo: GeoReference | None = None,
    min_opacity: float | str = "auto",
    dense: tuple[np.ndarray, ...] | None = None,
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
        dense: Precomputed dense cloud ``(xyz, rgb, ...)`` (see
            :func:`dense_point_cloud`); ``None`` takes the solid Gaussian centers
            with their splat colors.

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

    if dense is None:
        dense = dense_points_from_splat(params, scene, min_opacity=min_opacity)
    dense_xyz, dense_rgb = dense[0], dense[1]
    return {
        "points_sparse": save_point_cloud(
            out / "points_sparse.ply", to_out(scene.points), scene.colors, double=geo is not None
        ),
        "points_dense": save_point_cloud(out / "points_dense.ply", to_out(dense_xyz), dense_rgb, double=geo is not None),
    }
