"""True orthophotos and DSMs from a georeferenced Gaussian splat (GPU).

Two orthophotos are produced:

* ``orthophoto_splat.tif``: the splat rendered with an orthographic camera
  looking straight down.
* ``orthophoto.tif``: the *original photos* orthorectified onto the splat's
  surface model. Every ground cell is projected into all photos, occluded views
  are rejected using depth maps rendered from the splat, and the most nadir
  views are blended. This keeps the full sharpness of the photos.

Both, plus ``dsm.tif`` (surface elevation), are written as GeoTIFFs in the
georeference's CRS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .geo import GeoReference, read_gcps
from .render import render
from .scene import Camera, Scene

logger = logging.getLogger("geosplat")


@dataclass
class OrthoGrid:
    """A north-up raster grid in local geo coordinates (CRS minus origin).

    Attributes:
        left: Easting of the left edge (local meters).
        top: Northing of the top edge (local meters).
        gsd: Pixel size in meters.
        width: Raster width in pixels.
        height: Raster height in pixels.
        z_top: Height (local meters) of the virtual orthographic camera, above
            all content of interest.
        z_bottom: Gaussians below this height (local meters) are ignored.
    """

    left: float
    top: float
    gsd: float
    width: int
    height: int
    z_top: float
    z_bottom: float

    def pixel_centers(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Local easting / northing of every pixel center.

        Args:
            device: Torch device.

        Returns:
            Tuple ``(east [H, W], north [H, W])``.
        """
        e = self.left + (torch.arange(self.width, device=device, dtype=torch.float64) + 0.5) * self.gsd
        n = self.top - (torch.arange(self.height, device=device, dtype=torch.float64) + 0.5) * self.gsd
        return e[None].expand(self.height, -1), n[:, None].expand(-1, self.width)


def make_grid(
    scene: Scene, geo: GeoReference, gsd: float | None = None, margin: float = 0.02, max_pixels: int = 30_000_000
) -> OrthoGrid:
    """Choose the orthophoto extent and resolution from the reconstruction.

    The extent covers the 1st to 99th percentile of the sparse points. The
    default GSD is the median ``flight height / focal length`` over cameras,
    i.e. the native resolution of the (undistorted) photos.

    Args:
        scene: Scene (cameras at the resolution that will be orthorectified).
        geo: Georeference of the scene.
        gsd: Pixel size in meters (``None`` to estimate it).
        margin: Extra border as a fraction of the extent.
        max_pixels: If the raster would exceed this many pixels the GSD is
            increased (bounds GPU/CPU memory).

    Returns:
        The :class:`OrthoGrid`.
    """
    pts = geo.scene_to_local(scene.points.astype(np.float64))
    cams = geo.scene_to_local(np.stack([c.center for c in scene.cameras]))
    lo, hi = np.percentile(pts[:, :2], 1, axis=0), np.percentile(pts[:, :2], 99, axis=0)
    pad = (hi - lo) * margin
    lo, hi = lo - pad, hi + pad
    ground = float(np.median(pts[:, 2]))
    if gsd is None:
        f = np.array([c.K[0, 0] for c in scene.cameras])
        gsd = float(np.median((cams[:, 2] - ground) / f * 1.0))
        gsd = float(f"{max(gsd, 1e-3):.2g}")
    flight = float(np.median(cams[:, 2] - ground))
    area_px = (hi[0] - lo[0]) * (hi[1] - lo[1]) / gsd**2
    if area_px > max_pixels:
        gsd = float(f"{gsd * np.sqrt(area_px / max_pixels) * 1.01:.2g}")
        logger.info("Orthophoto GSD raised to %.3f m to stay under %d pixels", gsd, max_pixels)
    width = int(np.ceil((hi[0] - lo[0]) / gsd))
    height = int(np.ceil((hi[1] - lo[1]) / gsd))
    return OrthoGrid(
        left=float(lo[0]), top=float(hi[1]), gsd=gsd, width=width, height=height,
        z_top=float(cams[:, 2].max() + 1.0),
        z_bottom=float(np.percentile(pts[:, 2], 0.5) - 0.3 * abs(flight)),
    )


def _crop_params(
    params: torch.nn.ParameterDict | dict[str, torch.Tensor], geo: GeoReference, grid: OrthoGrid
) -> dict[str, torch.Tensor]:
    """Keep Gaussians inside the ortho volume (drops far background and sky).

    Args:
        params: Gaussian parameters in scene coordinates.
        geo: Georeference.
        grid: Ortho grid.

    Returns:
        Dictionary of cropped (detached) parameter tensors.
    """
    T = torch.as_tensor(geo.scene_to_geo, dtype=torch.float32, device=params["means"].device)
    local = params["means"].detach() @ T[:3, :3].T + T[:3, 3]
    pad = 0.05 * max(grid.width, grid.height) * grid.gsd
    keep = (
        (local[:, 2] < grid.z_top - 0.05) & (local[:, 2] > grid.z_bottom)
        & (local[:, 0] > grid.left - pad) & (local[:, 0] < grid.left + grid.width * grid.gsd + pad)
        & (local[:, 1] < grid.top + pad) & (local[:, 1] > grid.top - grid.height * grid.gsd - pad)
    )
    return {k: v.detach()[keep] for k, v in params.items()}


@torch.no_grad()
def render_splat_ortho(
    params: torch.nn.ParameterDict | dict[str, torch.Tensor],
    geo: GeoReference,
    grid: OrthoGrid,
    sh_degree: int,
    tile: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render the splat with a top-down orthographic camera.

    Args:
        params: Gaussian parameters (scene coordinates).
        geo: Georeference.
        grid: Ortho grid.
        sh_degree: SH degree of the model.
        tile: Maximum tile size in pixels rendered at once.

    Returns:
        Tuple ``(rgb [H, W, 3], elevation [H, W], alpha [H, W])`` on the model's
        device. Elevation is in local geo meters (add ``geo.origin[2]``).
    """
    device = params["means"].device
    p = _crop_params(params, geo, grid)
    s = geo.scale
    Rc = np.diag([1.0, -1.0, -1.0])  # camera axes: x = east, y = south, z = down
    C = np.array([grid.left + grid.width * grid.gsd / 2, grid.top - grid.height * grid.gsd / 2, grid.z_top])
    w2c = np.eye(4)
    w2c[:3, :3] = Rc @ geo.rotation
    w2c[:3, 3] = Rc @ (geo.scene_to_geo[:3, 3] - C) / s
    viewmat = torch.tensor(w2c, dtype=torch.float32, device=device)[None]
    f = s / grid.gsd

    rgb = torch.zeros(grid.height, grid.width, 3, device=device)
    elev = torch.zeros(grid.height, grid.width, device=device)
    alpha = torch.zeros(grid.height, grid.width, device=device)
    for y0 in range(0, grid.height, tile):
        for x0 in range(0, grid.width, tile):
            h, w = min(tile, grid.height - y0), min(tile, grid.width - x0)
            K = torch.tensor(
                [[[f, 0, grid.width / 2 - x0], [0, f, grid.height / 2 - y0], [0, 0, 1]]],
                dtype=torch.float32, device=device,
            )
            img, a, _ = render(p, viewmat, K, w, h, sh_degree, "gsplat", camera_model="ortho", render_mode="RGB+ED")
            rgb[y0 : y0 + h, x0 : x0 + w] = img[0, ..., :3]
            elev[y0 : y0 + h, x0 : x0 + w] = grid.z_top - img[0, ..., 3] * s
            alpha[y0 : y0 + h, x0 : x0 + w] = a[0, ..., 0]
    return rgb, elev, alpha


def _load_image(path: Path, size: tuple[int, int], device: torch.device) -> torch.Tensor:
    """Load an image as a float ``[3, H, W]`` tensor on ``device``.

    Args:
        path: Image path.
        size: Expected ``(width, height)``; the image is resized if different.
        device: Torch device.

    Returns:
        Image tensor in ``[0, 1]``.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != size:
            im = im.resize(size, Image.LANCZOS)
        arr = torch.from_numpy(np.asarray(im).copy())
    return arr.to(device).permute(2, 0, 1).float() / 255.0


@torch.no_grad()
def orthorectify_photos(
    params: torch.nn.ParameterDict | dict[str, torch.Tensor] | None,
    cameras: list[Camera],
    geo: GeoReference,
    grid: OrthoGrid,
    elevation: torch.Tensor,
    valid: torch.Tensor,
    sh_degree: int,
    blend_power: float = 8.0,
    occlusion_tolerance: float = 0.03,
    depth_scale: float = 0.5,
    chunk: int = 2_000_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an orthophoto from the original photos using the splat surface.

    Args:
        params: Gaussian parameters (scene coordinates) used to render depth
            maps for the occlusion test, or ``None`` to skip occlusion handling
            (fast_orto straight from SfM).
        cameras: Cameras with their (undistorted) images.
        geo: Georeference.
        grid: Ortho grid.
        elevation: Surface height per ortho pixel ``[H, W]`` (local meters).
        valid: Boolean mask ``[H, W]`` of pixels with a surface.
        sh_degree: SH degree of the model (unused for depth, kept for API symmetry).
        blend_power: Exponent on the per-view "nadir-ness" weight. Higher values
            approach picking the single best photo (sharper, harder seams).
        occlusion_tolerance: Relative depth tolerance for the visibility test.
        depth_scale: Resolution factor of the depth maps used for occlusion.
        chunk: Ortho pixels processed at once (bounds GPU memory).

    Returns:
        Tuple ``(rgb [H, W, 3], coverage [H, W] bool)``.
    """
    device = elevation.device
    east, north = grid.pixel_centers(device)
    local = torch.stack([east, north, elevation.double()], dim=-1).reshape(-1, 3)
    T = torch.as_tensor(geo.scene_to_geo, dtype=torch.float64, device=device)
    pts = ((local - T[:3, 3]) @ torch.linalg.inv(T[:3, :3]).T).float()  # scene coordinates
    del local, east, north
    n = pts.shape[0]
    acc = torch.zeros(n, 3, device=device)
    wsum = torch.zeros(n, device=device)
    # Fallback ignoring occlusion, used where every view fails the depth test.
    acc_any = torch.zeros(n, 3, device=device)
    wsum_any = torch.zeros(n, device=device)
    mask = valid.reshape(-1)

    for cam in cameras:
        img = _load_image(cam.image_path, (cam.width, cam.height), device)
        dcam = cam.scaled(depth_scale)
        vm = torch.tensor(dcam.world_to_cam, dtype=torch.float32, device=device)
        K = torch.tensor(dcam.K, dtype=torch.float32, device=device)
        if params is not None:
            depth, alpha, _ = render(params, vm[None], K[None], dcam.width, dcam.height, sh_degree, "gsplat", render_mode="ED")
            depth = torch.where(alpha[0, ..., 0] > 0.5, depth[0, ..., 0], torch.full_like(depth[0, ..., 0], 1e10))
        else:
            depth = torch.full((dcam.height, dcam.width), 1e10, device=device)
        Kf = torch.tensor(cam.K, dtype=torch.float32, device=device)
        half_diag = 0.5 * float(np.hypot(cam.width, cam.height))
        for s0 in range(0, n, chunk):
            p = pts[s0 : s0 + chunk]
            pc = p @ vm[:3, :3].T + vm[:3, 3]
            z = pc[:, 2]
            uv = (pc @ Kf.T)[:, :2] / z.clamp_min(1e-6)[:, None]
            inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
            gx = uv[:, 0] / cam.width * 2 - 1
            gy = uv[:, 1] / cam.height * 2 - 1
            grid_xy = torch.stack([gx, gy], -1)[None, None]
            d = F.grid_sample(depth[None, None], grid_xy, mode="nearest", align_corners=False)[0, 0, 0]
            seen = inside & mask[s0 : s0 + chunk]
            visible = seen & (z <= d * (1 + occlusion_tolerance))
            r = torch.linalg.norm(uv - Kf[:2, 2], dim=1) / half_diag
            w_any = (1 - r).clamp_min(0) ** blend_power * seen
            w = w_any * visible
            col = F.grid_sample(img[None], grid_xy, mode="bilinear", align_corners=False)[0, :, 0].T
            acc[s0 : s0 + chunk] += col * w[:, None]
            wsum[s0 : s0 + chunk] += w
            acc_any[s0 : s0 + chunk] += col * w_any[:, None]
            wsum_any[s0 : s0 + chunk] += w_any
    occluded_everywhere = (wsum <= 0) & (wsum_any > 0)
    acc = torch.where(occluded_everywhere[:, None], acc_any, acc)
    wsum = torch.where(occluded_everywhere, wsum_any, wsum)
    coverage = wsum > 0
    rgb = torch.where(coverage[:, None], acc / wsum.clamp_min(1e-12)[:, None], torch.zeros_like(acc))
    return rgb.reshape(grid.height, grid.width, 3), coverage.reshape(grid.height, grid.width)


@torch.no_grad()
def fill_holes(z: torch.Tensor, valid: torch.Tensor, downscale: int = 4) -> torch.Tensor:
    """Fill invalid cells of a height grid by iterative neighbourhood averaging (GPU).

    Filling runs on a ``downscale``-times coarser grid for speed; valid cells
    keep their original values.

    Args:
        z: Heights ``[H, W]``.
        valid: Boolean mask ``[H, W]`` of measured cells.
        downscale: Coarsening factor for the fill.

    Returns:
        Filled heights ``[H, W]``.
    """
    if bool(valid.all()):
        return z
    if not bool(valid.any()):
        return torch.zeros_like(z)
    H, W = z.shape
    v = valid.float()[None, None]
    zc = F.avg_pool2d((z * valid)[None, None], downscale, ceil_mode=True)
    vc = F.avg_pool2d(v, downscale, ceil_mode=True)
    zc = zc / vc.clamp_min(1e-6)
    vc = (vc > 0).float()
    for _ in range(max(zc.shape[-2:])):
        if vc.min() > 0:
            break
        zs = F.avg_pool2d(zc * vc, 3, 1, 1, count_include_pad=False)
        vs = F.avg_pool2d(vc, 3, 1, 1, count_include_pad=False)
        fill = (vc == 0) & (vs > 0)
        zc = torch.where(fill, zs / vs.clamp_min(1e-6), zc)
        vc = torch.where(fill, torch.ones_like(vc), vc)
    up = F.interpolate(zc, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    return torch.where(valid, z, up)


@torch.no_grad()
def dsm_from_points(
    points_local: np.ndarray, grid: OrthoGrid, device: torch.device, cell: float | None = None, smooth: int = 2
) -> torch.Tensor:
    """Interpolate a smooth surface model from sparse points (GPU).

    Points are averaged into a coarse grid, holes are filled by iterative
    neighbourhood averaging, the result is smoothed and upsampled to ``grid``.

    Args:
        points_local: Sparse points ``[N, 3]`` in local geo coordinates.
        grid: Target ortho grid.
        device: Torch device.
        cell: Coarse cell size in meters (``None``: about two points per cell).
        smooth: Number of 3x3 smoothing passes on the coarse grid.

    Returns:
        Elevation ``[H, W]`` (local meters) on ``device``.
    """
    pts = torch.as_tensor(points_local, dtype=torch.float32, device=device)
    lo, hi = torch.quantile(pts[:, 2], 0.01), torch.quantile(pts[:, 2], 0.99)
    pts = pts[(pts[:, 2] >= lo) & (pts[:, 2] <= hi)]
    w_m, h_m = grid.width * grid.gsd, grid.height * grid.gsd
    if cell is None:
        cell = max(float(np.sqrt(w_m * h_m / max(len(pts), 1) * 2.0)), grid.gsd)
    cw, ch = max(1, int(np.ceil(w_m / cell))), max(1, int(np.ceil(h_m / cell)))
    col = ((pts[:, 0] - grid.left) / cell).long()
    row = ((grid.top - pts[:, 1]) / cell).long()
    ok = (col >= 0) & (col < cw) & (row >= 0) & (row < ch)
    flat = row[ok] * cw + col[ok]
    zsum = torch.zeros(ch * cw, device=device).index_add_(0, flat, pts[ok, 2])
    cnt = torch.zeros(ch * cw, device=device).index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    z = (zsum / cnt.clamp_min(1)).view(1, 1, ch, cw)
    valid = (cnt > 0).float().view(1, 1, ch, cw)
    for _ in range(max(ch, cw)):
        if valid.min() > 0:
            break
        zs = F.avg_pool2d(z * valid, 3, 1, 1, count_include_pad=False)
        vs = F.avg_pool2d(valid, 3, 1, 1, count_include_pad=False)
        fill = (valid == 0) & (vs > 0)
        z = torch.where(fill, zs / vs.clamp_min(1e-6), z)
        valid = torch.where(fill, torch.ones_like(valid), valid)
    for _ in range(smooth):
        z = F.avg_pool2d(F.pad(z, (1, 1, 1, 1), mode="replicate"), 3, 1, 0)
    up = F.interpolate(z, size=(grid.height, grid.width), mode="bilinear", align_corners=False)
    return up[0, 0]


def make_fast_orthophoto(
    scene: Scene,
    geo: GeoReference,
    output_dir: str | Path,
    gsd: float | None = None,
    gcps: str | Path | None = None,
    device: torch.device | str | None = None,
) -> dict[str, object]:
    """Fast orthophoto straight from SfM (no splat training needed).

    The DSM is interpolated from the sparse SfM points and the photos are
    orthorectified onto it, choosing the most nadir view per ground cell.
    Without dense depth there is no occlusion handling, so tall objects
    (buildings, trees) lean; flat terrain is accurate.

    Args:
        scene: Scene with cameras, images and sparse points.
        geo: Georeference of the scene.
        output_dir: Folder for ``fast_orto.tif``, ``dsm_sparse.tif`` and a preview.
        gsd: Pixel size in meters (``None``: native photo resolution).
        gcps: Optional ``name,x,y,z`` CSV to draw on the preview.
        device: Torch device (default: CUDA if available).

    Returns:
        Dictionary with output paths and the GSD.
    """
    from .utils import get_device

    device = get_device(device)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    grid = make_grid(scene, geo, gsd)
    logger.info("fast_orto grid %dx%d at %.3f m/px", grid.width, grid.height, grid.gsd)
    elev = dsm_from_points(geo.scene_to_local(scene.points.astype(np.float64)), grid, device)
    valid = torch.ones_like(elev, dtype=torch.bool)
    rgb, cov = orthorectify_photos(None, scene.cameras, geo, grid, elev, valid, sh_degree=0)
    rgba = torch.cat([(rgb * 255).round().byte(), (cov.float() * 255).byte()[..., None]], -1).cpu().numpy()
    results: dict[str, object] = {"gsd": grid.gsd, "width": grid.width, "height": grid.height}
    results["fast_orto"] = write_geotiff(out / "fast_orto.tif", rgba, geo, grid)
    dsm = (elev + float(geo.origin[2])).cpu().numpy().astype(np.float32)
    results["dsm_sparse"] = write_geotiff(out / "dsm_sparse.tif", dsm, geo, grid, nodata=-9999.0)
    results["fast_orto_preview"] = _preview(rgba[..., :3], grid, geo, read_gcps(gcps) if gcps else {},
                                            out / "fast_orto_preview.jpg")
    return results


def write_geotiff(
    path: str | Path, data: np.ndarray, geo: GeoReference, grid: OrthoGrid, nodata: float | None = None
) -> Path:
    """Write a north-up raster as a compressed GeoTIFF.

    Args:
        path: Output ``.tif`` path.
        data: Array ``[H, W]`` or ``[H, W, B]``.
        geo: Georeference (provides CRS and origin).
        grid: Grid describing the raster's extent and pixel size.
        nodata: Optional nodata value.

    Returns:
        The written path.
    """
    import rasterio
    from rasterio.transform import from_origin

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bands = data[..., None] if data.ndim == 2 else data
    transform = from_origin(grid.left + geo.origin[0], grid.top + geo.origin[1], grid.gsd, grid.gsd)
    profile = dict(
        driver="GTiff", width=grid.width, height=grid.height, count=bands.shape[2], dtype=bands.dtype,
        crs=geo.crs, transform=transform, compress="deflate", tiled=True, nodata=nodata,
    )
    if bands.dtype == np.uint8 and bands.shape[2] in (3, 4):
        profile["photometric"] = "RGB"
        if bands.shape[2] == 4:
            profile["alpha"] = "YES"  # 4th band is transparency (GDAL creation option)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.moveaxis(bands, 2, 0))
    return path


def _preview(
    rgb: np.ndarray, grid: OrthoGrid, geo: GeoReference, gcps: dict[str, np.ndarray], path: Path,
    max_size: int = 2000,
) -> Path:
    """Save a downscaled JPEG preview with GCP markers.

    Args:
        rgb: ``uint8`` image ``[H, W, 3]``.
        grid: Ortho grid.
        geo: Georeference.
        gcps: Ground control points in the CRS (may be empty).
        path: Output path.
        max_size: Longest side of the preview.

    Returns:
        The written path.
    """
    im = Image.fromarray(rgb)
    scale = min(1.0, max_size / max(im.size))
    im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
    draw = ImageDraw.Draw(im)
    for name, xyz in gcps.items():
        x = (xyz[0] - geo.origin[0] - grid.left) / grid.gsd * scale
        y = (grid.top - (xyz[1] - geo.origin[1])) / grid.gsd * scale
        draw.ellipse([x - 8, y - 8, x + 8, y + 8], outline=(255, 0, 0), width=3)
        draw.text((x + 10, y - 10), name, fill=(255, 0, 0))
    im.save(path, quality=90)
    return path


def make_orthophoto(
    params: torch.nn.ParameterDict | dict[str, torch.Tensor],
    scene: Scene,
    geo: GeoReference,
    output_dir: str | Path,
    sh_degree: int,
    gsd: float | None = None,
    gcps: str | Path | None = None,
    from_photos: bool = True,
) -> dict[str, object]:
    """Produce GeoTIFF orthophotos and DSM for a georeferenced splat.

    Args:
        params: Trained Gaussian parameters (scene coordinates, on the GPU).
        scene: Scene whose cameras/images are used for orthorectification.
        geo: Georeference of the scene.
        output_dir: Folder for the outputs.
        sh_degree: SH degree of the model.
        gsd: Pixel size in meters (``None``: native photo resolution).
        gcps: Optional ``name,x,y,z`` CSV (same CRS) to mark on the previews and
            to compare against the DSM.
        from_photos: Also orthorectify the original photos.

    Returns:
        Dictionary with output paths, the GSD and, if GCPs were given, the
        per-point vertical difference ``dsm - gcp_z``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    grid = make_grid(scene, geo, gsd)
    logger.info("Orthophoto grid %dx%d at %.3f m/px", grid.width, grid.height, grid.gsd)
    rgb, elev, alpha = render_splat_ortho(params, geo, grid, sh_degree)
    measured = alpha > 0.5
    # Semi-transparent spots (thin or unseen-from-above splats): interpolate the surface.
    elev = fill_holes(elev, measured)
    valid = alpha > 0.05
    results: dict[str, object] = {"gsd": grid.gsd, "width": grid.width, "height": grid.height}

    a8 = (valid.float() * 255).byte()
    rgba = torch.cat([(rgb * 255).round().byte(), a8[..., None]], -1).cpu().numpy()
    results["orthophoto_splat"] = write_geotiff(out / "orthophoto_splat.tif", rgba, geo, grid)
    dsm = torch.where(valid, elev + float(geo.origin[2]), torch.full_like(elev, -9999.0)).cpu().numpy()
    results["dsm"] = write_geotiff(out / "dsm.tif", dsm.astype(np.float32), geo, grid, nodata=-9999.0)

    gcp_pts = read_gcps(gcps) if gcps else {}
    preview_src = rgba[..., :3]
    if from_photos:
        everywhere = torch.ones_like(valid)
        photo_rgb, cov = orthorectify_photos(params, scene.cameras, geo, grid, elev, everywhere, sh_degree)
        cov8 = (cov.float() * 255).byte()
        prgba = torch.cat([(photo_rgb * 255).round().byte(), cov8[..., None]], -1).cpu().numpy()
        results["orthophoto"] = write_geotiff(out / "orthophoto.tif", prgba, geo, grid)
        preview_src = prgba[..., :3]
    results["preview"] = _preview(preview_src, grid, geo, gcp_pts, out / "orthophoto_preview.jpg")

    if gcp_pts:
        diffs = {}
        for name, xyz in gcp_pts.items():
            col = int((xyz[0] - geo.origin[0] - grid.left) / grid.gsd)
            row = int((grid.top - (xyz[1] - geo.origin[1])) / grid.gsd)
            if 0 <= row < grid.height and 0 <= col < grid.width and dsm[row, col] != -9999.0:
                diffs[name] = float(dsm[row, col] - xyz[2])
        results["gcp_dz"] = diffs
        logger.info("DSM minus GCP height (m): %s", {k: round(v, 2) for k, v in diffs.items()})
    return results
