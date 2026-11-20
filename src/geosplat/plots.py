"""One-line notebook displays for every geosplat product.

Each function detects by itself whether data is georeferenced:

* :func:`photos`: map of photo positions with polygon selection (GPS), else thumbnails.
* :func:`ortho`: orthophoto over Google Satellite (GeoTIFF with CRS), else the image.
* :func:`point_cloud`: top and 3D views of a ``.ply`` point cloud.
* :func:`splat`: interactive 3D Gaussian splat viewer.
* :func:`renders`: held-out photos next to their renders.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .positions import photo_positions
from .sfm import list_images


def _show_image(path: str | Path, title: str = "", size: float = 10.0) -> None:
    """Display an image file with matplotlib.

    Args:
        path: Image path.
        title: Figure title.
        size: Figure size in inches.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    plt.figure(figsize=(size, size))
    plt.imshow(Image.open(path))
    plt.axis("off")
    plt.title(title)
    plt.show()


def photos(
    images_dir: str | Path, use_map: bool = True, thumbnails: int = 8, selection_file: str | Path | None = None
) -> object | None:
    """Show the input photos.

    With GPS (3+ photos), shows an interactive Google Hybrid/Satellite map of
    photo positions where photos can be included / excluded (shapes, clicks,
    buttons). Otherwise shows a grid of thumbnails.

    Args:
        images_dir: Photo folder.
        use_map: Use the interactive map when photos have GPS.
        thumbnails: Number of thumbnails without GPS.
        selection_file: Optional text file to load / save the selection.

    Returns:
        The :class:`geosplat.mapview.PhotoSelector` (use ``.selected``,
        ``.all_selected``, ``.save()``), or ``None`` without GPS.
    """
    from IPython.display import display

    paths = list_images(images_dir)
    if use_map and len(photo_positions(images_dir)) >= 3:
        from .mapview import PhotoSelector

        selector = PhotoSelector(images_dir, selection_file=selection_file)
        display(selector)
        return selector
    import matplotlib.pyplot as plt
    from PIL import Image

    picks = paths[:: max(1, len(paths) // thumbnails)][:thumbnails]
    cols = min(4, len(picks))
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, p in zip(axes.flat, picks):
        ax.imshow(Image.open(p))
        ax.set_title(p.name, fontsize=8)
    plt.tight_layout()
    plt.show()
    return None


def ortho(tif_path: str | Path, gcps: dict[str, np.ndarray] | None = None, preview: str | Path | None = None) -> None:
    """Show an orthophoto: over Google Satellite if it has a CRS, else as an image.

    Args:
        tif_path: Orthophoto GeoTIFF.
        gcps: Optional ground control points ``{name: [x, y, z]}`` (CRS) drawn as red circles.
        preview: Optional JPEG preview used when the raster has no CRS (default:
            ``*_preview.jpg`` in the same folder).
    """
    import rasterio
    from IPython.display import display

    tif_path = Path(tif_path)
    with rasterio.open(tif_path) as src:
        has_crs = src.crs is not None
    if has_crs:
        try:
            from .mapview import ortho_map

            display(ortho_map(tif_path, gcps=gcps))
            return
        except ImportError:
            pass  # no ipyleaflet: fall back to a static image
    if preview is None:
        candidates = sorted(tif_path.parent.glob("*preview.jpg"))
        preview = candidates[0] if candidates else tif_path
    _show_image(preview, tif_path.name)


def point_cloud(ply_path: str | Path, max_points: int = 150_000, point_size: float = 0.3) -> None:
    """Top and 3D views of a colored point cloud.

    Georeferenced clouds (float64 CRS coordinates) use Z as up; local clouds
    use the scene convention (up = -Y).

    Args:
        ply_path: ``.ply`` with ``x, y, z, red, green, blue``.
        max_points: Random subsample size for plotting.
        point_size: Marker size.
    """
    import matplotlib.pyplot as plt
    from plyfile import PlyData

    v = PlyData.read(str(ply_path))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    rgb = np.stack([v["red"], v["green"], v["blue"]], 1) / 255.0
    georef = v.data.dtype["x"] == np.float64
    if not georef:  # local scene frame (up = -Y) -> x, z, -y
        xyz = np.stack([xyz[:, 0], xyz[:, 2], -xyz[:, 1]], 1)
    idx = np.random.default_rng(0).choice(len(xyz), min(max_points, len(xyz)), replace=False)
    xyz, rgb = xyz[idx], rgb[idx]

    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(xyz[:, 0], xyz[:, 1], c=rgb, s=point_size)
    ax.set_aspect("equal")
    ax.set_title(f"{Path(ply_path).name}: {len(v.data):,} points (top view)")
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.scatter(*xyz.T, c=rgb, s=point_size)
    lo, hi = np.percentile(xyz, [1, 99], axis=0)
    span = (hi - lo).max() / 2
    mid = (hi + lo) / 2
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)
    ax.set_title("3D view")
    plt.tight_layout()
    plt.show()


def splat(params: object, cameras: list | None = None, max_splats: int = 300_000, height: int = 600) -> object:
    """Interactive 3D viewer of a Gaussian splat (drag: orbit, right-drag: pan, wheel: zoom).

    Args:
        params: Trained parameters or a path to ``splat.ply``.
        cameras: Optional training cameras for the initial view.
        max_splats: Maximum number of Gaussians sent to the browser.
        height: Viewer height in pixels.

    Returns:
        HTML display object (shown when it is the last expression of a cell).
    """
    from .viewer import show

    return show(params, cameras, max_splats, height)  # type: ignore[arg-type]


def renders(output_dir: str | Path, n: int = 3) -> None:
    """Show held-out photos next to their renders (``renders/*.jpg``).

    Args:
        output_dir: Output folder of a run.
        n: Number of images to show.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    files = sorted((Path(output_dir) / "renders").glob("*.jpg"))[:n]
    if not files:
        print("No renders found.")
        return
    fig, axes = plt.subplots(len(files), 1, figsize=(14, 4.5 * len(files)), squeeze=False)
    for ax, p in zip(axes[:, 0], files):
        ax.imshow(Image.open(p))
        ax.set_title(f"{p.stem}: photo (left) | render (right)")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
