"""Gaussian parameter initialization, PLY export and PLY import."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from .sh import num_sh_bases, rgb_to_sh


def knn_mean_distance(points: torch.Tensor, k: int = 3, chunk: int = 4096) -> torch.Tensor:
    """Mean distance from each point to its ``k`` nearest neighbours.

    Args:
        points: Point positions of shape ``[N, 3]``.
        k: Number of neighbours (excluding the point itself).
        chunk: Number of query points processed at once (bounds memory).

    Returns:
        Tensor of shape ``[N]`` with the mean neighbour distance.
    """
    n = points.shape[0]
    k = min(k, max(n - 1, 1))
    out = torch.empty(n, device=points.device, dtype=points.dtype)
    for start in range(0, n, chunk):
        q = points[start : start + chunk]
        d = torch.cdist(q, points)
        # k + 1 because the closest point is the point itself.
        vals = d.topk(min(k + 1, n), largest=False).values[:, 1:]
        out[start : start + chunk] = vals.mean(dim=1) if vals.numel() else 1.0
    return out


def init_gaussians(
    points: np.ndarray,
    colors: np.ndarray,
    sh_degree: int = 3,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    device: torch.device | str = "cpu",
) -> torch.nn.ParameterDict:
    """Create trainable Gaussian parameters from a colored point cloud.

    Parameters are stored in their unconstrained form: log-scales, logit
    opacities and unnormalized quaternions (``wxyz``).

    Args:
        points: Point positions of shape ``[N, 3]``.
        colors: Point colors of shape ``[N, 3]``, either ``uint8`` or floats in ``[0, 1]``.
        sh_degree: Maximum spherical-harmonics degree.
        init_opacity: Initial opacity of every Gaussian.
        init_scale: Multiplier on the nearest-neighbour based initial scale.
        device: Device the parameters are created on.

    Returns:
        A ``ParameterDict`` with keys ``means``, ``scales``, ``quats``,
        ``opacities``, ``sh0`` and ``shN``.
    """
    device = torch.device(device)
    pts = torch.as_tensor(points, dtype=torch.float32, device=device)
    rgb = torch.as_tensor(colors, dtype=torch.float32, device=device)
    if colors.dtype == np.uint8:
        rgb = rgb / 255.0
    n = pts.shape[0]

    dist = knn_mean_distance(pts).clamp_min(1e-7)
    scales = torch.log(dist * init_scale)[:, None].repeat(1, 3)
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0
    opacities = torch.logit(torch.full((n,), init_opacity, device=device))
    sh0 = rgb_to_sh(rgb)[:, None, :]
    shN = torch.zeros(n, num_sh_bases(sh_degree) - 1, 3, device=device)

    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(pts),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(opacities),
            "sh0": torch.nn.Parameter(sh0),
            "shN": torch.nn.Parameter(shN),
        }
    )


def random_point_cloud(
    num_points: int, center: np.ndarray, extent: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a random colored point cloud inside a cube.

    Used when SfM produced no 3D points.

    Args:
        num_points: Number of points to sample.
        center: Cube center of shape ``[3]``.
        extent: Half side length of the cube.
        seed: Random seed.

    Returns:
        Tuple ``(points [N, 3] float32, colors [N, 3] uint8)``.
    """
    rng = np.random.default_rng(seed)
    pts = center[None] + rng.uniform(-extent, extent, size=(num_points, 3))
    cols = rng.integers(0, 256, size=(num_points, 3), dtype=np.uint8)
    return pts.astype(np.float32), cols


def fibonacci_sphere(num_points: int, center: np.ndarray, radius: float) -> np.ndarray:
    """Evenly distribute points on a sphere (Fibonacci lattice).

    Args:
        num_points: Number of points.
        center: Sphere center of shape ``[3]``.
        radius: Sphere radius.

    Returns:
        Points of shape ``[num_points, 3]`` (float32).
    """
    i = np.arange(num_points) + 0.5
    phi = np.arccos(1 - 2 * i / num_points)
    theta = np.pi * (1 + 5**0.5) * i
    dirs = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
    return (center[None] + radius * dirs).astype(np.float32)


@torch.no_grad()
def colors_from_images(
    points: np.ndarray, cameras: list, images: list[torch.Tensor]
) -> tuple[np.ndarray, np.ndarray]:
    """Color 3D points by averaging the pixels they project to (runs on the images' device).

    Args:
        points: Points of shape ``[N, 3]``.
        cameras: :class:`geosplat.scene.Camera` objects matching ``images``.
        images: ``uint8`` image tensors of shape ``[H, W, 3]`` (on GPU for speed).

    Returns:
        Tuple ``(colors [N, 3] uint8, seen [N] bool)`` where ``seen`` marks
        points visible in at least one image.
    """
    device = images[0].device if images else torch.device("cpu")
    pts = torch.as_tensor(points, dtype=torch.float32, device=device)
    acc = torch.zeros(len(pts), 3, device=device)
    cnt = torch.zeros(len(pts), device=device)
    for cam, img in zip(cameras, images):
        w2c = torch.as_tensor(cam.world_to_cam, dtype=torch.float32, device=device)
        K = torch.as_tensor(cam.K, dtype=torch.float32, device=device)
        pc = pts @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        uv = (pc @ K.T)[:, :2] / z.clamp_min(1e-6)[:, None]
        u, v = uv[:, 0].floor().long(), uv[:, 1].floor().long()
        ok = (z > 0) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
        acc[ok] += img[v[ok], u[ok]].float()
        cnt[ok] += 1
    seen = cnt > 0
    cols = (acc / cnt.clamp_min(1)[:, None]).clamp(0, 255).byte()
    return cols.cpu().numpy(), seen.cpu().numpy()


def save_ply(params: torch.nn.ParameterDict | dict[str, torch.Tensor], path: str | Path) -> Path:
    """Write Gaussians to a PLY file in the standard 3DGS layout.

    The file can be opened by common viewers such as SuperSplat,
    antimatter15/splat or the original 3DGS SIBR viewer.

    Args:
        params: Gaussian parameters (see :func:`init_gaussians`).
        path: Output ``.ply`` path.

    Returns:
        The path that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        means = params["means"].detach().cpu().numpy()
        n = means.shape[0]
        normals = np.zeros_like(means)
        f_dc = params["sh0"].detach().reshape(n, 3).cpu().numpy()
        # 3DGS stores higher-order SH channel-major: [N, 3, K-1] flattened.
        f_rest = params["shN"].detach().transpose(1, 2).reshape(n, -1).cpu().numpy()
        opacities = params["opacities"].detach().reshape(n, 1).cpu().numpy()
        scales = params["scales"].detach().cpu().numpy()
        quats = torch.nn.functional.normalize(params["quats"].detach(), dim=-1).cpu().numpy()

    attrs = ["x", "y", "z", "nx", "ny", "nz"]
    attrs += [f"f_dc_{i}" for i in range(f_dc.shape[1])]
    attrs += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
    attrs += ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    data = np.concatenate([means, normals, f_dc, f_rest, opacities, scales, quats], axis=1)
    data = np.nan_to_num(data.astype(np.float32))
    vertices = np.empty(n, dtype=[(a, "f4") for a in attrs])
    for i, name in enumerate(attrs):
        vertices[name] = data[:, i]
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))
    return path


def load_ply(path: str | Path, device: torch.device | str = "cpu") -> torch.nn.ParameterDict:
    """Load Gaussians from a 3DGS-layout PLY file.

    Args:
        path: Path to a ``.ply`` written by :func:`save_ply` or by 3DGS.
        device: Device for the loaded parameters.

    Returns:
        A ``ParameterDict`` with the same keys as :func:`init_gaussians`.
    """
    v = PlyData.read(str(path))["vertex"]
    names = v.data.dtype.names

    def stack(prefix: str) -> np.ndarray:
        """Stack all numbered properties sharing ``prefix`` into columns.

        Args:
            prefix: Property prefix such as ``"f_rest_"``.

        Returns:
            Array of shape ``[N, num_properties]``.
        """
        cols = sorted((n for n in names if n.startswith(prefix)), key=lambda s: int(s.split("_")[-1]))
        if not cols:
            return np.zeros((len(v.data), 0), dtype=np.float32)
        return np.stack([np.asarray(v[c]) for c in cols], axis=1).astype(np.float32)

    n = len(v.data)
    means = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    f_rest = stack("f_rest_")
    k_rest = f_rest.shape[1] // 3
    tensors = {
        "means": means,
        "scales": stack("scale_"),
        "quats": stack("rot_"),
        "opacities": np.asarray(v["opacity"], dtype=np.float32),
        "sh0": stack("f_dc_").reshape(n, 1, 3),
        "shN": f_rest.reshape(n, 3, k_rest).transpose(0, 2, 1).copy(),
    }
    return torch.nn.ParameterDict(
        {k: torch.nn.Parameter(torch.from_numpy(a).to(device)) for k, a in tensors.items()}
    )
