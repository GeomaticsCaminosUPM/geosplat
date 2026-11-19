"""Rendering smooth camera fly-throughs of a trained model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .scene import Camera
from .trainer import render_camera


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a unit ``wxyz`` quaternion.

    Args:
        R: Rotation matrix ``[3, 3]``.

    Returns:
        Quaternion ``[4]``.
    """
    m = R
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert a unit ``wxyz`` quaternion to a rotation matrix.

    Args:
        q: Quaternion ``[4]``.

    Returns:
        Rotation matrix ``[3, 3]``.
    """
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between quaternions.

    Args:
        q0: Start quaternion ``[4]``.
        q1: End quaternion ``[4]``.
        t: Interpolation factor in ``[0, 1]``.

    Returns:
        Interpolated unit quaternion ``[4]``.
    """
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(dot)
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


def interpolate_cameras(cameras: list[Camera], num_frames: int = 120) -> list[Camera]:
    """Create a smooth path through the given cameras (in list order).

    Camera centers are linearly interpolated and orientations slerped.
    Intrinsics and resolution are taken from the first camera.

    Args:
        cameras: Key-frame cameras (e.g. training cameras sorted by name).
        num_frames: Number of output cameras.

    Returns:
        Interpolated cameras.
    """
    if len(cameras) == 1:
        return [cameras[0]] * num_frames
    ref = cameras[0]
    quats = [_rotmat_to_quat(c.world_to_cam[:3, :3]) for c in cameras]
    centers = [c.center for c in cameras]
    out = []
    for u in np.linspace(0, len(cameras) - 1, num_frames):
        i = min(int(np.floor(u)), len(cameras) - 2)
        t = u - i
        R = _quat_to_rotmat(_slerp(quats[i], quats[i + 1], t))
        center = (1 - t) * centers[i] + t * centers[i + 1]
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = -R @ center
        out.append(Camera(f"frame_{len(out):04d}", ref.image_path, ref.K.copy(), w2c, ref.width, ref.height))
    return out


def render_path(
    params: torch.nn.ParameterDict | dict[str, torch.Tensor],
    cameras: list[Camera],
    sh_degree: int,
    backend: str,
    num_frames: int = 120,
) -> list[np.ndarray]:
    """Render frames along a camera path interpolated through ``cameras``.

    Args:
        params: Trained Gaussian parameters.
        cameras: Key-frame cameras.
        sh_degree: SH degree of the model.
        backend: ``"gsplat"`` or ``"torch"``.
        num_frames: Number of frames.

    Returns:
        List of ``uint8`` frames of shape ``[H, W, 3]``.
    """
    frames = []
    for cam in interpolate_cameras(cameras, num_frames):
        img = render_camera(params, cam, sh_degree, backend)
        frames.append((img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    return frames


def save_gif(frames: list[np.ndarray], path: str | Path, fps: int = 24, max_width: int = 640) -> Path:
    """Save frames as an animated GIF.

    Args:
        frames: ``uint8`` frames ``[H, W, 3]``.
        path: Output ``.gif`` path.
        fps: Frames per second.
        max_width: Frames wider than this are downscaled to keep the file small.

    Returns:
        The written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for f in frames:
        im = Image.fromarray(f)
        if im.width > max_width:
            im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
        images.append(im)
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
    return path
