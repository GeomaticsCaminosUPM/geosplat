"""Loading COLMAP reconstructions into a simple camera / image / point container."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from PIL import Image

PINHOLE_MODELS = {"PINHOLE", "SIMPLE_PINHOLE"}


@dataclass
class Camera:
    """A calibrated pinhole camera with its image.

    Attributes:
        name: Image file name.
        image_path: Path to the (undistorted) image on disk.
        K: Intrinsic matrix of shape ``[3, 3]`` at full image resolution.
        world_to_cam: World-to-camera (OpenCV convention) matrix of shape ``[4, 4]``.
        width: Image width in pixels.
        height: Image height in pixels.
    """

    name: str
    image_path: Path
    K: np.ndarray
    world_to_cam: np.ndarray
    width: int
    height: int

    @property
    def center(self) -> np.ndarray:
        """Camera center in world coordinates.

        Returns:
            Array of shape ``[3]``.
        """
        R, t = self.world_to_cam[:3, :3], self.world_to_cam[:3, 3]
        return -R.T @ t

    def scaled(self, factor: float) -> "Camera":
        """Return a copy of the camera for an image resized by ``factor``.

        Args:
            factor: Resize factor (``0.5`` halves the resolution).

        Returns:
            A new :class:`Camera` with scaled intrinsics and size.
        """
        K = self.K.copy()
        K[:2] *= factor
        return Camera(
            self.name, self.image_path, K, self.world_to_cam.copy(),
            max(1, round(self.width * factor)), max(1, round(self.height * factor)),
        )


@dataclass
class Scene:
    """Cameras and sparse colored points of a reconstructed scene.

    Attributes:
        cameras: Registered cameras, sorted by image name.
        points: Sparse 3D points of shape ``[N, 3]``.
        colors: Point colors of shape ``[N, 3]`` (``uint8``).
        transform: ``[4, 4]`` matrix mapping the original COLMAP world
            coordinates to the scene's coordinates (identity if not normalized).
    """

    cameras: list[Camera]
    points: np.ndarray
    colors: np.ndarray
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    _scale: float | None = field(default=None, repr=False)

    @property
    def scene_scale(self) -> float:
        """Radius of the camera rig, used to scale learning rates and thresholds.

        Returns:
            ``1.1 *`` the maximum distance of a camera center to the mean center.
        """
        if self._scale is None:
            centers = np.stack([c.center for c in self.cameras])
            dist = np.linalg.norm(centers - centers.mean(0), axis=1)
            self._scale = float(max(dist.max(), 1e-3) * 1.1)
        return self._scale


def _cam_from_world(image: pycolmap.Image) -> np.ndarray:
    """Get the ``[4, 4]`` world-to-camera matrix of a pycolmap image.

    Supports both pycolmap 3.x (attribute) and 4.x (method) APIs.

    Args:
        image: A registered pycolmap image.

    Returns:
        World-to-camera matrix of shape ``[4, 4]``.
    """
    pose = image.cam_from_world
    pose = pose() if callable(pose) else pose
    mat = np.eye(4)
    mat[:3, :4] = pose.matrix()
    return mat


def find_sparse_dir(colmap_dir: str | Path) -> Path:
    """Locate the sparse model inside a COLMAP project folder.

    Args:
        colmap_dir: Folder containing ``sparse/`` or ``sparse/0``.

    Returns:
        Path of the folder that holds ``cameras.bin`` / ``cameras.txt``.

    Raises:
        FileNotFoundError: If no model is found.
    """
    colmap_dir = Path(colmap_dir)
    for cand in (colmap_dir / "sparse" / "0", colmap_dir / "sparse", colmap_dir):
        if (cand / "cameras.bin").exists() or (cand / "cameras.txt").exists():
            return cand
    raise FileNotFoundError(f"No COLMAP sparse model found in {colmap_dir}")


def is_colmap_project(path: str | Path) -> bool:
    """Whether ``path`` looks like a COLMAP project with ``images/`` and a sparse model.

    Args:
        path: Folder to check.

    Returns:
        ``True`` if it contains an ``images`` folder and a sparse model.
    """
    try:
        find_sparse_dir(path)
    except FileNotFoundError:
        return False
    return (Path(path) / "images").is_dir()


def normalize_scene(
    scene: Scene, down: np.ndarray | None = None, forward: np.ndarray | None = None
) -> Scene:
    """Center the scene and make it upright in the convention 3DGS viewers expect.

    The new origin is the median of the sparse points. The world is rotated so
    "down" becomes ``+Y`` (i.e. up is ``-Y``, the OpenCV / COLMAP / 3DGS
    convention used by SuperSplat and the SIBR viewer) and ``forward`` becomes
    ``+Z``. No scaling is applied.

    Without explicit directions, "down" is the average image-down vector of
    the cameras (hand-held captures) or, for nadir captures such as drone
    mapping flights (cameras looking roughly the same way with arbitrary roll),
    the average viewing direction.

    Args:
        scene: Scene in COLMAP world coordinates.
        down: Optional gravity direction in the input world coordinates
            (e.g. from GPS georeferencing).
        forward: Optional direction to map to ``+Z`` (projected to be
            perpendicular to ``down``).

    Returns:
        A new :class:`Scene` with transformed cameras and points, whose
        ``transform`` maps original to new coordinates.
    """
    w2cs = np.stack([c.world_to_cam for c in scene.cameras])
    cam_down = w2cs[:, 1, :3].mean(0)  # camera +Y (image down) expressed in world
    cam_fwd = w2cs[:, 2, :3].mean(0)  # camera +Z (viewing direction)
    if down is None:
        nadir = np.linalg.norm(cam_down) < 0.5 and np.linalg.norm(cam_fwd) > 0.8
        down = cam_fwd if nadir else cam_down
    down = np.asarray(down, dtype=np.float64) / np.linalg.norm(down)
    if forward is None:
        forward = cam_fwd
    forward = np.asarray(forward, dtype=np.float64)
    forward = forward - forward.dot(down) * down
    if np.linalg.norm(forward) < 1e-3:  # e.g. nadir cameras: pick any horizontal axis
        forward = np.cross(down, [1.0, 0.0, 0.0] if abs(down[0]) < 0.9 else [0.0, 1.0, 0.0])
    forward /= np.linalg.norm(forward)
    right = np.cross(down, forward)
    R = np.stack([right, down, forward])  # rows: new x, y (down), z axes in old world
    if len(scene.points):
        center = np.median(scene.points, axis=0)
    else:
        center = np.stack([c.center for c in scene.cameras]).mean(0)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ center
    T_inv = np.linalg.inv(T)

    cameras = [
        Camera(c.name, c.image_path, c.K.copy(), c.world_to_cam @ T_inv, c.width, c.height)
        for c in scene.cameras
    ]
    points = ((scene.points - center) @ R.T).astype(np.float32)
    return Scene(cameras, points, scene.colors.copy(), T @ scene.transform)


def load_colmap_scene(
    colmap_dir: str | Path, images_subdir: str = "images", normalize: bool = True
) -> Scene:
    """Load an undistorted COLMAP project.

    Args:
        colmap_dir: Folder containing ``images/`` and ``sparse/`` (as produced
            by :func:`geosplat.sfm.run_sfm` or ``colmap image_undistorter``).
        images_subdir: Name of the image folder inside ``colmap_dir``.
        normalize: Center and orient the scene with :func:`normalize_scene` so
            it appears upright and centered in viewers.

    Returns:
        The loaded :class:`Scene`.

    Raises:
        ValueError: If a camera uses a distorted (non-pinhole) model.
    """
    colmap_dir = Path(colmap_dir)
    rec = pycolmap.Reconstruction(str(find_sparse_dir(colmap_dir)))
    cameras: list[Camera] = []
    for image in rec.images.values():
        if hasattr(image, "has_pose") and not image.has_pose:
            continue
        cam = rec.cameras[image.camera_id]
        if cam.model_name not in PINHOLE_MODELS:
            raise ValueError(
                f"Camera model {cam.model_name} has distortion; undistort the images first "
                "(geosplat.sfm.run_sfm does this automatically)."
            )
        path = colmap_dir / images_subdir / image.name
        # Images may have been resized relative to the COLMAP calibration.
        with Image.open(path) as im:
            w, h = im.size
        K = np.asarray(cam.calibration_matrix(), dtype=np.float64)
        K[0] *= w / cam.width
        K[1] *= h / cam.height
        cameras.append(Camera(image.name, path, K, _cam_from_world(image), w, h))
    cameras.sort(key=lambda c: c.name)

    if rec.num_points3D() > 0:
        pts = np.stack([p.xyz for p in rec.points3D.values()]).astype(np.float32)
        cols = np.stack([p.color for p in rec.points3D.values()]).astype(np.uint8)
    else:
        pts = np.zeros((0, 3), np.float32)
        cols = np.zeros((0, 3), np.uint8)
    scene = Scene(cameras, pts, cols)
    return normalize_scene(scene) if normalize else scene


class ImageDataset:
    """Images of a :class:`Scene` loaded in memory at training resolution.

    Images are stored as ``uint8`` tensors to keep memory low and converted to
    float on access.

    Args:
        cameras: Cameras to load.
        max_resolution: Longest image side used for training (``0`` keeps the
            original resolution).
        device: Where image tensors are cached. Using the GPU avoids host to
            device copies but costs GPU memory.
    """

    def __init__(
        self, cameras: list[Camera], max_resolution: int = 1600, device: torch.device | str = "cpu"
    ) -> None:
        """Load and resize every image.

        Args:
            cameras: Cameras to load.
            max_resolution: Longest image side used for training.
            device: Device for cached image tensors.
        """
        self.cameras: list[Camera] = []
        self.images: list[torch.Tensor] = []
        for cam in cameras:
            factor = 1.0
            if max_resolution and max(cam.width, cam.height) > max_resolution:
                factor = max_resolution / max(cam.width, cam.height)
            scaled = cam.scaled(factor)
            with Image.open(cam.image_path) as im:
                arr = torch.from_numpy(np.asarray(im.convert("RGB"), dtype=np.uint8).copy())
            img = arr.to(device)
            if (img.shape[1], img.shape[0]) != (scaled.width, scaled.height):
                # Resize on the target device (GPU when training on CUDA).
                img = F.interpolate(
                    img.permute(2, 0, 1)[None].float(), size=(scaled.height, scaled.width),
                    mode="bilinear", antialias=True, align_corners=False,
                )[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)
            self.cameras.append(scaled)
            self.images.append(img.contiguous())

    def __len__(self) -> int:
        """Number of images.

        Returns:
            Dataset length.
        """
        return len(self.cameras)

    def __getitem__(self, idx: int) -> tuple[Camera, torch.Tensor]:
        """Get a camera and its image.

        Args:
            idx: Image index.

        Returns:
            Tuple ``(camera, image)`` where ``image`` is float ``[H, W, 3]`` in ``[0, 1]``.
        """
        return self.cameras[idx], self.images[idx].float() / 255.0


def split_train_test(cameras: list[Camera], test_every: int = 8) -> tuple[list[Camera], list[Camera]]:
    """Split cameras into train / test sets (every ``test_every``-th is test).

    Args:
        cameras: All cameras, in a stable order.
        test_every: Hold out one of every ``test_every`` images. ``0`` disables
            the split (all images are used for training, test set is empty).

    Returns:
        Tuple ``(train_cameras, test_cameras)``.
    """
    if test_every <= 0 or len(cameras) < 2 * test_every:
        return list(cameras), []
    test = [c for i, c in enumerate(cameras) if i % test_every == 0]
    train = [c for i, c in enumerate(cameras) if i % test_every != 0]
    return train, test
