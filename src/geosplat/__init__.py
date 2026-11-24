"""geosplat: turn a folder of images into a 3D Gaussian Splatting scene.

Typical use::

    from geosplat import images_to_splat

    result = images_to_splat("my_images/", "output/", preset="fast")
    print(result.outputs)  # splat.ply, points_sparse.ply, points_dense.ply, ortho/*.tif (GPS images)
"""

from . import notebook, plots, positions, sources
from .gaussians import init_gaussians, load_ply, save_ply
from .geo import GeoConfig, GeoReference, georeference
from .pipeline import PipelineResult, images_to_splat
from .presets import PRESETS, make_configs
from .render import render
from .scene import Camera, Scene, load_colmap_scene
from .sfm import SfMConfig, run_sfm
from .trainer import TrainConfig, Trainer, TrainResult, render_camera
from .utils import get_device, gsplat_available
from .video import render_path, save_gif
from .viewer import save_viewer, show
from .ortho import make_fast_orthophoto, make_orthophoto, make_targets_orthophoto
from .pointcloud import export_point_clouds

__version__ = "1.0.0"

__all__ = [
    "Camera",
    "GeoConfig",
    "GeoReference",
    "PRESETS",
    "notebook",
    "positions",
    "sources",
    "plots",
    "PipelineResult",
    "make_configs",
    "export_point_clouds",
    "georeference",
    "make_fast_orthophoto",
    "make_orthophoto", "make_targets_orthophoto",
    "Scene",
    "SfMConfig",
    "TrainConfig",
    "TrainResult",
    "Trainer",
    "get_device",
    "gsplat_available",
    "images_to_splat",
    "init_gaussians",
    "load_colmap_scene",
    "load_ply",
    "render",
    "render_camera",
    "render_path",
    "run_sfm",
    "save_gif",
    "save_ply",
    "save_viewer",
    "show",
]
