"""Speed / quality presets that fill in :class:`SfMConfig`, :class:`TrainConfig` and :class:`GeoConfig`.

Presets only set a handful of fields; everything else keeps its default, and
any explicit override wins over the preset.

Measured on 40 drone photos (4000x3000) with an RTX 4060 Laptop GPU:

=========  =================  ====================================================
preset     time (drone2024)    what it does
=========  =================  ====================================================
very_fast  ~1 min             800 px images, 1k steps, <=150k Gaussians, SH degree 1
fast       ~2 min             1200 px images, 3k steps, <=500k Gaussians, early stop
medium     ~13 min            1600 px images, 7k steps, auto GPU memory budget
quality    ~50-60 min (est.)  2400 px SfM / finer ortho, 30k steps (3DGS paper)
=========  =================  ====================================================
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from .geo import GeoConfig
from .sfm import SfMConfig
from .trainer import TrainConfig

PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "very_fast": {
        "sfm": {"max_image_size": 800, "max_num_features": 4096, "video_frames": 15},
        "train": {
            "iterations": 1000, "max_resolution": 800, "max_gaussians": 150_000, "sh_degree": 1,
            "sh_degree_interval": 300, "refine_start_iter": 200, "refine_every": 50,
        },
        "geo": {"ortho_fullres": False},
    },
    "fast": {
        "sfm": {"max_image_size": 1200, "video_frames": 30},
        "train": {"iterations": 3000, "max_resolution": 1200, "max_gaussians": 500_000, "early_stop": True},
        "geo": {},
    },
    "medium": {
        "sfm": {"max_image_size": 1600, "video_frames": 50},
        "train": {"iterations": 7000, "max_resolution": 1600},
        "geo": {},
    },
    "quality": {
        "sfm": {"max_image_size": 2400, "video_frames": 100},
        "train": {"iterations": 30_000, "max_resolution": 1600},
        "geo": {},
    },
}

DEFAULT_PRESET = "medium"


def _build(cls: type, values: dict[str, Any]) -> Any:
    """Instantiate a config dataclass, rejecting unknown field names.

    Args:
        cls: Dataclass type (``SfMConfig``, ``TrainConfig`` or ``GeoConfig``).
        values: Field values.

    Returns:
        The dataclass instance.

    Raises:
        ValueError: If ``values`` contains a field that ``cls`` does not have.
    """
    names = {f.name for f in fields(cls)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} option(s): {sorted(unknown)}")
    return cls(**values)


def make_configs(
    preset: str = DEFAULT_PRESET,
    sfm: dict[str, Any] | None = None,
    train: dict[str, Any] | None = None,
    geo: dict[str, Any] | None = None,
) -> tuple[SfMConfig, TrainConfig, GeoConfig]:
    """Build the three configs from a preset plus explicit overrides.

    Args:
        preset: ``"very_fast"``, ``"fast"``, ``"medium"`` or ``"quality"``
            (``-`` is accepted instead of ``_``).
        sfm: :class:`SfMConfig` fields that override the preset (``None`` values are ignored).
        train: :class:`TrainConfig` fields that override the preset.
        geo: :class:`GeoConfig` fields that override the preset.

    Returns:
        Tuple ``(sfm_config, train_config, geo_config)``.

    Raises:
        ValueError: If the preset name or an override field is unknown.

    Example:
        >>> sfm, train, geo = make_configs("fast", train={"iterations": 4000}, geo={"crs": "EPSG:25830"})
    """
    preset = preset.replace("-", "_")
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose one of {sorted(PRESETS)}")
    p = PRESETS[preset]

    def merge(base: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
        """Overlay non-``None`` overrides on preset values.

        Args:
            base: Preset values.
            extra: Overrides.

        Returns:
            Merged values.
        """
        return {**base, **{k: v for k, v in (extra or {}).items() if v is not None}}

    return (
        _build(SfMConfig, merge(p["sfm"], sfm)),
        _build(TrainConfig, merge(p["train"], train)),
        _build(GeoConfig, merge(p["geo"], geo)),
    )
