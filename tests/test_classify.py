"""Tests for the per-Gaussian surface / vegetation features and their automatic thresholds."""

from __future__ import annotations

import numpy as np
import torch

from geosplat.classify import classify_scene, gaussian_features, otsu
from geosplat.sh import rgb_to_sh


def _params(n_ground: int = 4000, n_tree: int = 4000, seed: int = 0) -> dict[str, torch.Tensor]:
    """A flat ground of opaque discs (grey half, green grass half) and a green, semi-transparent, volumetric crown.

    Args:
        n_ground: Ground Gaussians.
        n_tree: Tree Gaussians.
        seed: Random seed.

    Returns:
        Gaussian parameters (unconstrained form).
    """
    g = torch.Generator().manual_seed(seed)
    ground = torch.cat([torch.rand(n_ground, 2, generator=g) * 10, torch.zeros(n_ground, 1)], 1)
    tree = torch.randn(n_tree, 3, generator=g) * 1.0 + torch.tensor([5.0, 5.0, 4.0])
    means = torch.cat([ground, tree])
    scales = torch.cat([torch.tensor([[0.1, 0.1, 0.005]]).expand(n_ground, 3),
                        torch.tensor([[0.08, 0.06, 0.05]]).expand(n_tree, 3)]).log()
    quats = torch.cat([torch.tensor([[1.0, 0, 0, 0]]).expand(n_ground, 4), torch.randn(n_tree, 4, generator=g)])
    grass = (ground[:, 0] > 5)[:, None]
    ground_col = torch.where(grass, torch.tensor([[0.3, 0.6, 0.2]]), torch.tensor([[0.55, 0.55, 0.55]]))
    colors = torch.cat([ground_col, torch.tensor([[0.2, 0.6, 0.15]]).expand(n_tree, 3)])
    opac = torch.cat([torch.full((n_ground,), 0.95), torch.full((n_tree,), 0.35)])
    return {"means": means, "scales": scales, "quats": quats, "sh0": rgb_to_sh(colors)[:, None],
            "opacities": torch.logit(opac)}


def test_features_separate_ground_and_tree() -> None:
    """Ground is grey, flat, coherent and layered; the crown is green and volumetric."""
    f = gaussian_features(_params())
    ground, tree = slice(0, 4000), slice(4000, 8000)
    grey = np.flatnonzero(f["greenness"][ground] < 0.1)
    assert f["greenness"][tree].mean() > 0.3 > f["greenness"][grey].mean()
    assert f["flatness"][ground].mean() > 0.9 > f["flatness"][tree].mean()
    assert f["coherence_medium"][ground].mean() > f["coherence_medium"][tree].mean()
    assert f["scatter_medium"][tree].mean() > f["scatter_medium"][ground].mean()


def test_otsu() -> None:
    """Otsu splits a bimodal sample between its modes."""
    v = np.concatenate([np.random.default_rng(0).normal(0, 0.1, 5000), np.random.default_rng(1).normal(1, 0.1, 5000)])
    t, sep = otsu(v)
    assert 0.3 < t < 0.7 and sep > 0.8


def _standing_scene() -> dict[str, torch.Tensor]:
    """Flat ground (scene up = -Y), an opaque grey wall and a green, semi-transparent, volumetric crown.

    Returns:
        Gaussian parameters (unconstrained form).
    """
    g = torch.Generator().manual_seed(0)
    ground = torch.cat([torch.rand(20000, 1, generator=g) * 40, torch.zeros(20000, 1),
                        torch.rand(20000, 1, generator=g) * 40], 1)
    wall = torch.cat([torch.rand(4000, 1, generator=g) * 8 + 6, -torch.rand(4000, 1, generator=g) * 3,
                      torch.full((4000, 1), 20.0) + 0.05 * torch.randn(4000, 1, generator=g)], 1)
    crown = torch.randn(4000, 3, generator=g) * 1.2 + torch.tensor([30.0, -4.0, 20.0])
    means = torch.cat([ground, wall, crown])
    n_g, n_w, n_c = len(ground), len(wall), len(crown)
    scales = torch.cat([torch.tensor([[0.15, 0.005, 0.15]]).expand(n_g, 3), torch.tensor([[0.15, 0.15, 0.005]]).expand(n_w, 3),
                        torch.tensor([[0.08, 0.06, 0.05]]).expand(n_c, 3)]).log()
    quats = torch.cat([torch.tensor([[1.0, 0, 0, 0]]).expand(n_g + n_w, 4), torch.randn(n_c, 4, generator=g)])
    colors = torch.cat([torch.tensor([[0.5, 0.48, 0.45]]).expand(n_g, 3), torch.tensor([[0.6, 0.6, 0.6]]).expand(n_w, 3),
                        torch.tensor([[0.15, 0.4, 0.12]]).expand(n_c, 3)])
    opac = torch.cat([torch.full((n_g + n_w,), 0.97), torch.full((n_c,), 0.35)])
    return {"means": means, "scales": scales, "quats": quats, "sh0": rgb_to_sh(colors)[:, None],
            "opacities": torch.logit(opac)}


def test_classify_scene_tree_and_structure() -> None:
    """The crown is a tree object, the wall a structure object, the ground no object."""
    c = classify_scene(_standing_scene(), None, m_per_unit=1.0)
    ids, votes = c["ids"], c["object_votes"]
    wall_id = np.bincount(ids[20000:24000]).argmax()
    crown_id = np.bincount(ids[24000:]).argmax()
    assert wall_id > 0 and crown_id > 0 and wall_id != crown_id
    assert votes[wall_id] > votes[crown_id]
    assert (ids[:20000] == 0).mean() > 0.95


def test_segment_objects_separates_blobs_from_ground() -> None:
    """Two separate blobs on a flat ground become two objects; the ground stays 0."""
    from geosplat.classify import segment_objects

    g = torch.Generator().manual_seed(0)
    ground = torch.cat([torch.rand(20000, 1, generator=g) * 40, torch.zeros(20000, 1),
                        torch.rand(20000, 1, generator=g) * 40], 1)          # scene up = -Y
    blobs = [torch.randn(3000, 3, generator=g) * 0.8 + torch.tensor([cx, -3.0, 20.0]) for cx in (10.0, 30.0)]
    means = torch.cat([ground, *blobs])
    n = len(means)
    params = {"means": means, "scales": torch.full((n, 3), 0.05).log(), "quats": torch.tensor([[1.0, 0, 0, 0]]).expand(n, 4),
              "sh0": torch.zeros(n, 1, 3), "opacities": torch.full((n,), 3.0)}
    ids = segment_objects(params, None, m_per_unit=1.0)["ids"]
    assert ids.max() == 2
    assert (ids[:20000] == 0).mean() > 0.95
    a, b = ids[20000:23000], ids[23000:]
    assert np.bincount(a).argmax() != np.bincount(b).argmax() and (a > 0).mean() > 0.8 and (b > 0).mean() > 0.8
