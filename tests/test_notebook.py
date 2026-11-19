"""End-to-end test of ``examples/run.ipynb`` with the ``very_fast`` preset (``pytest --run-slow``).

Test copies are generated from the example notebook into ``tests/notebooks/``
(so they never drift from it) and executed there; executed copies are kept
next to them for inspection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples" / "run.ipynb"
OUT_DIR = Path(__file__).resolve().parent / "notebooks"

DATASETS = {
    # name: (IMAGES_DIR relative to the repo root, SAMPLE value, needs local photos)
    "seneca": ("examples/seneca/images", '"seneca"', False),
    "dron2024": ("examples/dron2024/images", "None", True),
}


def make_test_notebook(name: str) -> Path:
    """Write a ``very_fast`` copy of the example notebook for one dataset.

    Args:
        name: Key of :data:`DATASETS`.

    Returns:
        Path of the generated notebook in ``tests/notebooks/``.
    """
    images, sample, _ = DATASETS[name]
    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        src = src.replace('PRESET = "fast"', 'PRESET = "very_fast"')
        src = src.replace('IMAGES_DIR = Path("examples/seneca/images")', f'IMAGES_DIR = Path("{images}")')
        src = src.replace('SAMPLE = "seneca"', f"SAMPLE = {sample}")
        cell["source"] = src
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"run_{name}_very_fast.ipynb"
    path.write_text(json.dumps(nb, indent=1))
    return path


@pytest.mark.slow
@pytest.mark.parametrize("name", list(DATASETS))
def test_notebook_very_fast(name: str) -> None:
    """The example notebook runs end to end without errors.

    Args:
        name: Dataset key.
    """
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    images, _, needs_local = DATASETS[name]
    if needs_local and not (ROOT / images).is_dir():
        pytest.skip(f"{images} not present")
    path = make_test_notebook(name)
    nb = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(nb, timeout=1800, kernel_name="python3",
                                     resources={"metadata": {"path": str(OUT_DIR)}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, OUT_DIR / f"run_{name}_very_fast.executed.ipynb")
    errors = [o for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, errors
