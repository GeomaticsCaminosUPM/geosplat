# geosplat documentation

| Page | For |
| --- | --- |
| [Tutorial: `examples/run.py`](tutorial_run_py.md) | Command line, step by step, on public sample data |
| [Tutorial: `examples/run.ipynb`](tutorial_notebook.md) | Notebook (Colab or local), step by step, on public sample data |
| [Input files](input_files.md) | Photo folders, videos, photo positions, ground control points, `located_targets.json` |
| [Main README](../README.md) | Install, outputs, presets, every option, viewers |

The tutorials use three public samples that are downloaded automatically the first time they are used:

| Sample | What it is | Download | Shows |
| --- | --- | --- | --- |
| `seneca` | OpenDroneMap drone flight, GPS in EXIF | 40 photos, ~90 MB | basic run, maps, orthophotos |
| `south_building` | COLMAP hand-held photos, no GPS | ~420 MB archive | local (non-georeferenced) mode |
| `aerial_gcps` | Agisoft sample survey (data by GeoScan): positions table + surveyed GCPs | 40 photos, ~440 MB | positions file, ground control points, locating targets |

Sample data stays under its owners' terms: Seneca from
[OpenDroneMap/odm_data_seneca](https://github.com/OpenDroneMap/odm_data_seneca), South Building from
[COLMAP](https://colmap.github.io/datasets.html), and "Aerial images with GCPs" from the
[Agisoft sample data](https://www.agisoft.com/downloads/sample-data/) (provided by GeoScan).
