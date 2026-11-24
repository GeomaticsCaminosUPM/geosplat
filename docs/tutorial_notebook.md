# Tutorial: `examples/run.ipynb`

The notebook runs the whole workflow one step at a time, with maps and 3D views. It works on Google Colab (no
install) or locally. These steps use public sample data, so you don't need photos of your own.

## 0. Open it

- **Colab:** open
  [run.ipynb in Colab](https://colab.research.google.com/github/GeomaticsCaminosUPM/geosplat/blob/main/examples/run.ipynb).
  Choose a GPU runtime: *Runtime ▸ Change runtime type ▸ T4 GPU*.
- **Local:** `uv run --group dev jupyter lab examples/run.ipynb`.

The first cell installs geosplat and checks the GPU.

## 1. First run: `seneca` (drone, GPS)

In section 2 (Options) set:

```python
IMAGES_DIR = "seneca"
PRESET = "very_fast"
```

Then *Runtime ▸ Run all*. What you get, section by section:

| Section | Result |
| --- | --- |
| 3. Photos | Map of the photo positions. Optional: draw a shape to include or exclude photos, then re-run from section 4 |
| 4. COLMAP | Camera poses (cached in `OUTPUT_DIR/colmap/`) |
| 5. Fast orthophoto | First orthophoto over Google Satellite, seconds after COLMAP |
| 6. Training | Gaussian splat |
| 7. Ground control points | Skipped: Seneca has no GCP file |
| 8–9. Final products, results | True orthophoto + DSM, point cloud, interactive 3D splat |
| 10. Output files | Zip of the results (downloaded automatically on Colab) |

## 2. Photos without GPS: `south_building`

```python
IMAGES_DIR = "south_building"
```

The same steps run in a local frame: photo thumbnails instead of a map, orthophotos as plain images.

## 3. Ground control points: `aerial_gcps`

```python
IMAGES_DIR = "aerial_gcps"
PRESET = "very_fast"
```

This downloads 40 survey photos with a positions table (`Cameras_WGS84.txt`) and surveyed targets
(`GCPs_WGS84.txt`), about 440 MB. Both are detected automatically.

In section 7:

1. Choose a point in *Place:* and click the centre of its target on the orthophoto (zoom in). A pink ✕ appears. Do
   this for at least 3 points; drag a ✕ to adjust it.
2. Run the next cell. It prints the fit (method, RMSE, residuals per point).
3. Run the rest. The final orthophoto, DSM and point clouds use the GCP georeference.

The picks are saved to `OUTPUT_DIR/targets/located_targets.json`. `examples/run.py` on the same output folder applies them too.

## 4. Your own photos

Upload a `.zip` of your photos (Colab: *Files* panel ▸ upload) and set:

```python
IMAGES_DIR = "/content/my_flight.zip"
```

| Option | Use |
| --- | --- |
| `IMAGES_DIR` | a `.zip`, a folder, or a sample name (`"seneca"`, `"south_building"`, `"aerial_gcps"`) |
| `OUTPUT_DIR` | `None`: `output_<PRESET>` inside the dataset folder |
| `PRESET` | `"very_fast"` (check), `"fast"` (preview), `"medium"` (good), `"quality"` (final) |
| `TARGETS_ORTO` | subfolder whose photos build a full-resolution orthophoto for locating the targets; only its GCP files are used (`None`: all) |
| `GCPS_CRS` | CRS of a GCP CSV, e.g. `"EPSG:25830"` (`None`: inferred) |

See [Input files](input_files.md) for what the zip can contain.
