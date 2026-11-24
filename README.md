# geosplat

**Version 1.0.0**

<a href="https://colab.research.google.com/github/GeomaticsCaminosUPM/geosplat/blob/main/examples/run.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Turn a folder of photos into a **fast orthophoto**, a **point cloud**, a **3D Gaussian Splatting** scene and a
**true orthophoto + DSM**. GPS-tagged photos (e.g. drones) give georeferenced outputs in a real CRS.

```
images/ ─► COLMAP SfM (GPU) ─► georeference (EXIF GPS) ─► fast_orto/fast_orto.tif   (seconds)
                                                     └─► Gaussian Splatting training (GPU)
                                                              ├─► splat.ply, viewer.html  (3DGS scene)
                                                              ├─► points_sparse.ply / points_dense.ply
                                                              └─► ortho/orthophoto.tif, dsm.tif
```

Without GPS, orthophotos are produced in a local frame (no CRS, SfM units). With ground control points, locate
the targets on the orthophoto in `gcp/picker.html` and run again: every product is corrected.

**Tutorials** on public sample data (nothing of your own needed): [run.py](docs/tutorial_run_py.md) ·
[notebook](docs/tutorial_notebook.md) · [input files](docs/input_files.md).

- **SfM:** `pycolmap` with CUDA (`pycolmap-cuda12` on Linux x86_64), covering SIFT extraction, matching and bundle
  adjustment. Matching is exhaustive, sequential, or GPS-guided spatial.
- **Training:** `gsplat` CUDA kernels with the 3DGS recipe. It adds a far "background sphere" so sky and distant
  background don't become floaters, and an optional MCMC strategy that caps the Gaussian count.
- **Georeferencing:** camera GPS from EXIF goes through a similarity fit to a projected CRS (auto UTM, or e.g.
  `EPSG:25830`). The result is metric, Z-up and exportable in real coordinates.
- **Orthophoto:** the original photos are orthorectified on the splat's surface model. Occluded views are rejected
  with depth maps rendered from the splat, and the most nadir view wins. It all runs in torch on the GPU and is
  written as GeoTIFF.
- **GPU everywhere it matters:** SfM, training, densification, background coloring, orthorectification and rendering.
  A pure-PyTorch rasterizer exists for machines without CUDA, but it is very slow.

## Install

Requires [uv](https://docs.astral.sh/uv/) and, for GPU use, an NVIDIA driver plus the CUDA toolkit (`nvcc`).

```bash
git clone https://github.com/GeomaticsCaminosUPM/geosplat.git && cd geosplat
uv sync                       # or, in any environment: pip install git+https://github.com/GeomaticsCaminosUPM/geosplat.git
```

gsplat compiles its CUDA kernels **the first time it runs** (2–5 min, then cached in `~/.cache/torch_extensions`).
geosplat finds `nvcc` on `PATH` or in `/usr/local/cuda*`; you can also set `CUDA_HOME`.

### Faster start on Google Colab (prebuilt gsplat)

On Colab the notebook first tries to install a **prebuilt gsplat wheel** from this repository's GitHub releases. That
takes seconds instead of the 3–5 min compile. Wheels are built by the **gsplat wheels** workflow
(`.github/workflows/gsplat-wheels.yml`) for a given Python / torch / CUDA combination:

1. Open the notebook in Colab. If no wheel matches, the setup cell prints the versions to use, e.g.
   `python=3.13, torch=2.8.0, cuda=12.6, gsplat=1.5.3`.
2. On GitHub: **Actions ▸ gsplat wheels ▸ Run workflow**, enter those versions (several can be given,
   comma-separated) and run. A build takes about 15–30 min on GitHub's free runners.
3. The next Colab session installs the wheel automatically. Re-run the workflow when Colab upgrades torch.

Without a matching wheel the CUDA code is compiled once per Colab session (3–5 min).

## Quick start

```bash
uv run geosplat path/to/images -o path/to/output --preset medium      # very_fast | fast | medium | quality
```

```python
from geosplat import images_to_splat

result = images_to_splat("path/to/images", "path/to/output", preset="medium")
print(result.outputs)   # fast_orto, splat, viewer, points_sparse, points_dense, orthophoto, dsm, ...
print(result.timings)   # seconds per stage
```

Or open the notebook: [`examples/run.ipynb`](examples/run.ipynb) (button above for Colab).

If the input folder is already an undistorted COLMAP project (it has `images/` and `sparse/`), SfM is skipped.

## Outputs

```
output/
├── splat.ply                  # Gaussian splat (3DGS format) → SuperSplat / any 3DGS viewer
├── viewer.html                # interactive splat viewer: open in a browser
├── points_sparse.ply          # SfM tie points (colored)
├── points_dense.ply           # dense colored points (centers of solid Gaussians)
├── geo.json                   # georeference: CRS, origin, scene→CRS transform, GPS fit RMSE
├── ortho_targets/             # with targets_orto: full-resolution orthophoto of that flight (+ dsm.tif)
├── gcp/                       # with GCPs and positions:
│   ├── picker.html            #   locate the targets in a browser, "Save picks.json"
│   ├── picks.json             #   your picks (saved by the page); applied when you run again
│   └── report.json            #   GCP fit: method, residuals, RMSE
├── fast_orto/                 # FIRST orthophoto, seconds after COLMAP (before training)
│   ├── fast_orto.tif          # photos orthorectified on a DSM interpolated from SfM points
│   ├── dsm_sparse.tif         # that coarse DSM
│   └── fast_orto_preview.jpg
├── ortho/
│   ├── orthophoto.tif         # true orthophoto from the photos (RGBA GeoTIFF)
│   ├── orthophoto_splat.tif   # orthographic render of the splat
│   ├── dsm.tif                # digital surface model (float32 GeoTIFF, nodata -9999)
│   ├── orthophoto_preview.jpg # preview with GCPs marked in red
│   └── gcp_check.json         # DSM height minus GCP height per point (if --gcps)
├── renders/                   # held-out photo | render, side by side
├── metrics.json, timings.json
└── colmap/                    # SfM work dir (database, sparse model, undistorted images)
```

With a georeference, point clouds are written in CRS coordinates (float64) and line up with the GeoTIFFs. The splat
is kept in a local frame: centered, Y-down/up = −Y, the convention 3DGS viewers expect. `geo.json` maps it to the
CRS.

## Speed presets

One option sets the speed/quality trade-off. Times are measured on the `drone2024` example (40 drone photos,
4000×3000) with an RTX 4060 Laptop GPU:

| Preset | Total time | COLMAP | Training | Settings |
| --- | --- | --- | --- | --- |
| `very_fast` | **34 s** | 21 s | 11 s | 800 px, 1,000 steps, ≤150k Gaussians, SH degree 1 |
| `fast` | **2 min 15 s** | 42 s | 88 s | 1200 px, 3,000 steps, ≤500k Gaussians, early stop |
| `medium` (default) | **~13 min** | 72 s | 705 s | 1600 px, 7,000 steps, model capped to free GPU memory |
| `quality` | ~1 h (estimated) | | | 2400 px for SfM / orthophoto, 30,000 steps |

`fast_orto`, point clouds and the final orthophoto take a few seconds in every preset. Use `very_fast` to check a
dataset, `fast` for previews, `medium` for good results and `quality` for final products.

### Default run (medium)

```bash
uv run geosplat path/to/images -o path/to/output --preset medium
uv run python examples/run.py drone2024 --preset medium
```

### Choosing a preset

```bash
uv run geosplat path/to/images -o out_very_fast --preset very_fast
uv run geosplat path/to/images -o out_fast      --preset fast
uv run geosplat path/to/images -o out_quality   --preset quality
uv run python examples/run.py drone2024 --preset fast                    # writes examples/drone2024/output_fast/
```

```python
from geosplat import images_to_splat, make_configs

result = images_to_splat("path/to/images", "out", preset="fast")

# A preset with a few changes:
sfm, train, geo = make_configs("fast", train={"iterations": 4000}, geo={"crs": "EPSG:25830"})
result = images_to_splat("path/to/images", "out", sfm, train, geo)
```

Flags given together with `--preset` override it (e.g. `--preset fast --iterations 4000`). In an example folder,
`config.json` may set `"preset"` as well.

## All speed options

Individual settings, for fine control (each one overrides the preset):

| Option (Python / CLI) | Effect |
| --- | --- |
| `SfMConfig(max_images=N)` / `--max-images N` | Use N evenly spaced images |
| `SfMConfig(image_stride=k)` / `--image-stride k` | Use every k-th image |
| `SfMConfig(polygon="aoi.gpkg", polygon_buffer=20)` / `--polygon aoi.gpkg` | Only GPS photos inside an area (`.gpkg`, `.shp`, `.geojson`, any CRS) |
| `SfMConfig(max_image_size=1000)` / `--max-image-size 1000` | Smaller images for SfM, training and the ortho |
| `matcher="spatial"` (default `auto` with GPS) | Match only nearby photos: linear instead of quadratic |
| `mapper="global"` / `--mapper global` | GLOMAP global SfM, much faster on large sets |
| `TrainConfig(iterations=3000)` / `--iterations 3000` | Shorter training (7000 is a good preview, 30000 full quality) |
| `TrainConfig(max_resolution=1000)` / `--max-resolution 1000` | Train on smaller images |
| `TrainConfig(max_gaussians=300_000)` / `--max-gaussians 300000` | Cap the model size (MCMC): bounded memory and time per step |
| `TrainConfig(sh_degree=1)` / `--sh-degree 1` | Less view-dependent color; fewer parameters |
| `TrainConfig(early_stop=True)` / `--early-stop` | Stop once held-out PSNR plateaus (checked after densification) |
| `GeoConfig(gsd=0.05)` / `--gsd 0.05` | Coarser orthophoto (fewer pixels) |
| `GeoConfig(dense_min_opacity=0.5)` / `--dense-min-opacity 0.5` | Opacity threshold for `points_dense.ply` (default `"auto"`: Otsu on the opacity histogram; higher = fewer, cleaner points) |
| `progressive_resolution=True` (default) | Coarse-to-fine: first 10 % of steps at 1/4 resolution, next 15 % at 1/2 |
| `selective_adam=True` (default) | Fused CUDA optimizer that only updates Gaussians visible in the current view |
| `gaussian_budget="auto"` (default) | Caps the model at what fits in free GPU memory (no out-of-memory crashes) |

Always on: SIFT extraction, matching and bundle adjustment run on the GPU. Images are resized on the GPU and stay
cached there, and camera matrices stay on the GPU too. SSIM uses separable, batched GPU convolutions. Background
coloring, orthorectification and DSM rendering are GPU tensor code. Image decoding and COLMAP undistortion are the
only CPU steps; they take a few seconds.

## Examples

Each dataset has its own folder in `examples/` (photos in `images/`, results in `output*/`). No configuration is
needed. With GPS photos, positions and the CRS (local UTM zone) come from the photo metadata, and a `name,x,y,z`
CSV in `images/` is used as ground control points. An optional `config.json` can still override settings.

```
examples/
├── run.ipynb                # notebook: any photo folder, local or Colab
├── run.py                   # script: example name or any photo folder, every option
├── seneca/                  # OpenDroneMap drone flight, GPS (downloadable)
│   ├── download.py
│   └── images/
├── south_building/          # hand-held photos, no GPS (downloadable)
│   ├── download.py
│   └── images/
├── aerial_gcps/             # Agisoft sample survey: positions table + surveyed GCPs (downloadable)
│   ├── download.py
│   └── images/              # photos + Cameras_WGS84.txt + GCPs_WGS84.txt
├── drone2024/                # private drone flight, GPS + ground control points (not in git)
│   └── images/              # photos + targets.csv (GCPs); any file names, all detected automatically
└── droneNov2026/            # private: several flights, a video, logs with targets, ignore/ and results/ (not in git)
    └── config.json          # {"geo": {"targets_orto": "Vuelo orto programado con dianas"}}
```

```bash
uv run python examples/run.py seneca --preset fast                  # downloads 40 GPS drone photos (~90 MB) on first use
uv run python examples/run.py south_building --preset fast          # downloads 40 photos (~420 MB archive) on first use
uv run python examples/run.py aerial_gcps --preset fast             # downloads 40 survey photos + GCPs (~440 MB) on first use
uv run python examples/run.py drone2024 --preset medium
```

**Script** (`examples/run.py`): give it an example name or any folder of photos. It accepts every option of the
`geosplat` command plus `--config file.json`. Results go to `<dataset>/output_<preset>/` unless you pass `-o`.

```bash
uv run python examples/run.py my_scene --preset fast                    # examples/my_scene/images/
uv run python examples/run.py /data/flight1 --preset fast --max-images 80 --polygon /data/aoi.gpkg
uv run python examples/run.py /data/flight1 --preset quality --crs EPSG:25830 -o /data/results
uv run python examples/run.py --help                                   # all options
```

**Notebook** ([`examples/run.ipynb`](examples/run.ipynb)): set `IMAGES_DIR`, `OUTPUT_DIR` and `PRESET`,
then run all. It installs geosplat from GitHub if missing and checks that every stage runs on the GPU. It detects
whether the photos have GPS:

- **With GPS:** you get a Google Satellite/Hybrid map of the photo positions (pink dots). Where the metadata
  provides it (e.g. DJI gimbal yaw/pitch), the viewing direction is drawn too: a square for nadir photos, an arrow for
  oblique ones. You can draw a polygon to process only part of the flight. The fast and final orthophotos are overlaid on the map.
- **Without GPS:** the same steps run in a local frame, and the orthophotos are shown as plain images.

Either way, it shows the fast orthophoto, the point cloud, an interactive 3D viewer of the splat and the final
orthophoto, then zips the results. `IMAGES_DIR = "seneca"`, `"south_building"` or `"aerial_gcps"` downloads a
public sample. To run it locally: `uv run --group dev jupyter lab examples/run.ipynb`.

## Visualizing the results

### The Gaussian splat: built-in viewer

- `viewer.html` (written next to `splat.ply`): double-click it to open it in a browser. Drag to orbit,
  right-drag to pan, use the wheel to zoom. It needs internet access to load the WebGL viewer library.
- In a notebook: `geosplat.show(result.params, result.train_cameras)`, or `geosplat.show("output/splat.ply")`.

### The Gaussian splat: SuperSplat in the browser (nothing to install)

1. Open <https://superspl.at/editor> in Chrome, Edge or Firefox.
2. Drag and drop `splat.ply` onto the page (or **File ▸ Import**). The file stays local in your browser.
3. Navigate:
   - orbit with left-drag
   - pan with right-drag or shift + left-drag
   - zoom with the wheel
   - fly with WASD
   - press **F** to frame the scene, or double-click a point to re-center
4. geosplat exports scenes centered and upright for 3DGS viewers. If one still appears upside down, select it and
   set rotation X = 180 in the **Transform** panel.
5. Optional cleanup: delete floaters with the sphere, box or brush selection, then **File ▸ Export** a cleaned or
   compressed `.ply`, or a shareable HTML viewer.

Other viewers that read the same file: [antimatter15/splat](https://antimatter15.com/splat/) (web), the
[SIBR viewer](https://github.com/graphdeco-inria/gaussian-splatting#interactive-viewers) (native), and Blender, Unity
or Unreal plugins.

How it looks: the splat is sharpest near the photo positions. Where nothing was photographed (behind objects, above
a nadir flight, far background), expect blur and floaters. For a drone survey, add oblique photos to get good
3D views.

### Point clouds

Open `points_sparse.ply` / `points_dense.ply` in [CloudCompare](https://www.danielgm.net/cc/), MeshLab or Blender.
Georeferenced clouds use CRS coordinates; CloudCompare will offer to shift them on import, so accept.

### Orthophoto and DSM

For GPS imagery two orthophotos are produced:

1. **`fast_orto/fast_orto.tif`**, written seconds after COLMAP finishes. The surface is interpolated from the sparse
   SfM points and there is no occlusion handling, so it is accurate on flat terrain, while buildings and trees lean
   and edges can smear. Use it for a first check of coverage and georeferencing.
2. **`ortho/orthophoto.tif`**, after splat training. It uses the dense surface of the splat and occlusion-aware view
   selection: a true orthophoto, together with `ortho/dsm.tif`.

Open `fast_orto/fast_orto.tif`, `ortho/orthophoto.tif` and `ortho/dsm.tif` in QGIS (drag and drop; the CRS is embedded). For the DSM, use
**Layer Properties ▸ Symbology ▸ Singleband pseudocolor**, or **Raster ▸ Analysis ▸ Hillshade**. Overlay your GCPs
or a reference orthophoto (e.g. PNOA WMS in Spain) to check the georeferencing.

Input folders: give geosplat a folder (or a `.zip`) organised however you like. It is scanned recursively:

- **Photos:** every subfolder with photos is used, e.g. one folder per flight. Each folder gets its own camera
  calibration, and equal file names in different folders don't clash.
- **Videos** (`.mp4`, `.mov`, …): evenly spaced frames are extracted (`--frames-per-video`; the preset sets 15 / 30
  / 50 / 100) and reconstructed with their own camera. Identical copies of a video in several folders are used
  once. A DJI `.SRT` subtitle file next to a video gives its frames GPS positions.
- **Side files** anywhere in the tree (CSV, TXT, GeoPackage, MRK, GPX, SRT, XMP…) are checked for photo positions
  and ground control points. Copies of the same GCP file, or the same targets with and without names, are merged.
- **Skipped folders:** `ignore`, `ignored`, `exclude`, `results`, `output*`, hidden folders and `__MACOSX`, at any
  depth. Add more with `--ignore-folder PATTERN` (or `SfMConfig(ignore_folders=...)`). Results are written to
  `output_<preset>` inside the dataset folder, so they're never read as input.

Photo positions: they are read from every common source, so the photos folder can contain whatever your
workflow produces. Files are searched in the photo folder and its parent; `--positions FILE` forces one.

| Source | Example |
| --- | --- |
| EXIF GPS in each photo | most drones and phones |
| XMP sidecars | `IMG_1.xmp` or `IMG_1.JPG.xmp` (Lightroom, RealityCapture) |
| OpenDroneMap `geo.txt` | CRS on the first line, then `name x y z [yaw pitch roll]` |
| CSV / TXT with photo names | Metashape / Pix4D reference exports, PPK output; lat/lon or X/Y, header optional |
| Point layer with a photo-name field | `.gpkg`, `.geojson`, `.shp`, `.kml` |
| DJI RTK `.MRK` | `*_Timestamp.MRK`, matched by the number at the end of each photo name |
| GPX track | matched by capture time; the camera clock offset is read from EXIF or estimated |

A positions file (geo.txt, CSV, vector, MRK) overrides EXIF, since it usually holds RTK/PPK positions. XMP sidecars
override EXIF too, and GPX only fills photos that have no position. Projected coordinates need a CRS: it's read
from the file (geo.txt, vector), otherwise give `--positions-crs EPSG:25830`; lon/lat is recognised automatically.
With positions, every output GeoTIFF and point cloud is georeferenced. Without them, the outputs are in a local
frame with no CRS.

Ground control points: put a GCP file (any name) next to the photos and it is detected automatically:

- **`.gpkg`, `.geojson`, `.shp`** (or other vector formats): points with a CRS. The name comes from a `name` / `id` /
  `label` field and the height from a 3D geometry or a `z` / `h` / `height` field.
- **CSV / text:** one point per line, `name,x,y,z` (header optional, `,` `;` tab or space separated). A CSV has no
  CRS: set it with `--gcps-crs EPSG:25830` (CLI / `run.py`) or `GCPS_CRS` (notebook), or let geosplat infer it. It
  tries lon/lat and the UTM zones around the photos (WGS84, ETRS89, NAD83) and keeps the one that puts most points
  inside the photographed area.

If you don't set an output CRS (`--crs`), a projected GCP CRS is used for all outputs, so everything matches your
survey. The GCPs are drawn on the previews and compared with the DSM (`gcp_check.json`). Without picks,
georeferencing uses only the photo positions: with consumer drone GPS expect meter-level error horizontally, and
sometimes tens of meters vertically (barometric altitude). The internal geometry is much better than that.

**Locating the targets** corrects that error:

1. A run with GCPs writes `gcp/picker.html`. Open it in a browser, click each target on the orthophoto (pink ✕,
   draggable; 3+ points for a 7-parameter fit, 1–2 for a shift) and press **Save picks.json** into `gcp/`.
2. Run the same command again. The picks correct the georeference of every product; SfM and the trained splat are
   reused, so it takes seconds. `gcp/report.json` has the residuals.

In the notebook the same step is section 7. The picks file is shared, so `run.py` applies notebook picks too.

**`targets_orto`** (`--targets-orto FOLDER`, `geo.targets_orto` in `config.json`, `TARGETS_ORTO` in the notebook):
a subfolder, e.g. a low nadir flight flown for the targets. Its photos build `ortho_targets/orthophoto.tif` at the
full resolution of the original photos, which the picker uses. Only the GCP files inside it are read, and target
files of other campaigns are ignored. `None` (default) means all photos and all GCP files. The fast and final
orthophotos always use every photo. That flight must be part of the same reconstruction (shared features). A flight
from another season or year may not be, and the run then warns that its targets can't correct the other photos.

### Quick looks without a viewer

- `renders/*.jpg`: held-out photo (left) next to its render (right).
- `flythrough.gif` (examples and notebook): a camera path through the training views.

## Tips for good captures

- 30–200 sharp photos with 60–80 % overlap; move around the subject rather than rotating in place.
- Keep exposure fixed and avoid moving objects; reflective and textureless surfaces are hard.
- Drone mapping: 70–80 % front and side overlap. Add a ring of oblique photos for 3D views.
- Out of GPU memory: lower `max_resolution` (e.g. 1000) or set `max_gaussians`.

## Tests

```bash
uv run pytest                 # quick unit tests (seconds)
uv run pytest --run-slow      # + end-to-end tests with the very_fast preset (~2 min, GPU)
```

The slow tests run `examples/run.ipynb` (copies generated in `tests/notebooks/`) on the Seneca sample (and on
`drone2024` if present), and run `examples/run.py` on every example whose photos are present. With `aerial_gcps`
downloaded they also check the picks round trip: saved picks correct the georeference on the next run.

## Package layout

| Module | Purpose |
| --- | --- |
| `sfm.py` | COLMAP pipeline (`run_sfm`, `SfMConfig`), GPU, image selection |
| `selection.py` | Subsampling, polygon filtering |
| `scene.py` | COLMAP loading, scene normalization, image dataset |
| `geo.py` | EXIF GPS, CRS, similarity fit (`GeoReference`, `GeoConfig`), GCP files and CRS inference |
| `sources.py` | Dataset scan (subfolders, videos, side files, ignored folders), frame extraction, catalogue |
| `positions.py` | Photo positions from EXIF, XMP, geo.txt / CSV, vector files, DJI MRK, GPX |
| `gcp.py` | Notebook GCP picker, georeference refit from picks |
| `picker_html.py` | Stand-alone `gcp/picker.html` page |
| `datasets.py` | Sample downloads (`seneca`, `south_building`, `aerial_gcps`), zip handling |
| `trainer.py` | Optimization loop, densification (default / MCMC), metrics |
| `render.py` | gsplat CUDA backend (pinhole/ortho, RGB/depth) + PyTorch fallback |
| `gaussians.py` | Initialization, background sphere, splat PLY export/import |
| `pointcloud.py` | Sparse / dense colored point clouds |
| `ortho.py` | Orthophoto from photos and from the splat, targets orthophoto, DSM, GeoTIFF |
| `video.py` | Camera paths, GIF export |
| `viewer.py` | Interactive WebGL splat viewer (`show`, `save_viewer`) |
| `mapview.py` | Notebook maps: photo positions with polygon selection, orthophoto overlay |
| `presets.py` | Speed presets (`very_fast`, `fast`, `medium`, `quality`), `make_configs` |
| `pipeline.py` | `images_to_splat`, end to end |
| `cli.py` | `geosplat` command |
