# Tutorial: `examples/run.py`

`run.py` takes a photo folder (or a sample name) and writes every product. These steps use public sample data,
so you don't need photos of your own. Times are for an RTX 4060 Laptop GPU; a Colab T4 is about 2× slower.

## 0. Install

```bash
git clone https://github.com/GeomaticsCaminosUPM/geosplat.git && cd geosplat
uv sync
```

The first run compiles gsplat's CUDA code (2–5 min, once).

## 1. A drone flight with GPS (`seneca`)

```bash
uv run python examples/run.py seneca --preset very_fast
```

- Downloads 40 GPS-tagged photos into `examples/seneca/images/` (first time only).
- Writes everything to `examples/seneca/output_very_fast/`.
- Positions and the UTM zone come from the photos' EXIF. Nothing to configure.

Open the results:

| File | Open with |
| --- | --- |
| `fast_orto/fast_orto.tif`, `ortho/orthophoto.tif`, `ortho/dsm.tif` | QGIS (drag and drop) |
| `points_dense.ply` | CloudCompare |
| `viewer.html` | any browser |
| `splat.ply` | <https://superspl.at/editor> |

Then try a better preset: `--preset fast` (previews) or `--preset medium` (good results).

## 2. Photos without GPS (`south_building`)

```bash
uv run python examples/run.py south_building --preset very_fast
```

The same products are made in a local frame (no CRS; units are SfM units). No maps, no ground control points.

## 3. Ground control points: locate the targets, then run again (`aerial_gcps`)

```bash
uv run python examples/run.py aerial_gcps --preset very_fast
```

The download (~440 MB, a few minutes) puts in `examples/aerial_gcps/images/`:

```
images/
├── 2019_08_06_PUTI_SonyRX1_..._002.JPG ...   40 photos, no GPS in EXIF
├── Cameras_WGS84.txt                         photo positions (name, lat, lon, height, angles)
└── GCPs_WGS84.txt                            surveyed targets (name, lat, lon, ellipsoidal height)
```

Both text files are found and recognised automatically. Any file names work.

The run ends with:

```
Targets: open .../output_very_fast/gcp/picker.html, save .../gcp/picks.json (not found yet) and run this command again.
```

1. Open `gcp/picker.html` in a browser (double-click it). The orthophoto sits on Google Satellite. Red circles mark
   where the surveyed coordinates fall with the current (GPS) georeference.
2. Choose a point in the list (it starts with the first one) and click the centre of its target on the orthophoto.
   A pink ✕ appears and the next point is selected. Drag a ✕ to adjust it; ✖ in the list deletes it. Zoom in for
   precision. Place at least 3 points for a full fit (1–2 only shift the georeference).
3. Press **Save picks.json**. Save it in the `gcp/` folder next to the page; in Firefox and Safari it goes to
   Downloads, so move it there.
4. Run the same command again:

```bash
uv run python examples/run.py aerial_gcps --preset very_fast
```

```
Georeference corrected with 3 picked targets (similarity (7 parameters)): RMSE xy ... m, z ... m
Reusing the trained splat .../splat.ply (same scene and training options)
```

SfM and training are reused, so the second run takes seconds. Every product (orthophotos, DSM, point clouds,
`geo.json`) now uses the GCPs. `gcp/report.json` lists the residuals; `ortho/gcp_check.json` gives DSM height
minus GCP height. The page now shows your picks in the corrected frame. Adjust them, save, and run again if needed.

## 4. Several flights: a separate flight for the targets (`--targets-orto`)

A dataset folder can hold several flights (one subfolder each), videos and target files. If one flight was flown
to see the targets (e.g. low and nadir), use it for the target orthophoto:

```bash
uv run python examples/run.py /data/survey --preset fast --targets-orto "nadir flight"
```

- `ortho_targets/orthophoto.tif`: orthophoto from that folder's photos only, at the **full resolution of the
  original photos**. `gcp/picker.html` uses it.
- Only the GCP files inside that folder are used; target files elsewhere (other campaigns) are ignored.
- `fast_orto/` and `ortho/` still use all photos.

That flight must share features with the other photos, so it ends up in the same reconstruction. If it was flown in
another season or year, it may not. The run then says so (`No photo of ... is in the main reconstruction`), and
targets visible only there can't correct the other photos.

The same option can go in `<dataset>/config.json`, which `run.py` reads automatically:

```json
{"preset": "fast", "geo": {"targets_orto": "nadir flight", "gcps_crs": "EPSG:25830"}}
```

## 5. Your own photos

```bash
uv run python examples/run.py /data/flight1 --preset fast           # folder (or flight1.zip)
uv run python examples/run.py /data/flight1 --preset fast -o /data/results --crs EPSG:25830
uv run python examples/run.py --help                                # every option
```

See [Input files](input_files.md) for what can be in the folder.
