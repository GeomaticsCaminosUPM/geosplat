# Input files

Give geosplat one folder (or a `.zip` of it). Everything inside is found automatically; file names don't matter.

```
my_survey/
├── flight_1/            photos (each folder gets its own camera calibration)
├── flight_2/
├── video/DJI_0058.MOV   frames are extracted (+ DJI .SRT next to it for GPS)
├── positions.csv        photo positions (optional)
├── targets.csv          ground control points (optional)
├── ignore/              skipped (also: results, output*, exclude, hidden folders)
└── output_fast/         results (written here, never read as input)
```

## Photo positions

The first source found wins, in this order: positions file, then XMP sidecar, then EXIF GPS, then GPX.

| Source | Example |
| --- | --- |
| EXIF GPS | most drones and phones |
| XMP sidecar | `IMG_1.xmp`, `IMG_1.JPG.xmp` |
| OpenDroneMap `geo.txt` | CRS on line 1, then `name x y z [yaw pitch roll]` |
| CSV / TXT with photo names | Metashape / Pix4D exports, PPK output; lat/lon or x/y; header optional (`# file WGS84_lat WGS84_lon WGS84_H ...` works) |
| Point layer | `.gpkg`, `.geojson`, `.shp`, `.kml` with a photo-name field |
| DJI RTK `.MRK` | `*_Timestamp.MRK` |
| GPX track | matched by capture time |

Projected coordinates without a CRS in the file need `--positions-crs` (CLI). Lon/lat is recognised by itself.

## Ground control points (GCPs)

| Format | Example |
| --- | --- |
| Vector with CRS | `.gpkg`, `.geojson`, `.shp`: name from `name`/`id`/`label`, height from 3D geometry or a `z`/`h` field |
| CSV / TXT | `name,x,y,z` or `id,x,y,z[,name]`, header optional, `,` `;` tab or space |
| CSV / TXT with a header | columns by name, any order: `#Name Lat Lon Ell.H(m)`, `id;este;norte;cota` |

```
name,x,y,z
D1,411339.128,4496352.53,839.571
D2,411345.226,4496318.78,837.526
```

CSV files have no CRS: set `--gcps-crs` / `GCPS_CRS`, or let geosplat infer it from the photo positions. Several
GCP files are merged, and duplicates (same point, with or without a name) are removed. With `targets_orto`, only the
files inside that folder are used.

## `located_targets.json` (targets located on the orthophoto)

Written by `targets/locate_targets.html` (button **Save located_targets.json**) or by the notebook. Default place: `<output>/targets/located_targets.json`;
`--located-targets FILE` uses another one.

```json
{
 "picks": {"4": [59.844529, 31.477427, 64.71], "5": [59.844451, 31.480372, 67.30]},
 "geo": {"crs": "EPSG:32636", "origin": [...], "scene_to_geo": [...]},
 "scene_transform": [...]
}
```

Each pick is latitude, longitude and the DSM height at that spot. `geo` is the georeference of the orthophoto the
picks were made on, so the file stays valid after products are regenerated. If the SfM reconstruction changes (other
photos or options), the picks no longer apply and the run says so. Pick again on the new page.
