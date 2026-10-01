# Bringing the GEE project down to your Mac

Run every command in **Terminal on your own machine**, from the project root:

```bash
cd ~/Taipei_LULC/Taipei_LULC
```

There are three separate things to pull, and they use different mechanisms. GEE
scripts live in a Google-hosted Git repository. GEE *assets* (your training
points, boundary, classified rasters) live in a separate asset store and are not
part of that Git repo — they come down via the CLI or an export task.

---

## 1. Environment

Conda is the least painful route because `gdal`/`rasterio` wheels are a
recurring source of pain on Apple Silicon.

```bash
conda env create -f env/environment.yml
conda activate taipei-lulc
```

If you would rather use pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r env/requirements.txt
```

Verify:

```bash
python -c "import ee, geemap, xgboost, lightgbm; print('ok')"
```

---

## 2. Authenticate and confirm your Cloud project

Since November 2024 every GEE call must be attached to a Cloud project. Find
yours in the Code Editor: gear icon (top right) → the project name shown under
"Cloud Project".

```bash
earthengine authenticate            # opens a browser, one time only
earthengine set_project <YOUR_CLOUD_PROJECT>
python -c "import ee; ee.Initialize(project='<YOUR_CLOUD_PROJECT>'); print(ee.Number(1).add(1).getInfo())"
```

A printed `2` means you're connected. Then put that same project name into
`src/config.py` (`EE_PROJECT`). It is currently set to a guess,
`ee-amanjmi01` — replace it if the gear icon shows something else.

Note that the Cloud project is separate from where your assets live. This
project's assets sit under legacy `users/amanjmi01/` paths, which is fine and
needs no migration; the Cloud project is only what `ee.Initialize` bills calls
against.

---

## 3. Clone the script repository

Your Code Editor scripts sit in a Git repo at
`https://earthengine.googlesource.com/users/<GEE_USERNAME>/<REPO_NAME>`.
The repo name is the top-level folder in the Code Editor's **Scripts** tab
(left panel) — the one your `.js` files sit under.

First generate credentials: visit
<https://earthengine.googlesource.com/new-password>, sign in with the same
Google account, and run the snippet it gives you. It writes a `.gitcookies`
entry so `git` can talk to the host.

```bash
git clone https://earthengine.googlesource.com/users/<GEE_USERNAME>/<REPO_NAME> gee_repo/<REPO_NAME>
```

To list what repos exist if you're unsure of the name:

```bash
earthengine ls users/<GEE_USERNAME>
```

The clone is an ordinary Git repo. You can push back to it, and — importantly
for the editor's reproducibility request — you can add a GitHub remote and
mirror it publicly:

```bash
cd gee_repo/<REPO_NAME>
git remote add github git@github.com:<you>/taipei-lulc-gee.git
git push github main
```

That public GitHub URL is what goes into the revised Data Availability
Statement.

---

## 4. Find and download your assets

List everything in your asset tree:

```bash
earthengine ls users/amanjmi01                  # legacy tree - where your assets are
earthengine ls projects/<YOUR_CLOUD_PROJECT>/assets   # Cloud tree, probably empty
```

Check what a given asset actually is before you pull it:

```bash
earthengine asset info users/amanjmi01/ROI_TAIPEI_01
earthengine asset info users/amanjmi01/Taipei_City
```

`src/config.py` is already filled in with these two, taken from the Imports
panel of `Taipei_City_2020_LULC_S2_RF`. `check_env.py` verifies them against
the server once you have authenticated, so you'll be told if either is wrong.

### Vector assets (training points, boundary)

Small tables come down directly through the Python API — that's what
`src/gee_export_samples.py` does. Once `config.py` is filled in:

```bash
python src/gee_export_samples.py
```

This rebuilds the summer-2020 composite, samples band values at every point,
and writes `data/samples/taipei_samples.csv` with a per-class, per-split
breakdown printed to the console. That printout is the raw material for the
training-data table the editor asked for.

If a table is too big to pull client-side (you'll see a "payload size exceeds"
error), route it through Drive instead:

```bash
python src/gee_export_samples.py --to-drive
```

then watch the **Tasks** tab in the Code Editor and drop the resulting CSV into
`data/samples/`.

### Raster assets (classified maps)

`ee.Image` assets cannot be downloaded with the CLI. Two options:

```bash
# Option A - geemap, straight to GeoTIFF (fine for a 272 sq.km city at 10 m)
python - <<'PY'
import ee, geemap, sys
sys.path.insert(0, "src")
import config as C
ee.Initialize(project=C.EE_PROJECT)
aoi = ee.FeatureCollection(C.ASSETS["taipei_boundary"]).geometry()
for key in ("lulc_cart", "lulc_svm", "lulc_rf"):
    img = ee.Image(C.ASSETS[key])
    geemap.ee_export_image(
        img, filename=str(C.EXPORTS / f"{key}.tif"),
        scale=C.SCALE, region=aoi, file_per_band=False)
PY
```

Option B, for anything that hits the 32 MB direct-download ceiling: run
`Export.image.toDrive` from the Code Editor (the reference script in
`gee_repo/taipei_lulc_reference.js` already contains these export blocks), then
sync the folder down with `rclone` or the Drive desktop app into
`data/exports/`.

---

## 5. Confirm the import worked

```bash
python -c "
import pandas as pd, sys; sys.path.insert(0,'src'); import config as C
df = pd.read_csv(C.SAMPLE_CSV)
print(df.shape); print(df.groupby(['split','class']).size())
"
```

You should see the five classes with sensible counts on both sides of the
split. If `class` is missing, your point asset uses a different property name —
change `CLASS_PROPERTY` in `src/config.py`.

---

## 6. Run the revision analyses

Run in this order. Each step reads a file the previous one wrote, so skipping
one leaves stale numbers behind rather than raising an error.

```bash
python src/gee_export_samples.py            # 1. re-sample on the native grid   ~5 min, needs GEE
python src/train_compare.py                 # 2. tuning + 6 models + bootstrap CIs   ~25 min
python src/mcnemar_test.py                  # 3. pairwise significance, cluster-corrected
python src/error_analysis.py                # 4. separability, error concentration
python src/classify_map.py --skip-download  # 5. all six maps + per-model areas   ~10 min
python src/olofsson.py                      # 6. area-weighted accuracy + area CIs
python src/draw_reference_sample.py         # 7. redraw the blind reference sample
```

Only step 1 talks to Earth Engine. `--skip-download` on step 5 reuses
`data/exports/s2_composite_2020.tif`; drop it only if that file is missing or
the date range in `config.py` changed.

### What to check at each step

**Step 1** prints a grid check. It must read close to `100.0% of samples on the
EPSG:32651 10 m grid; 0 duplicate pixel(s)`. Anything lower means
`sampleRegions` fell back to the composite's default lat/lon projection and the
`projection=` argument in `sample_polygons()` has gone missing again. Row counts
drop about 9% against the pre-fix run — that is the duplicate pixels being
removed, not data loss.

**Step 2** is the slow one, almost all of it LightGBM's search. Use `--quick`
first to confirm the plumbing before committing to a full run.

**Step 3** should still report 0 of 15 comparisons significant under clustering
against 9 of 15 naive. If that gap closes, something upstream changed.

**Step 5** prints the total mapped area. It must stay near 270.4 sq.km. A jump
to ~597 means the nodata handling in `classify_raster` has regressed and the
whole bounding box is being classified instead of the city.

**Step 7** overwrites `data/samples/reference_points_blank.csv`. Do not run it
after you have started interpreting points — it will discard the work. A
stratified estimate is only valid for the map its sample was drawn from, so the
sample has to be redrawn whenever the maps change, and interpreted only once the
maps are final.

### Why the order

`olofsson.py` needs `results/tables/map_areas.json`, which only `classify_map.py`
produces, because the area weights have to come from the classified map rather
than from the sample. It now reads a per-model block from that file, so every
model is weighted by its own map; running it against a `map_areas.json` that
predates the six-model run will fail with a clear message rather than silently
reusing one model's weights for all of them. `draw_reference_sample.py` needs the
classified raster for the same reason — it draws points from the map strata.

The total area printed by `classify_map.py` settles the 272 vs 292.8 sq.km
question: it came out at 270.44 sq.km, so the Study Area figure is the one to
keep and the Results-table figure needs correcting.

---

## 7. The reference-sample loop

`draw_reference_sample.py` writes `data/samples/reference_points_blank.csv`
with an empty `reference_class` column and the rows shuffled so the map class
is not visible in row order. Interpret each point without looking at
`map_class` — otherwise the reference labels inherit the map's errors and the
whole assessment is circular. Then:

```bash
python src/olofsson.py --reference data/samples/reference_points_blank.csv
```

That run is the one whose numbers go in the paper. Everything before it is the
best that can be squeezed out of the polygon-derived sample, which is a
clustered, purposive sample rather than a probability sample — see the header
of `src/olofsson.py` for what that does and does not invalidate.

---

## Common snags

**`git clone` returns 403.** The `.gitcookies` step wasn't completed, or you're
signed into a different Google account in the browser than the one that owns
the repo.

**`ee.Initialize` complains about no project, or rejects the one given.** You
skipped `earthengine set_project`, or `EE_PROJECT` in `config.py` doesn't match
the gear icon. The value there (`ee-amanjmi01`) is an educated guess.

**"Asset not found" on a `users/...` path.** Legacy assets are owned by the
Google account, not the Cloud project, so make sure `earthengine authenticate`
ran under the account that owns `users/amanjmi01`.

**`sampleRegions` returns zero features.** Almost always a CRS or geometry
mismatch — check that the point asset actually falls inside the boundary
geometry, and that the composite isn't fully masked (print
`composite.bandNames()` and add it to the map).

**"User memory limit exceeded".** Raise `tileScale` in `sampleRegions` (4 → 8 →
16) before assuming the data is too big.

**"Collection query aborted after accumulating over 5000 elements".** A hard
ceiling on `getInfo()`, unrelated to memory or quota — it applies to any
collection pulled client-side in one request. `gee_export_samples.py` works
around it by filtering the polygons into chunks and sampling each chunk
separately, then reassembling locally; oversized chunks are halved and retried.
Use `--chunk-size 5` if the default still struggles, or `--to-drive` to sidestep
the client-side path entirely.
