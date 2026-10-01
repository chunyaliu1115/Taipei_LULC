# Data manifest

Every dataset in this repository, with its provenance, its shape and what
depends on it. Unless noted otherwise these files are CC BY 4.0 (see
`LICENSE-DATA`) and derive from Sentinel-2 Level-1C top-of-atmosphere
reflectance (`COPERNICUS/S2_HARMONIZED`), © Copernicus, free and open under
the Copernicus data policy. Level-1C rather than Level-2A: see the note under
`data/exports/` below.

The class scheme has five classes throughout: Water (1), Built-up (2),
Forest/Vegetation (3), Plantation (4), Swamp (5). The `Id` column carries the
integer code and `Class` the name.

## `data/raw/` — vector inputs

| File | What it is |
|---|---|
| `Taipei_City.gpkg` / `.geojson` | Administrative boundary of Taipei City, the study area. Defines the clip extent for the composite and the denominator for every area figure. |
| `ROI_TAIPEI.gpkg` / `.geojson` | The 110 hand-digitised training polygons, one feature per class as MultiPolygon, class code in `Id`. This is the reference data the whole paper rests on. Digitised on Sentinel-2 true-colour with Google Earth imagery for context. |
| `single_parts.gpkg` | `ROI_TAIPEI` after every MultiPolygon is exploded into its constituent parts, with a stable `poly_id` and the assigned `split`. Produced by `src/assign_split.py`; committed so the exact split used in the paper is recoverable rather than merely reproducible. |

The polygons are purposive, not a probability sample. Accuracy figures
computed from them are conditional on the polygons being spectrally
representative of their classes; this is stated as a limitation in the paper
and is the reason `reference_points_blank.csv` exists.

## `data/samples/` — labelled pixels

| File | Rows | What it is |
|---|---|---|
| `taipei_samples.csv` | 8,493 | Band values at every 10 m pixel falling inside a training polygon, sampled from the 2020 annual median composite. Columns: `B2 B3 B4 B8` (top-of-atmosphere reflectance, scaled to 0–1), `Class`, `Id`, `poly_id`, `lon`, `lat`, `split`. **This is the file that makes the analysis reproducible without an Earth Engine account.** |
| `reference_points_blank.csv` | 605 | The independent probability sample designed but not yet interpreted. Stratified on the SVM map, rare classes floored at 75 points. `reference_class`, `interpreter` and `confidence` are empty by design and are to be filled by visual interpretation. |
| `reference_points.geojson` | 605 | The same points as geometry, for loading into QGIS or Earth Engine to interpret. |

Split composition in `taipei_samples.csv`, after the polygon-level 70/30
split stratified by class (80 training polygons, 30 validation polygons, no
polygon on both sides):

| Class | Train | Validation | Total |
|---|---|---|---|
| Water | 1,551 | 153 | 1,704 |
| Built-up | 1,577 | 440 | 2,017 |
| Forest/Vegetation | 2,383 | 970 | 3,353 |
| Plantation | 775 | 150 | 925 |
| Swamp | 378 | 116 | 494 |
| **Total** | **6,664** | **1,829** | **8,493** |

## `data/interim/`

| File | Rows | What it is |
|---|---|---|
| `test_predictions.csv` | 1,829 | One row per validation pixel: `y_true`, `poly_id`, and the predicted class from each of the six models. Everything in the significance testing — the naive McNemar test, the cluster-corrected form of Durkalski et al. (2003), the design effect and the effective sample size — is computed from this one file. `poly_id` is what makes the clustering visible; without it the correction cannot be applied. |

## `data/exports/` — classified rasters

Six GeoTIFFs, one per model, ~300 KB each: `taipei_lulc_CART.tif`,
`taipei_lulc_Random_Forest.tif`, `taipei_lulc_XGBoost.tif`,
`taipei_lulc_LightGBM.tif`, `taipei_lulc_SVM_RBF.tif`,
`taipei_lulc_MLP_deep.tif`.

Single band, unsigned integer, class codes 1–5 as above, 10 m, clipped to the
Taipei City boundary. Produced by `src/classify_map.py` applying each fitted
model to the composite. These are the source of every mapped-area figure and
of the model-agreement figure.

### Not included: `s2_composite_2020.tif`

The Sentinel-2 annual median composite (four bands, 10 m, 20 MB) is excluded
from the Git repository to keep clone size down. It is fully deterministic —
regenerate it by running `gee_repo/taipei_lulc_reference.js` in the Earth
Engine Code Editor, or download it from the archived release. Recipe:
`COPERNICUS/S2_HARMONIZED`, calendar year 2020, scenes filtered at
`CLOUDY_PIXEL_PERCENTAGE < 20`, per-pixel cloud and cirrus mask from QA60,
reflectance divided by 10,000, per-band median, clipped to the city boundary,
reprojected to UTM 51N at 10 m, bands B2/B3/B4/B8 only.

### Level-1C, not Level-2A

`COPERNICUS/S2_HARMONIZED` is top-of-atmosphere reflectance. It is not
atmospherically corrected; `COPERNICUS/S2_SR_HARMONIZED` is the surface
reflectance collection and is *not* what was used. This was inherited from the
originally submitted analysis and retained so that the six classifiers are
compared on exactly the imagery the earlier result was built on. For a
single-city, single-year comparison the atmosphere shifts all six models
together rather than favouring one, so the comparison stands; the absolute
accuracies should be read as conditional on the product level, and anyone
extending this to multi-temporal work should move to Level-2A.

The four-band feature set is likewise deliberately minimal, to keep continuity
with the originally submitted analysis and to isolate the effect of the
classifier from the effect of the feature space. Neither choice is a
recommendation.

## `results/`

`results/tables/` holds every number that appears in the paper as CSV or JSON,
regenerated from the analysis outputs by `src/manuscript/make_tables.py`.
`src/manuscript/check_numbers.py` verifies the prose against these files, so
the text and the archive cannot diverge.

`results/figures/` holds every figure as both PNG and PDF, regenerated by
`src/figures/make_figures.py`. `results/figures/FIGURE_CAPTIONS.md` maps each
file to its figure number in the paper.
