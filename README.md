# Taipei LULC — six-classifier comparison with cluster-aware validation

Code, data and figures for:

> Liu, C.-y. **How Much Does the Classifier Matter? A Cluster-Aware
> Comparison of Six Machine Learning Algorithms for Land Use Land Cover
> Mapping of Taipei, Taiwan, Using Google Earth Engine and Python.**
> Submitted to the *ICCK Journal of Image Analysis and Processing*
> (manuscript JIAP-2026-490298).

Six classifiers — CART, Random Forest, XGBoost, LightGBM, an RBF support
vector machine and a multilayer perceptron — were tuned under an identical
budget on the same four Sentinel-2 bands and compared on a 2020 annual median
composite of Taipei City. Hold-out overall accuracy spans 2.96 percentage
points, from 0.9267 to 0.9563. **None of the fifteen pairwise differences
survives a significance test that accounts for the polygon structure of the
validation sample.** The repository contains everything needed to reproduce
that result.

## What is here

```
gee_repo/          Earth Engine scripts: compositing, sampling, asset export
src/               analysis pipeline (see Reproducing below)
src/figures/       every figure in the paper, one script per figure
src/manuscript/    table generation and a numeric consistency check
data/raw/          study-area and training polygons as GeoPackage + GeoJSON
data/samples/      band values at all 8,493 labelled pixels, with split labels
data/interim/      per-model test predictions, feeds the significance tests
data/exports/      the six classified GeoTIFFs
results/tables/    every number that appears in the paper, as CSV/JSON
results/figures/   every figure, as PNG and PDF
docs/              Earth Engine import guide and publication notes
env/               conda environment and pip requirements
```

`data/MANIFEST.md` describes each dataset, its provenance and its licence.

## Reproducing

Python 3.11 or newer is required; the `earthengine` command-line tool does not
import on 3.9.

```bash
conda env create -f env/environment.yml && conda activate taipei-lulc
# or: python3.11 -m venv .venv && source .venv/bin/activate
#     pip install -U pip && pip install -r env/requirements.txt

python src/check_env.py               # verifies the stack and EE credentials
```

The pipeline splits into a part that needs Earth Engine and a part that does
not. **Every accuracy, significance and separability result in the paper is
reproducible offline** from the sampled band values committed here, with no
Earth Engine account.

```bash
# --- offline: needs only what is committed in this repository ---
python src/assign_split.py            # polygon-level 70/30 split, stratified by class
python src/train_compare.py           # tuning, six models, bootstrap intervals
python src/mcnemar_test.py            # naive and cluster-corrected pairwise tests
python src/simulate_clustered_null.py # false-positive rates of both tests under a null
python src/olofsson.py                # area-weighted accuracy on the effective sample
python src/error_analysis.py          # error concentration and class separability
python src/draw_reference_sample.py   # design for the independent probability sample

python src/figures/make_figures.py all # all figures -> results/figures/
python src/manuscript/make_tables.py   # all tables  -> results/tables/
python src/manuscript/check_numbers.py # verifies the paper's prose against the tables
```

Two steps need Earth Engine, and their outputs are committed so you do not
have to run them:

```bash
python src/gee_export_samples.py      # composite -> data/samples/taipei_samples.csv
python src/classify_map.py            # composite -> the six classified GeoTIFFs
python src/download_assets.py         # pulls assets down from Earth Engine
```

`classify_map.py --skip-download` will work offline if you already have
`data/exports/s2_composite_2020.tif`, which is not committed here (see below).
Without it, the committed rasters in `data/exports/` are the starting point,
and every mapped-area figure regenerates from them.

`src/config.py` is the only file you should need to edit; it holds the Earth
Engine project, the asset paths and the class scheme.

### The split is at polygon level, not pixel level

This is the methodological point of the paper and it is worth stating in the
README. The 8,493 labelled pixels come from 110 hand-digitised polygons.
Pixels inside one polygon are near-duplicates of one another, so a pixel-level
train/test split places copies of the same observation on both sides of the
divide and inflates the hold-out score. `assign_split.py` explodes every
MultiPolygon into its parts and splits whole polygons: 80 for training (6,664
pixels) and 30 for validation (1,829 pixels), with no polygon on both sides.

The same clustering deflates the validation sample. The Kish design effect
runs from 25 to 42 across classes and models, so 1,829 validation pixels carry
the information of between 43 and 73 independent observations. Every interval
and every hypothesis test in `src/` is computed on that effective count.

## Data not in this repository

The Sentinel-2 annual median composite (`s2_composite_2020.tif`, 20 MB) is
excluded to keep clone size reasonable. It is deterministic: regenerate it with
`gee_repo/taipei_lulc_reference.js`, or take it from the archived release.
Sentinel-2 imagery itself is available from Copernicus and from the Earth
Engine catalogue as `COPERNICUS/S2_HARMONIZED`.

Note that this collection is **Level-1C, top-of-atmosphere**, not Level-2A
surface reflectance. That was inherited from the originally submitted analysis
and kept so the six classifiers are compared on the same imagery the earlier
result used. See the Limitations section of the paper.

## Licence

Code (`src/`, `gee_repo/`) is MIT — see `LICENSE`.
Data, tables and figures (`data/`, `results/`) are CC BY 4.0 — see
`LICENSE-DATA`. Sentinel-2 imagery is © Copernicus, free and open under the
Copernicus data policy.

## Citing

See `CITATION.cff`, or cite the paper above. The archived release carries its
own DOI.
