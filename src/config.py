"""Central configuration for the Taipei LULC revision analysis.

Edit ONLY this file when your GEE paths differ. Everything else reads from here.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Google Earth Engine
# --------------------------------------------------------------------------
# Your Cloud project. Still required for ee.Initialize() since Nov 2024, even
# when the assets themselves live under a legacy users/ path. Find it at
# https://code.earthengine.google.com -> gear icon -> "Cloud Project".
EE_PROJECT = "ee-amanjmi01"

# GEE username, for legacy asset paths (users/<username>/...).
EE_USER = "amanjmi01"

# Where the assets actually live. This project uses LEGACY paths - the assets
# predate Cloud-project asset folders and appear in the Code Editor Imports
# panel as "users/amanjmi01/...". Legacy paths still resolve fine; there is no
# need to migrate them for the revision.
ASSET_ROOT = f"users/{EE_USER}"

# If you ever move the assets into the Cloud project, switch to:
#     ASSET_ROOT = f"projects/{EE_PROJECT}/assets"
# and nothing else here changes.

ASSETS = {
    # Training polygons - class label in the 'Id' property.
    # NOTE the _01 suffix: the GEE script imports ROI_TAIPEI_01, not ROI_TAIPEI.
    "roi_polygons": f"{ASSET_ROOT}/ROI_TAIPEI_01",
    # City boundary
    "taipei_boundary": f"{ASSET_ROOT}/Taipei_City",
    # Classified rasters exported previously (optional, for map figures).
    # These may not exist yet - only used by the map-figure step.
    "lulc_cart": f"{ASSET_ROOT}/taipei_lulc_CART",
    "lulc_svm": f"{ASSET_ROOT}/taipei_lulc_SVM",
    "lulc_rf": f"{ASSET_ROOT}/taipei_lulc_RF",
}

# --------------------------------------------------------------------------
# Imagery / classification settings (mirror the published manuscript)
# --------------------------------------------------------------------------
# COPERNICUS/S2 (used in the original script) is deprecated -> harmonised L1C.
S2_COLLECTION = "COPERNICUS/S2_HARMONIZED"

# NOTE: these match the ACTUAL GEE script, not the submitted manuscript.
# The manuscript says "summer 2020" and cloud cover "minimum"; the script uses
# the full year and a 20% threshold. The manuscript text needs correcting.
DATE_START = "2020-01-01"
DATE_END = "2020-12-31"
MAX_CLOUD_PCT = 20
SCALE = 10  # metres

# Sentinel-2's native grid over Taipei: WGS84 / UTM zone 51N. Both the sampling
# and the exported composite are pinned to it.
#
# This is not cosmetic. A median() composite is a COMPUTED image, and a computed
# image's default projection in Earth Engine is EPSG:4326 at 1 degree - not the
# projection of its inputs. sampleRegions() without an explicit `projection`
# therefore samples on a lat/lon tessellation, which does not line up with the
# 10 m UTM pixels the data actually lives on. The band values still come back
# correct (nearest-neighbour lookup finds the right native pixel), but the
# tessellation is wrong, so some native pixels get sampled twice and others not
# at all. Measured on the first run: 8.8% of sample rows were repeats.
EXPORT_CRS = "EPSG:32651"

# The script uses four bands. Manuscript Table 1 bolds only three (B3/B4/B8)
# and omits B2 - another text/code mismatch to fix.
BANDS_PUBLISHED = ["B2", "B3", "B4", "B8"]
BANDS_EXTENDED = ["B2", "B3", "B4", "B8", "B11", "B12"]
INDICES = ["NDVI", "NDBI", "MNDWI"]

# Class labels live in the 'Id' property and are numbered 1..5
# (palette in the GEE script runs min:1 max:5).
CLASS_PROPERTY = "Id"
CLASS_NAMES = {
    1: "Water",
    2: "Built-up",
    3: "Forest/Vegetation",
    4: "Plantation",
    5: "Swamp",
}
CLASS_COLORS = {
    1: "#1200ff",
    2: "#ff0000",
    3: "#2a7f00",
    4: "#96b058",
    5: "#04d1ff",
}

# Split at polygon level, not pixel level: pixels from one training polygon are
# near-duplicates, so a pixel-level split leaks information across the divide.
GROUP_COLUMN = "poly_rnd"

# --------------------------------------------------------------------------
# Experiment settings
# --------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.30          # stratified hold-out fraction
CV_FOLDS = 5              # stratified k-fold for hyperparameter search
N_ITER_SEARCH = 60        # RandomizedSearchCV iterations
N_BOOTSTRAP = 2000        # bootstrap replicates for accuracy CIs

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
SAMPLES = DATA / "samples"
EXPORTS = DATA / "exports"
INTERIM = DATA / "interim"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

for _p in (RAW, SAMPLES, EXPORTS, INTERIM, RESULTS, FIGURES, TABLES):
    _p.mkdir(parents=True, exist_ok=True)

# Primary sample table produced by src/gee_export_samples.py
SAMPLE_CSV = SAMPLES / "taipei_samples.csv"
