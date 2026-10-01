"""Step 1 of the revision pipeline: get the training data out of GEE.

Rebuilds the Sentinel-2 composite, samples band values at every pixel inside
the ROI_TAIPEI training polygons, assigns a POLYGON-level train/val split, and
writes a flat CSV.

The polygon-level split matters. Pixels drawn from the same polygon are
near-duplicates; splitting them at random puts near-identical rows on both
sides of the divide and inflates accuracy. Splitting whole polygons keeps the
two sides independent.

Everything downstream runs on this CSV, so GEE is needed once.

The download is paged a few polygons at a time. `getInfo()` aborts on any
collection over 5000 elements and the full sample is around 9,400 pixels, so a
single request cannot return it.

Usage:
    python src/gee_export_samples.py                # 4 bands, as per the script
    python src/gee_export_samples.py --extended     # 6 bands + NDVI/NDBI/MNDWI
    python src/gee_export_samples.py --chunk-size 5 # smaller pages if it stalls
    python src/gee_export_samples.py --to-drive     # route via Drive if large
"""

import argparse
import sys

import ee
import pandas as pd

import config as C


def init_ee():
    try:
        ee.Initialize(project=C.EE_PROJECT)
    except Exception:
        print("Authenticating with Earth Engine (a browser window will open)...")
        ee.Authenticate()
        ee.Initialize(project=C.EE_PROJECT)


def mask_s2_clouds(img):
    qa = img.select("QA60")
    mask = (qa.bitwiseAnd(1 << 10).eq(0)
            .And(qa.bitwiseAnd(1 << 11).eq(0)))
    return img.updateMask(mask).divide(10000).copyProperties(
        img, ["system:time_start"])


def build_composite(extended=False):
    aoi = ee.FeatureCollection(C.ASSETS["taipei_boundary"]).geometry()

    col = (ee.ImageCollection(C.S2_COLLECTION)
           .filterDate(C.DATE_START, C.DATE_END)
           .filterBounds(aoi)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", C.MAX_CLOUD_PCT))
           .map(mask_s2_clouds))

    composite = col.median().clip(aoi)
    bands = C.BANDS_EXTENDED if extended else C.BANDS_PUBLISHED
    img = composite.select(bands)

    if extended:
        ndvi = composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
        ndbi = composite.normalizedDifference(["B11", "B8"]).rename("NDBI")
        mndwi = composite.normalizedDifference(["B3", "B11"]).rename("MNDWI")
        img = img.addBands([ndvi, ndbi, mndwi])

    return img, aoi


def explode_parts(fc):
    """Split MultiPolygon features into their constituent single polygons.

    ROI_TAIPEI_01 stores ONE MultiPolygon per class - five features covering
    110 separately drawn areas. Calling randomColumn() on that collection
    assigns one random number per CLASS, so any threshold split sends whole
    classes to one side of the divide. Exploding first gives 110 groups to
    split over, which is what "polygon-level split" was meant to mean.

    Single-part features pass through unchanged, so this is safe on any ROI
    asset.
    """
    def split_one(feat):
        feat = ee.Feature(feat)
        geom = feat.geometry()
        # MultiPolygon coordinates are a list of polygon ring-lists.
        # geodesic=False matches the planar interpretation of an uploaded asset.
        parts = ee.List(
            ee.Algorithms.If(
                ee.String(geom.type()).compareTo("MultiPolygon").eq(0),
                geom.coordinates().map(
                    lambda c: ee.Geometry.Polygon(ee.List(c), None, False)),
                ee.List([geom]),
            )
        )
        props = feat.toDictionary([C.CLASS_PROPERTY, "Class"])
        return ee.FeatureCollection(
            parts.map(lambda g: ee.Feature(ee.Geometry(g), props)))

    return ee.FeatureCollection(fc.map(split_one)).flatten()


def build_parts():
    """Exploded ROI polygons carrying a stable sequential `poly_id`."""
    polys = explode_parts(ee.FeatureCollection(C.ASSETS["roi_polygons"]))
    part_list = polys.toList(polys.size())
    return ee.FeatureCollection(
        part_list.map(lambda f: ee.Feature(f).set(
            "poly_id", part_list.indexOf(f)))
    )


def sample_polygons(img, parts=None):
    """Sample every pixel in the ROI polygons, tagged with its part id.

    The train/val split is NOT assigned here. A single global random threshold
    over 110 parts leaves a ~4% chance that all nine swamp parts land in train,
    which would make swamp unmeasurable. assign_split() does it per class
    instead, once the data is local and the counts are visible.
    """
    if parts is None:
        parts = build_parts()

    # `projection` is not optional here. Without it the sample is taken on the
    # composite's default projection, which for a computed image is EPSG:4326 -
    # a lat/lon tessellation that does not align with Sentinel-2's 10 m UTM
    # grid. See config.EXPORT_CRS for what that costs.
    return img.sampleRegions(
        collection=parts,
        properties=[C.CLASS_PROPERTY, "Class", "poly_id"],
        scale=C.SCALE,
        projection=ee.Projection(C.EXPORT_CRS).atScale(C.SCALE),
        tileScale=4,
        geometries=True,
    )


def assign_split(df, train_frac=0.7, seed=C.RANDOM_SEED):
    """Assign train/val at PART level, stratified by class.

    Every class is guaranteed parts on both sides. Within a class the parts are
    shuffled and the first ceil(train_frac * n) go to train, so the allocation
    is deterministic given the seed rather than merely probable.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    df = df.copy()
    df["split"] = "val"

    for cls, grp in df.groupby(C.CLASS_PROPERTY):
        parts = np.sort(grp["poly_id"].unique())
        if len(parts) < 2:
            print(f"  WARNING: class {cls} has only {len(parts)} part(s); it "
                  f"cannot be split and is going entirely to train")
            df.loc[df[C.CLASS_PROPERTY] == cls, "split"] = "train"
            continue
        shuffled = rng.permutation(parts)
        n_train = int(np.ceil(train_frac * len(parts)))
        n_train = min(max(n_train, 1), len(parts) - 1)   # never empty a side
        df.loc[df["poly_id"].isin(shuffled[:n_train]), "split"] = "train"

    return df


def fc_to_rows(fc):
    """Flatten one FeatureCollection response into a list of dicts."""
    rows = []
    for feat in fc.getInfo()["features"]:
        props = dict(feat["properties"])
        geom = feat.get("geometry")
        if geom and geom.get("type") == "Point":
            props["lon"], props["lat"] = geom["coordinates"]
        rows.append(props)
    return rows


def _is_size_limit(exc):
    msg = str(exc).lower()
    return "5000 elements" in msg or "aborted after accumulating" in msg


def fetch_in_chunks(img, parts, n_parts, chunk_size=10):
    """Download the samples a few polygons at a time and reassemble locally.

    `getInfo()` refuses any collection over 5000 elements, and the full sample
    is roughly 9,400 pixels. Rather than thin the data, this filters the
    POLYGONS before sampling, so each request only ever computes the pixels it
    is about to return. Chunks that still come back too large are halved and
    retried, down to a single polygon.

    Filtering the polygons rather than the sampled pixels matters: it keeps the
    server-side work proportional to the chunk instead of re-sampling the whole
    ROI on every request.
    """
    rows, pending = [], []
    start = 0
    while start < n_parts:
        pending.append(list(range(start, min(start + chunk_size, n_parts))))
        start += chunk_size

    done = 0
    while pending:
        ids = pending.pop(0)
        subset = parts.filter(ee.Filter.inList("poly_id", ids))
        try:
            got = fc_to_rows(sample_polygons(img, parts=subset))
        except ee.ee_exception.EEException as exc:
            if _is_size_limit(exc) and len(ids) > 1:
                mid = len(ids) // 2
                pending[:0] = [ids[:mid], ids[mid:]]
                continue
            if _is_size_limit(exc):
                sys.exit(
                    f"Polygon {ids[0]} alone exceeds the 5000-element limit "
                    f"({exc}).\nUse --to-drive for this dataset."
                )
            raise

        rows.extend(got)
        done += len(ids)
        print(f"  polygons {done}/{n_parts} - {len(rows)} pixels so far",
              end="\r", flush=True)

    print(f"  polygons {done}/{n_parts} - {len(rows)} pixels total" + " " * 12)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extended", action="store_true",
                    help="6 bands + NDVI/NDBI/MNDWI instead of B2/B3/B4/B8")
    ap.add_argument("--to-drive", action="store_true",
                    help="export via Google Drive instead of pulling directly")
    ap.add_argument("--train-frac", type=float, default=1 - C.TEST_SIZE)
    ap.add_argument("--chunk-size", type=int, default=10,
                    help="polygons per request when paging around the "
                         "5000-element getInfo limit (default 10)")
    args = ap.parse_args()

    init_ee()
    img, aoi = build_composite(extended=args.extended)
    print(f"Composite bands: {img.bandNames().getInfo()}")

    raw_polys = ee.FeatureCollection(C.ASSETS["roi_polygons"])
    parts = build_parts()
    n_feat, n_part = raw_polys.size().getInfo(), parts.size().getInfo()
    print(f"ROI: {n_feat} features -> {n_part} single-part polygons")
    if n_part == n_feat and n_feat < 20:
        print("  WARNING: only %d groups to split over. The val set will be "
              "very coarse." % n_feat)

    if args.to_drive:
        task = ee.batch.Export.table.toDrive(
            collection=sample_polygons(img, parts=parts),
            description="taipei_samples",
            folder="GEE_exports",
            fileNamePrefix="taipei_samples",
            fileFormat="CSV",
        )
        task.start()
        print("Export started. Watch the Tasks tab at "
              "https://code.earthengine.google.com, then move the CSV into "
              f"{C.SAMPLES} and run:\n"
              "    python src/assign_split.py")
        return

    print(f"Downloading in chunks of {args.chunk_size} polygons "
          f"(getInfo refuses more than 5000 elements at once)")
    df = fetch_in_chunks(img, parts, n_part, chunk_size=args.chunk_size)
    if df.empty:
        sys.exit("No samples returned. Check that the ROI polygons overlap the "
                 "boundary and that the composite is not fully cloud-masked.")

    got = df["poly_id"].nunique()
    if got < n_part:
        print(f"  NOTE: {n_part - got} polygon(s) yielded no pixels - too "
              f"small for the {C.SCALE} m grid, or fully cloud-masked.")

    df = assign_split(df, train_frac=args.train_frac)
    check_grid(df)

    df.to_csv(C.SAMPLE_CSV, index=False)
    print(f"\nWrote {len(df)} rows -> {C.SAMPLE_CSV}")

    report(df)


def check_grid(df, tol=0.5):
    """Verify the sample really landed on the Sentinel-2 10 m UTM grid.

    Snaps every returned coordinate to EXPORT_CRS and checks it sits at a pixel
    centre. If sampleRegions silently fell back to the composite's default
    lat/lon projection the points scatter across the cell instead, and the same
    native pixel gets picked up more than once - which inflates n without
    adding information and makes the run unreproducible.
    """
    if not {"lon", "lat"}.issubset(df.columns):
        return
    import numpy as np
    from pyproj import Transformer

    tf = Transformer.from_crs("EPSG:4326", C.EXPORT_CRS, always_xy=True)
    x, y = tf.transform(df["lon"].to_numpy(), df["lat"].to_numpy())

    # Pixel centres sit at a half-pixel offset from the multiple-of-SCALE grid.
    off = lambda v: np.abs(((v - C.SCALE / 2) % C.SCALE + C.SCALE / 2)
                           % C.SCALE - C.SCALE / 2)
    on_grid = float(np.mean((off(x) < tol) & (off(y) < tol)))

    cell = list(zip(np.floor(x / C.SCALE).astype(int),
                    np.floor(y / C.SCALE).astype(int)))
    n_dup = len(cell) - len(set(cell))

    print(f"\nGrid check: {100 * on_grid:.1f}% of samples on the {C.EXPORT_CRS} "
          f"{C.SCALE} m grid; {n_dup} duplicate pixel(s) "
          f"({100 * n_dup / max(len(cell), 1):.1f}%)")
    if on_grid < 0.95:
        print("  WARNING: the sample is NOT on the native grid. sampleRegions\n"
              "  fell back to the composite's default projection. Check that\n"
              "  `projection=` is still being passed in sample_polygons().")


def report(df):
    """Print the per-class pixel and polygon breakdown, and sanity-check it."""
    if C.CLASS_PROPERTY not in df.columns:
        return

    named = df.assign(cls=df[C.CLASS_PROPERTY].map(C.CLASS_NAMES))

    tab = named.groupby(["cls", "split"]).size().unstack(fill_value=0)
    print("\nPixels per class and split:\n", tab.to_string())

    if "poly_id" not in df.columns:
        return

    polys = (named.groupby(["cls", "split"])["poly_id"].nunique()
                  .unstack(fill_value=0))
    print("\nPolygons per class and split:\n", polys.to_string())

    for side in ("train", "val"):
        if side not in polys.columns:
            print(f"\nPROBLEM: no polygons at all on the {side} side.")
            continue
        empty = polys.index[polys[side] == 0].tolist()
        if empty:
            print(f"\nPROBLEM: {empty} have 0 polygons in {side}. "
                  f"Add training areas for those classes.")

    thin = polys.index[polys.get("val", 0) < 3].tolist()
    if thin:
        print(f"\nNOTE: {thin} have fewer than 3 validation polygons. "
              f"Per-class accuracy for them will be unstable, and that is a "
              f"limitation to state in the paper rather than hide.")


if __name__ == "__main__":
    main()
