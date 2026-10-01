"""Apply the tuned classifier to the whole city and measure the mapped areas.

Two things depend on this:

  1. The Olofsson estimator needs W_i, the fraction of the map occupied by each
     class. Those weights cannot come from the sample - they have to come from
     the classified map itself.

  2. The manuscript's area figures are internally inconsistent: the Study Area
     section says 272 sq.km while back-calculating from the Results tables gives
     ~292.8 sq.km. Whatever this script reports for the total is the number the
     revised paper should use, and the boundary asset is where the discrepancy
     will be visible.

The composite is exported in UTM 51N rather than lat/lon so that a pixel is
exactly 10 x 10 m and areas are a straight pixel count. Doing the area sum in
EPSG:4326 would silently inflate it, since a degree of longitude is not a
constant distance.

Every tuned model is applied to the composite, not just the winner, so the
manuscript can show the maps side by side and so the spread in per-class area
between models is visible. The leading model by hold-out OA supplies the area
weights at the top level of map_areas.json.

Usage:
    python src/classify_map.py                  # every tuned model
    python src/classify_map.py --model best
    python src/classify_map.py --model "SVM (RBF),Random Forest"
    python src/classify_map.py --skip-download  # composite already on disk
    python src/classify_map.py --tile-mb 20     # smaller download tiles
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

import config as C

UTM_TAIPEI = C.EXPORT_CRS          # WGS84 / UTM zone 51N; shared with sampling
COMPOSITE_TIF = C.EXPORTS / "s2_composite_2020.tif"

# getDownloadURL refuses anything over 48 MiB. Aim well under it: the server
# estimate and the actual payload do not always agree, and a tile that fails
# costs a round trip.
DOWNLOAD_LIMIT_MB = 50331648 / 1e6

# The naive pixel-count estimate below runs about 1.5x under what Earth Engine
# actually bills the request at - on this AOI it predicted 48 MB against the
# 69.7 MB the server reported. So the target is set well under the cap rather
# than close to it. Tiles that still come back too large are halved and
# retried, so the estimate only has to be roughly right.
DEFAULT_TILE_MB = 20.0


# ---------------------------------------------------------------------------
# Composite download
# ---------------------------------------------------------------------------
def _bounds_utm(aoi):
    """AOI bounding box in UTM 51N metres.

    The bounds are fetched in lat/lon and reprojected client-side with pyproj
    rather than asking Earth Engine to reproject, because ee.Geometry.bounds
    with a projected CRS returns coordinates whose interpretation depends on
    the geometry's own projection and is easy to get subtly wrong.
    """
    from pyproj import Transformer

    ring = aoi.bounds(maxError=1).coordinates().getInfo()[0]
    lons = [c[0] for c in ring]
    lats = [c[1] for c in ring]

    # Densify each edge before reprojecting. A lat/lon rectangle does not map
    # to a rectangle in UTM - the edges bow, and Taipei sits 1.5 degrees off
    # the zone 51 central meridian, so taking only the two opposite corners
    # would clip a sliver off the north and south edges.
    dense_lon, dense_lat = [], []
    for i in range(len(ring)):
        (lo0, la0), (lo1, la1) = ring[i], ring[(i + 1) % len(ring)]
        t = np.linspace(0, 1, 25)
        dense_lon += list(lo0 + t * (lo1 - lo0))
        dense_lat += list(la0 + t * (la1 - la0))

    tf = Transformer.from_crs("EPSG:4326", UTM_TAIPEI, always_xy=True)
    xs, ys = tf.transform(dense_lon, dense_lat)

    # One pixel of slack on each side. Pixels outside the boundary come back as
    # zero and classify_raster already treats those as nodata, so over-reaching
    # is free while under-reaching silently truncates the map.
    #
    # The box is then snapped outwards to whole multiples of SCALE so the
    # exported grid coincides with Sentinel-2's own 10 m grid rather than
    # starting at an arbitrary float. Without this the map is resampled onto a
    # grid offset by up to half a pixel, which shifts the classified raster
    # relative to the imagery and makes the tile boundaries depend on the
    # floating-point bounds of the AOI.
    pad = C.SCALE
    s = C.SCALE
    return (np.floor((min(xs) - pad) / s) * s, np.floor((min(ys) - pad) / s) * s,
            np.ceil((max(xs) + pad) / s) * s, np.ceil((max(ys) + pad) / s) * s)


def _estimate_mb(x0, y0, x1, y1, n_bands, scale, bytes_per_px=2):
    nx = int(np.ceil((x1 - x0) / scale))
    ny = int(np.ceil((y1 - y0) / scale))
    return nx * ny * n_bands * bytes_per_px / 1e6


def _grid(x0, y0, x1, y1, n_bands, scale, target_mb):
    """Split the box into the smallest square-ish grid whose tiles fit."""
    total = _estimate_mb(x0, y0, x1, y1, n_bands, scale)
    n = int(np.ceil(np.sqrt(total / target_mb))) if total > target_mb else 1
    xs = np.linspace(x0, x1, n + 1)
    ys = np.linspace(y0, y1, n + 1)
    return [(xs[i], ys[j], xs[i + 1], ys[j + 1])
            for j in range(n) for i in range(n)], n, total


def _download_tile(ee, geemap, img, box, path, depth=0, max_depth=3):
    """Download one tile, halving it and retrying if the server rejects it."""
    x0, y0, x1, y1 = box
    region = ee.Geometry.Rectangle([x0, y0, x1, y1], proj=UTM_TAIPEI,
                                   geodesic=False)
    try:
        geemap.ee_export_image(img, filename=str(path), scale=C.SCALE,
                               region=region, crs=UTM_TAIPEI,
                               file_per_band=False)
        if path.exists():
            return [path]
        raise RuntimeError("export reported success but wrote no file")
    except Exception as exc:
        if depth >= max_depth:
            raise RuntimeError(f"tile {box} still failing after "
                               f"{max_depth} subdivisions: {exc}") from exc
        print(f"    tile too large ({exc}); splitting into 4")
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        out = []
        for k, sub in enumerate([(x0, y0, mx, my), (mx, y0, x1, my),
                                 (x0, my, mx, y1), (mx, my, x1, y1)]):
            out += _download_tile(ee, geemap, img,
                                  sub, path.with_name(f"{path.stem}_{k}.tif"),
                                  depth + 1, max_depth)
        return out


def download_composite(extended=False, path=COMPOSITE_TIF,
                       target_mb=DEFAULT_TILE_MB):
    """Download the composite, tiling it if it exceeds the direct-download cap.

    A 272 sq.km city at 10 m in four int16 bands is about 66 MB, which is over
    Earth Engine's 48 MiB getDownloadURL ceiling. Rather than sending the user
    to Export.image.toDrive and a manual round trip through Drive, the AOI is
    split into a grid, each tile is pulled separately, and the pieces are
    mosaicked locally. Tiles that still come back too large are halved and
    retried, so this degrades gracefully if the estimate is optimistic.
    """
    import ee
    import geemap
    import rasterio
    from rasterio.merge import merge
    from gee_export_samples import build_composite, init_ee

    init_ee()
    img, aoi = build_composite(extended=extended)
    bands = img.bandNames().getInfo()
    print(f"Exporting composite bands {bands} at {C.SCALE} m in {UTM_TAIPEI}")

    # Reflectance is 0-1 after the /10000 scaling; int16 at 1e4 keeps full
    # precision in a quarter of the bytes of float32.
    out = img.multiply(10000).toInt16()

    x0, y0, x1, y1 = _bounds_utm(aoi)
    boxes, n, total_mb = _grid(x0, y0, x1, y1, len(bands), C.SCALE, target_mb)
    print(f"  AOI {(x1 - x0) / 1000:.1f} x {(y1 - y0) / 1000:.1f} km, "
          f"~{total_mb:.0f} MB uncompressed "
          f"(direct-download cap {DOWNLOAD_LIMIT_MB:.0f} MB)")

    path.parent.mkdir(parents=True, exist_ok=True)

    if len(boxes) == 1:
        print("  fits in one request")
        _download_tile(ee, geemap, out, boxes[0], path)
        pieces = [path]
    else:
        print(f"  splitting into a {n} x {n} grid of ~{total_mb / len(boxes):.0f} "
              f"MB tiles")
        tmp = path.parent / "_tiles"
        tmp.mkdir(exist_ok=True)
        pieces = []
        for i, box in enumerate(boxes, 1):
            # Skip tiles that miss the boundary entirely - a coastal or
            # mountainous AOI wastes several requests on empty corners.
            rect = ee.Geometry.Rectangle(list(box), proj=UTM_TAIPEI,
                                         geodesic=False)
            if not aoi.intersects(rect, maxError=10).getInfo():
                print(f"  tile {i}/{len(boxes)}: outside the boundary, skipped")
                continue
            print(f"  tile {i}/{len(boxes)} ...")
            pieces += _download_tile(ee, geemap, out, box, tmp / f"tile_{i:02d}.tif")

        if not pieces:
            sys.exit("No tiles intersected the boundary - check the AOI asset.")

        print(f"  mosaicking {len(pieces)} tiles -> {path}")
        srcs = [rasterio.open(p) for p in pieces]
        try:
            mosaic, transform = merge(srcs)
            profile = srcs[0].profile.copy()
            profile.update(height=mosaic.shape[1], width=mosaic.shape[2],
                           transform=transform, count=mosaic.shape[0],
                           compress="lzw")
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(mosaic)
                # geemap does not always carry band names through; set them
                # explicitly so classify_raster can match features by name
                # instead of trusting positional order.
                dst.descriptions = tuple(bands)
        finally:
            for s in srcs:
                s.close()

    if not path.exists():
        sys.exit(
            f"Download produced no file at {path}.\n"
            f"Fall back to the Code Editor:\n"
            f"    Export.image.toDrive({{image: composite.multiply(10000).toInt16(),\n"
            f"      description:'s2_composite_2020', folder:'GEE_exports',\n"
            f"      region: select_feature.geometry(), scale: 10,\n"
            f"      crs: '{UTM_TAIPEI}', maxPixels: 1e13}});\n"
            f"then put the file at {path} and re-run with --skip-download."
        )
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return bands


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def resolve_models(requested=None):
    """Return the model names to classify with, best-first.

    `requested` is None (every tuned model), "best", or a comma-separated list.
    Ordering follows hold-out OA so that "best-first" is meaningful downstream:
    map_areas.json carries the leading model at its top level, which is what
    olofsson.py and draw_reference_sample.py read.
    """
    hp_path = C.TABLES / "best_hyperparameters.json"
    cmp_path = C.TABLES / "model_comparison.csv"
    if not hp_path.exists():
        sys.exit(f"{hp_path} not found. Run `python src/train_compare.py` first.")
    tuned = list(json.loads(hp_path.read_text()))

    if cmp_path.exists():
        ranked = [m for m in pd.read_csv(cmp_path)
                  .sort_values("OA", ascending=False)["model"] if m in tuned]
        ranked += [m for m in tuned if m not in ranked]
    else:
        ranked = tuned

    if requested is None:
        return ranked
    if requested.strip().lower() in ("all", "*"):
        return ranked
    if requested.strip().lower() == "best":
        return ranked[:1]

    want = [m.strip() for m in requested.split(",") if m.strip()]
    unknown = [m for m in want if m not in tuned]
    if unknown:
        sys.exit(f"Unknown model(s) {unknown}. Available: {tuned}")
    return sorted(want, key=ranked.index)


def fit_models(names):
    """Refit each named model on the training split with its tuned settings.

    Yields (name, fitted_estimator). The training data and the split are loaded
    once and shared, so every map comes from the same pixels - otherwise the
    per-model area differences would partly reflect a different split rather
    than a different classifier.
    """
    from train_compare import build_models, load_data, split_data

    best_params = json.loads((C.TABLES / "best_hyperparameters.json").read_text())
    X, y, groups, feats, df = load_data(str(C.SAMPLE_CSV))
    Xtr, _, ytr, _, _, _, desc = split_data(X, y, groups, df)
    builders = dict(build_models())
    print(f"Refitting on {len(ytr)} training pixels ({desc})")

    def _iter():
        for name in names:
            est = builders[name][0]
            est.set_params(**best_params[name])
            est.fit(Xtr, ytr)
            yield name, est

    return _iter(), feats


# ---------------------------------------------------------------------------
# Classify
# ---------------------------------------------------------------------------
def classify_raster(est, feats, src_path, dst_path):
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(src_path) as src:
        band_names = list(src.descriptions)
        if not all(band_names):
            band_names = C.BANDS_PUBLISHED[:src.count]
            print(f"  raster has no band names; assuming {band_names}")

        missing = [f for f in feats if f not in band_names]
        if missing:
            sys.exit(
                f"The model needs {feats} but the raster has {band_names}.\n"
                f"Missing: {missing}. If the model was trained with "
                f"--extended, download the composite with --extended too."
            )
        order = [band_names.index(f) + 1 for f in feats]

        profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=0, compress="lzw")

        # Pixels outside the city boundary are masked in Earth Engine, and the
        # int16 export fills them with the band nodata value (-32768), NOT with
        # zero. Testing only for zero classified the whole 21 x 28 km bounding
        # box - 597 sq.km against the city's real 272 - because -32768 passed
        # as a legitimate reflectance.
        nodata = src.nodata

        counts = {}
        with rasterio.open(dst_path, "w", **profile) as dst:
            for _, win in src.block_windows(1):
                raw = src.read(order, window=win)
                arr = raw.astype("float32") / 10000.0
                nb, h, w = arr.shape
                flat = arr.reshape(nb, -1).T
                rawflat = raw.reshape(nb, -1).T

                valid = np.isfinite(flat).all(axis=1) & (flat != 0).any(axis=1)
                if nodata is not None:
                    valid &= ~(rawflat == nodata).any(axis=1)
                out = np.zeros(flat.shape[0], dtype="uint8")
                if valid.any():
                    out[valid] = est.predict(flat[valid]).astype("uint8")

                dst.write(out.reshape(h, w), 1, window=win)
                for k, v in zip(*np.unique(out[out > 0], return_counts=True)):
                    counts[int(k)] = counts.get(int(k), 0) + int(v)

        px_area_m2 = abs(src.transform.a * src.transform.e)
        if src.crs and src.crs.is_geographic:
            print("  WARNING: raster is in a geographic CRS. Pixel area is not "
                  "constant and the areas below are wrong. Re-export in "
                  f"{UTM_TAIPEI}.")

    return counts, px_area_m2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="'all' (default), 'best', or a comma-separated list "
                         "of model names from best_hyperparameters.json")
    ap.add_argument("--extended", action="store_true",
                    help="6 bands + indices; must match how the model was trained")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--composite", default=str(COMPOSITE_TIF))
    ap.add_argument("--tile-mb", type=float, default=DEFAULT_TILE_MB,
                    help=f"target size per download tile (default "
                         f"{DEFAULT_TILE_MB:.0f} MB)")
    args = ap.parse_args()

    from pathlib import Path
    src_path = Path(args.composite)
    if not src_path.is_absolute():
        src_path = C.ROOT / src_path

    if not args.skip_download:
        download_composite(extended=args.extended, path=src_path,
                           target_mb=args.tile_mb)
    elif not src_path.exists():
        sys.exit(f"--skip-download given but {src_path} is missing.")

    names = resolve_models(args.model)
    print(f"Classifying with: {', '.join(names)}")
    fitted, feats = fit_models(names)

    all_rows, per_model, rasters = [], {}, {}
    for name, est in fitted:
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        dst = C.EXPORTS / f"taipei_lulc_{safe}.tif"
        print(f"\n{name} -> {dst}")
        counts, px_m2 = classify_raster(est, feats, str(src_path), str(dst))
        if not counts:
            sys.exit(f"{name}: no valid pixels classified - check the composite.")

        areas = {k: v * px_m2 / 1e6 for k, v in sorted(counts.items())}
        total = sum(areas.values())
        rasters[name] = str(dst)
        per_model[name] = {
            "raster": str(dst),
            "pixel_area_m2": px_m2,
            "total_area_km2": total,
            "areas_km2": {str(k): areas[k] for k in sorted(areas)},
            "pixel_counts": {str(k): counts[k] for k in sorted(counts)},
        }
        for k in sorted(areas):
            all_rows.append({"model": name, "class_id": k,
                             "class": C.CLASS_NAMES.get(k, str(k)),
                             "pixels": counts[k],
                             "area_km2": round(areas[k], 4),
                             "percent": round(100 * areas[k] / total, 2)})
        print(pd.DataFrame(all_rows[-len(areas):])
              .drop(columns="model").to_string(index=False))
        print(f"  TOTAL: {total:.2f} sq.km")

    tab = pd.DataFrame(all_rows)
    tab.to_csv(C.TABLES / "map_areas.csv", index=False)

    # Side-by-side area comparison. Class areas that swing widely between
    # models are the ones the accuracy figures are least able to pin down, and
    # that is worth saying in the paper rather than reporting one model's
    # areas as if they were the areas.
    if len(names) > 1:
        wide = tab.pivot(index="class", columns="model", values="area_km2")
        wide = wide[[n for n in names if n in wide.columns]]
        wide.loc["TOTAL"] = wide.sum()
        wide["range_km2"] = (wide.max(axis=1) - wide.min(axis=1)).round(2)
        print("\nArea by model (sq.km):\n", wide.round(2).to_string())
        wide.round(4).to_csv(C.TABLES / "map_areas_by_model.csv")

    lead = names[0]
    total = per_model[lead]["total_area_km2"]
    print(f"\nArea weights for the Olofsson estimator come from {lead} "
          f"({total:.2f} sq.km).")
    print("  Manuscript Study Area section says 272 sq.km; the Results tables "
          "imply ~292.8 sq.km.\n  Whichever this matches is the one to keep; "
          "if it matches neither, the boundary asset is the problem.")

    # Top level stays single-model so olofsson.py and draw_reference_sample.py
    # keep working unchanged; the rest sits under "models".
    payload = dict(per_model[lead])
    payload.update({"model": lead, "features": feats,
                    "models": per_model, "rasters": rasters})
    (C.TABLES / "map_areas.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {C.TABLES}/map_areas.json, map_areas.csv"
          + (", map_areas_by_model.csv" if len(names) > 1 else ""))
    print("Next: python src/olofsson.py")


if __name__ == "__main__":
    main()
