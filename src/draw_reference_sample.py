"""Design and draw a proper stratified random reference sample.

This is the piece that actually answers the editor's second comment. Everything
else in the pipeline squeezes more honesty out of the existing polygon-derived
sample; this replaces it.

Two problems with the current validation data, neither fixable by reweighting:

  - It is clustered. 1,999 pixels come from 30 polygons, and pixels inside a
    polygon are near-duplicates. The effective sample size is around 50.
  - It is purposive. The polygons were drawn over unambiguous, spectrally pure
    examples, so the sample systematically excludes the mixed and edge pixels
    where classifiers actually fail. That biases accuracy upward by an amount
    the data cannot reveal.

A stratified random sample drawn from the MAP - one point at a time, each
interpreted independently - has neither problem, and it is what Olofsson et al.
(2014) call the good-practice minimum.

Sample size follows Olofsson eq. 13:

    n = ( sum_i W_i * S_i / S(O_hat) )^2 ,   S_i = sqrt(U_i * (1 - U_i))

with U_i a prior guess at each class's user's accuracy and S(O_hat) the target
standard error on overall accuracy. Allocation is then proportional to W_i, but
with a floor per stratum, because proportional allocation alone would give swamp
about four points and leave its producer's accuracy exactly as unestimable as it
is now.

Usage:
    python src/draw_reference_sample.py                      # design + draw
    python src/draw_reference_sample.py --target-se 0.01     # tighter CI
    python src/draw_reference_sample.py --min-per-class 100
    python src/draw_reference_sample.py --design-only        # just the numbers
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

import config as C


# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------
def sample_size(W, U, target_se):
    """Olofsson eq. 13. W and U are arrays over strata."""
    W = np.asarray(W, dtype=float)
    U = np.asarray(U, dtype=float)
    S = np.sqrt(np.clip(U * (1 - U), 0, None))
    return float(((W * S).sum() / target_se) ** 2)


def allocate(n_total, W, labels, min_per_class=75, cap_frac=0.6):
    """Proportional allocation with a floor, then renormalise the remainder.

    The floor is the whole point. Under strict proportional allocation swamp
    (W ~ 0.06) would get a handful of points and its producer's accuracy would
    stay unestimable - which is the exact criticism the revision has to answer.
    Olofsson et al. explicitly endorse departing from proportional allocation
    for rare classes, at some cost to the precision of overall accuracy.
    """
    W = np.asarray(W, dtype=float)
    n = np.maximum(np.round(W * n_total), min_per_class).astype(int)

    # Don't let one dominant class swallow the sample.
    cap = int(cap_frac * max(n_total, n.sum()))
    n = np.minimum(n, cap)
    return dict(zip(labels, n.tolist()))


def load_weights(areas_path):
    from pathlib import Path
    path = Path(areas_path) if areas_path else (C.TABLES / "map_areas.json")
    if not path.exists():
        sys.exit(f"{path} not found. Run `python src/classify_map.py` first.")
    payload = json.loads(path.read_text())
    areas = {int(k): float(v) for k, v in payload["areas_km2"].items()}
    total = float(payload.get("total_area_km2", sum(areas.values())))
    labels = sorted(areas)
    W = np.array([areas[l] / total for l in labels])
    return labels, W, areas, total, payload


def prior_ua(labels, per_class_path=None):
    """Prior user's accuracy per class, for the sample-size formula.

    Uses the existing (optimistic) estimates if they are on disk. Olofsson's
    formula only needs the right order of magnitude, and being wrong here costs
    precision rather than validity - but note that priors taken from an
    optimistic sample give an optimistically SMALL n, so the values are shaded
    downward below.
    """
    from pathlib import Path
    path = Path(per_class_path) if per_class_path \
        else (C.TABLES / "olofsson_per_class.csv")
    if not path.exists():
        print("  No prior accuracies on disk; assuming UA = 0.80 for common "
              "classes and 0.65 for rare ones.")
        return None
    df = pd.read_csv(path)
    df = df[df["model"] == df["model"].iloc[0]]
    got = dict(zip(df["class_id"].astype(int), df["UA"].astype(float)))
    # Shade toward 0.5 (max variance) so the design is not undersized.
    return {k: float(np.clip(0.5 + 0.8 * (v - 0.5), 0.4, 0.95))
            for k, v in got.items() if np.isfinite(v)}


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------
def draw_points(alloc, raster_path, seed=C.RANDOM_SEED):
    """Simple random points within each map stratum, drawn from the raster."""
    import rasterio

    rng = np.random.default_rng(seed)
    rows = []
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs

        for cls, n in alloc.items():
            ys, xs = np.nonzero(arr == cls)
            if len(ys) == 0:
                print(f"  class {cls} has no pixels in the map; skipped")
                continue
            if n > len(ys):
                print(f"  class {cls}: only {len(ys)} pixels, requesting {n}; "
                      f"taking all")
                n = len(ys)
            pick = rng.choice(len(ys), size=n, replace=False)
            for p in pick:
                r, c = int(ys[p]), int(xs[p])
                # Pixel centre, not corner.
                x, y = transform * (c + 0.5, r + 0.5)
                rows.append({"point_id": len(rows) + 1,
                             "map_class": int(cls),
                             "map_class_name": C.CLASS_NAMES.get(int(cls), str(cls)),
                             "row": r, "col": c, "x": x, "y": y,
                             "reference_class": "",
                             "interpreter": "", "confidence": "", "notes": ""})

    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("No points drawn.")

    # Add lon/lat so the points can be pasted straight into the Code Editor.
    try:
        from pyproj import Transformer
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(df["x"].to_numpy(), df["y"].to_numpy())
        df["lon"], df["lat"] = lon, lat
    except Exception as exc:
        print(f"  (no lon/lat: {exc})")

    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def write_geojson(df, path):
    feats = [{
        "type": "Feature",
        "geometry": {"type": "Point",
                     "coordinates": [float(r["lon"]), float(r["lat"])]},
        "properties": {"point_id": int(r["point_id"]),
                       "map_class": int(r["map_class"])},
    } for _, r in df.iterrows() if "lon" in df.columns]
    if feats:
        path.write_text(json.dumps(
            {"type": "FeatureCollection", "features": feats}))
        return True
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", default=None)
    ap.add_argument("--raster", default=None,
                    help="classified GeoTIFF from classify_map.py")
    ap.add_argument("--target-se", type=float, default=0.015,
                    help="target standard error on overall accuracy "
                         "(0.015 -> about +/-3 percentage points at 95%%)")
    ap.add_argument("--min-per-class", type=int, default=75)
    ap.add_argument("--design-only", action="store_true")
    args = ap.parse_args()

    labels, W, areas, total, payload = load_weights(args.areas)
    priors = prior_ua(labels)

    if priors is None:
        U = np.array([0.80 if w > 0.1 else 0.65 for w in W])
    else:
        U = np.array([priors.get(l, 0.75) for l in labels])

    n_req = sample_size(W, U, args.target_se)
    n_total = int(np.ceil(n_req / 25) * 25)      # round up to a tidy number
    alloc = allocate(n_total, W, labels, min_per_class=args.min_per_class)
    n_drawn = sum(alloc.values())

    design = pd.DataFrame({
        "class_id": labels,
        "class": [C.CLASS_NAMES.get(l, str(l)) for l in labels],
        "map_area_km2": [round(areas[l], 3) for l in labels],
        "W_i": np.round(W, 4),
        "prior_UA": np.round(U, 3),
        "proportional_n": np.round(W * n_total).astype(int),
        "allocated_n": [alloc[l] for l in labels],
    })

    print(f"Target SE on overall accuracy: {args.target_se} "
          f"(95% CI approx +/-{100*1.96*args.target_se:.1f} pp)")
    print(f"Olofsson eq. 13 requires n = {n_req:.0f}; using {n_total}")
    print(f"Floor of {args.min_per_class} per class raises the drawn total "
          f"to {n_drawn}\n")
    print(design.to_string(index=False))
    design.to_csv(C.TABLES / "reference_sample_design.csv", index=False)
    print(f"\nSaved -> {C.TABLES}/reference_sample_design.csv")

    print("\nThis table is the sample-design table the editor asked for in "
          "comment 1.\nIt still needs three columns only you can fill: the "
          "source imagery used for\ninterpretation, the interpreter, and the "
          "date.")

    if args.design_only:
        return

    raster = args.raster
    if raster is None:
        model = payload.get("model", "")
        safe = model.replace(" ", "_").replace("(", "").replace(")", "")
        guess = C.EXPORTS / f"taipei_lulc_{safe}.tif"
        if not guess.exists():
            sys.exit(f"No --raster given and {guess} not found. "
                     f"Run `python src/classify_map.py` first, or pass "
                     f"--design-only.")
        raster = str(guess)

    print(f"\nDrawing points from {raster}")
    pts = draw_points(alloc, raster)

    out_csv = C.SAMPLES / "reference_points_blank.csv"
    pts.to_csv(out_csv, index=False)
    print(f"Wrote {len(pts)} points -> {out_csv}")

    out_geo = C.SAMPLES / "reference_points.geojson"
    if write_geojson(pts, out_geo):
        print(f"Wrote {out_geo}")

    print("\nThe points are shuffled so the map class is not visible in row "
          "order.\nInterpret `reference_class` WITHOUT looking at `map_class` - "
          "otherwise the\nreference labels inherit the map's errors and the "
          "assessment is circular.\nUpload the completed CSV as a GEE asset, or "
          "point src/olofsson.py at it.")


if __name__ == "__main__":
    main()
