"""Sampling design and effective sample size.

Produces
    fig_sampling_design   where the training polygons are, how big their pixel
                          clusters are, and what the clustering costs in
                          effective sample size
    fig_reference_design  allocation of the blind reference sample

The reviewer's objection was that 1829 validation pixels is a large sample and
the accuracies should therefore be precise. Panel (c) is the answer: those
1829 pixels come from 30 polygons, and pixels inside one polygon are
near-duplicates. Thinning the pixels while keeping every polygon barely moves
the effective sample size; thinning the polygons while keeping every pixel
moves it almost proportionally. Information lives in the polygons, not in the
pixels.

    python src/figures/fig_sampling_design.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import style as S  # noqa: E402
import maptools as M  # noqa: E402
from olofsson import icc_oneway  # noqa: E402

PRED_CSV = C.INTERIM / "test_predictions.csv"
TRAIN_COLOR = "#4c78a8"
VAL_COLOR = "#e45756"


# --------------------------------------------------------------------------
# Effective sample size
# --------------------------------------------------------------------------
def m0_of(clusters):
    """Donner's (1986) average cluster size for unequal clusters."""
    sizes = np.asarray([np.sum(clusters == c) for c in np.unique(clusters)],
                       dtype=float)
    n, k = sizes.sum(), len(sizes)
    if k < 2:
        return float(n)
    return float((n - (sizes ** 2).sum() / n) / (k - 1))


def n_eff(correct, clusters, icc=None):
    """n / deff with the Kish design effect, deff = 1 + (m0 - 1) * ICC.

    `icc` is passed in for the thinning curves. Re-estimating it inside every
    subsample looks more empirical but is not: with roughly sixty errors in the
    whole validation set, a small subsample can contain almost none, the
    between-cluster mean square collapses, and the ICC estimate swings from its
    true value to near zero. That produces a curve with a spike in it that is
    an artefact of the estimator rather than a property of the design. The ICC
    is a property of the population, so it is estimated once on the full sample
    and held fixed while the design is varied - which is the textbook way to
    plan a clustered sample in the first place.
    """
    correct = np.asarray(correct, dtype=float)
    clusters = np.asarray(clusters)
    n, k = len(correct), len(np.unique(clusters))
    if n == 0 or k == 0:
        return 0.0
    if icc is None:
        if correct.std() == 0:
            return float(k)
        icc, _ = icc_oneway(correct, clusters)
    deff = max(1.0, 1.0 + (m0_of(clusters) - 1.0) * icc)
    return float(n / deff)


def thinning_curves(df, model, icc, n_rep=12, seed=0):
    """Two ways of shrinking the validation set, and what each costs.

    pixel-thinning  keep every polygon, keep a fraction of the pixels in each
    polygon-thinning keep every pixel, keep a fraction of the polygons

    Both are plotted against the number of pixels retained, so the two curves
    are directly comparable: at any given pixel count, the polygon-thinned
    sample carries far more information than the pixel-thinned one.
    """
    rng = np.random.default_rng(seed)
    ok = (df[model] == df["y_true"]).to_numpy().astype(float)
    pid = df["poly_id"].to_numpy()
    polys = np.unique(pid)
    fracs = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.55, 0.7,
                      0.85, 1.0])

    px_rows, poly_rows = [], []
    for f in fracs:
        for _ in range(n_rep if f < 1.0 else 1):
            # --- thin pixels within every polygon -------------------------
            keep = np.zeros(len(df), dtype=bool)
            for p in polys:
                idx = np.flatnonzero(pid == p)
                take = max(1, int(round(f * len(idx))))
                keep[rng.choice(idx, take, replace=False)] = True
            px_rows.append((keep.sum(), n_eff(ok[keep], pid[keep], icc),
                            len(np.unique(pid[keep]))))

            # --- thin polygons, keep their pixels whole -------------------
            take = max(2, int(round(f * len(polys))))
            sel = rng.choice(polys, take, replace=False)
            keep = np.isin(pid, sel)
            poly_rows.append((keep.sum(), n_eff(ok[keep], pid[keep], icc),
                              len(sel)))

    cols = ["n_px", "n_eff", "n_poly"]
    return (pd.DataFrame(px_rows, columns=cols),
            pd.DataFrame(poly_rows, columns=cols))


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------
def polygons_with_split():
    """single_parts.gpkg joined to the sample's train/val assignment.

    poly_id is the index of the part in the exploded collection, but rather
    than trusting the GeoPackage to have preserved that order, each polygon is
    matched to the pixels that fall inside it. The join is then correct even if
    the file is rewritten or reordered.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    src = C.RAW / "single_parts.gpkg"
    if not src.exists():
        return None
    g = gpd.read_file(src).to_crs("EPSG:4326")

    d = pd.read_csv(C.SAMPLE_CSV)
    cent = (d.groupby("poly_id")
              .agg(lon=("lon", "mean"), lat=("lat", "mean"),
                   split=("split", "first"), cls=(C.CLASS_PROPERTY, "first"),
                   n_px=("lon", "size")).reset_index())
    pts = gpd.GeoDataFrame(
        cent, geometry=[Point(x, y) for x, y in zip(cent.lon, cent.lat)],
        crs="EPSG:4326")

    j = gpd.sjoin(g, pts, predicate="contains", how="left")
    # A concave polygon can fail to contain its own centroid; fall back to the
    # nearest sampled centroid for anything the containment join missed.
    miss = j["poly_id"].isna()
    if miss.any():
        near = gpd.sjoin_nearest(g[miss.to_numpy()], pts, how="left")
        j = pd.concat([j[~miss], near])
    return gpd.GeoDataFrame(j, geometry="geometry", crs="EPSG:4326") \
             .to_crs(C.EXPORT_CRS)


def panel_map(ax, gdf, boundary):
    if gdf is None:
        ax.axis("off")
        return
    if boundary is not None:
        boundary.plot(ax=ax, facecolor="#f7f7f7", edgecolor="#222222",
                      lw=0.5, zorder=1)
    for split, col in (("train", TRAIN_COLOR), ("val", VAL_COLOR)):
        sub = gdf[gdf["split"] == split]
        if len(sub):
            sub.plot(ax=ax, facecolor=col, edgecolor=col, lw=0.6, alpha=0.85,
                     zorder=3)
    # Polygons are a few hundred metres across on a 25 km city, so at column
    # width a filled patch is roughly one printed dot. An open ring at each
    # centroid, sized by the pixel count, makes both the spatial spread and
    # the variation in cluster size visible without inflating the polygons
    # themselves into something they are not.
    cen = gdf.geometry.representative_point()
    ax.scatter(cen.x, cen.y, s=np.clip(gdf["n_px"].to_numpy() / 4.0, 4, 60),
               facecolor="none",
               edgecolor=[TRAIN_COLOR if s == "train" else VAL_COLOR
                          for s in gdf["split"]],
               linewidth=0.5, zorder=5)
    M.tidy_map_axes(ax, boundary=boundary, crop=True, pad=400)
    M.scalebar(ax, 5000, loc=(0.05, 0.05), label="5 km", halo=True)
    M.north_arrow(ax, loc=(0.93, 0.85), halo=True)
    n_t = int((gdf["split"] == "train").sum())
    n_v = int((gdf["split"] == "val").sum())
    ax.legend(handles=[plt.Line2D([], [], marker="s", ls="", ms=4,
                                  color=TRAIN_COLOR,
                                  label=f"train ({n_t} polygons)"),
                       plt.Line2D([], [], marker="s", ls="", ms=4,
                                  color=VAL_COLOR,
                                  label=f"val ({n_v} polygons)")],
              loc="upper left", fontsize=6.2, handletextpad=0.3)
    S.panel_label(ax, "(a) training and validation polygons")


def panel_cluster_sizes(ax, d):
    """Pixels per polygon, by class and split. This is m, the cluster size."""
    ids = sorted(C.CLASS_NAMES)
    g = (d.groupby(["poly_id", C.CLASS_PROPERTY, "split"])
           .size().reset_index(name="n_px"))
    rng = np.random.default_rng(3)
    for i, cid in enumerate(ids):
        for split, dx, col in (("train", -0.16, TRAIN_COLOR),
                               ("val", 0.16, VAL_COLOR)):
            sub = g[(g[C.CLASS_PROPERTY] == cid) & (g["split"] == split)]
            if not len(sub):
                continue
            x = i + dx + rng.uniform(-0.07, 0.07, len(sub))
            ax.scatter(x, sub["n_px"], s=5, color=col, edgecolor="white",
                       linewidth=0.2, zorder=3)
            med = sub["n_px"].median()
            ax.plot([i + dx - 0.10, i + dx + 0.10], [med, med],
                    color="#333333", lw=0.8, zorder=4)
    ax.set_yscale("log")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels([C.CLASS_NAMES[c] for c in ids], rotation=18,
                       ha="right")
    ax.set_ylabel("pixels in the polygon (log)")
    for i, c in enumerate(ids):
        ax.axvspan(i - 0.45, i + 0.45, color=C.CLASS_COLORS[c], alpha=0.07,
                   lw=0, zorder=0)
    ax.grid(axis="y", color=S.LIGHT, lw=0.4)
    ax.set_axisbelow(True)
    S.panel_title(ax, "b", "cluster size: pixels per polygon")


def panel_neff(ax, px, poly, n_px_full, n_eff_full, n_poly_full, icc):
    """The two thinning curves.

    The 1:1 line is deliberately left off: at 1829 pixels it would sit forty
    times above the data and flatten both curves into the axis. The point is
    the contrast between the two curves, and the distance from independence is
    quoted in the annotation instead.
    """
    for d, col, lab in (
            (px, "#b2182b", f"thin pixels, keep all {n_poly_full} polygons"),
            (poly, "#2166ac", "thin polygons, keep their pixels")):
        m = d.groupby(pd.cut(d["n_px"], 22), observed=True).median()
        ax.plot(m["n_px"], m["n_eff"], "-o", ms=2.2, lw=0.9, color=col,
                label=lab, zorder=3)
    ax.plot(n_px_full, n_eff_full, "*", ms=8, color="#333333", zorder=5)

    # Two readings that make the contrast concrete: what half the pixels costs
    # against what half the polygons costs.
    half = n_px_full / 2
    for d, col in ((px, "#b2182b"), (poly, "#2166ac")):
        near = d.iloc[(d["n_px"] - half).abs().argsort()[:24]]
        ax.plot(half, near["n_eff"].median(), "o", ms=3.4, mfc="white",
                mec=col, mew=0.9, zorder=6)
    ax.axvline(half, color="#999999", lw=0.5, ls=":", zorder=1)
    ax.annotate(f"full sample: {n_px_full} px in {n_poly_full} polygons,"
                f"\nn$_{{eff}}$ = {n_eff_full:.0f}  (ICC = {icc:.2f})",
                (n_px_full, n_eff_full), textcoords="offset points",
                xytext=(-4, -14), ha="right", va="top", fontsize=6.2,
                color="#333333")
    # Low on the axis: at the top this label lands on the saturated red curve.
    ax.text(half, 0.5, " half the pixels", fontsize=6.2, va="bottom",
            color="#777777")
    ax.set_xlabel("validation pixels retained")
    ax.set_ylabel("effective sample size n$_{eff}$")
    ax.set_xlim(0, n_px_full * 1.06)
    ax.set_ylim(0, max(px["n_eff"].max(), poly["n_eff"].max()) * 1.30)
    ax.legend(loc="upper left", fontsize=6.2, handletextpad=0.4)
    ax.grid(color=S.LIGHT, lw=0.4)
    ax.set_axisbelow(True)
    S.panel_title(ax, "c", "information lives in polygons, not pixels")


def panel_neff_by_class(ax, model_order):
    """n_i against n_eff_i per class, every model, on a log-log square."""
    p = C.TABLES / "olofsson_per_class.csv"
    S.require(p)
    d = pd.read_csv(p)
    for m in model_order:
        sub = d[d.model == m]
        ax.scatter(sub["n_i"], sub["n_eff_i"], s=9,
                   color=S.MODEL_COLORS.get(m, "#666666"), edgecolor="white",
                   linewidth=0.25, label=m, zorder=3)
    lo, hi = 1, max(d["n_i"].max(), 10) * 1.6
    ax.plot([lo, hi], [lo, hi], ls=":", lw=0.6, color="#888888")
    ax.text(hi, hi, "1:1 ", fontsize=6.2, ha="right", va="bottom",
            color="#666666")
    for frac, lab in ((0.1, "10x"), (0.01, "100x")):
        ax.plot([lo, hi], [lo * frac, hi * frac], ls="--", lw=0.5,
                color="#cccccc")
        ax.text(hi, hi * frac, f"{lab} ", fontsize=6.2, ha="right",
                va="bottom", color="#999999")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(1, hi)
    ax.set_xlabel("validation pixels in the class, n$_i$")
    ax.set_ylabel("effective n$_{eff,i}$")
    ax.grid(color=S.LIGHT, lw=0.4, which="both")
    ax.set_axisbelow(True)
    S.panel_title(ax, "d", "the design effect, class by class")


def fig_design():
    S.require(C.SAMPLE_CSV, PRED_CSV)
    d = pd.read_csv(C.SAMPLE_CSV)
    pred = pd.read_csv(PRED_CSV)

    cmp_path = C.TABLES / "model_comparison.csv"
    order = (list(pd.read_csv(cmp_path).sort_values("OA", ascending=False)
                  ["model"]) if cmp_path.exists() else S.MODEL_ORDER)
    model = next((m for m in order if m in pred.columns), None)

    ok_full = (pred[model] == pred["y_true"]).to_numpy().astype(float)
    icc, _ = icc_oneway(ok_full, pred["poly_id"].to_numpy())
    px, poly = thinning_curves(pred, model, icc)
    ne_full = n_eff(ok_full, pred["poly_id"].to_numpy(), icc)

    fig, axes = plt.subplots(2, 2, figsize=(S.DOUBLE, S.DOUBLE * 0.80),
                             gridspec_kw=dict(wspace=0.28, hspace=0.42))
    panel_map(axes[0, 0], polygons_with_split(), M.load_boundary())
    panel_cluster_sizes(axes[0, 1], d)
    panel_neff(axes[1, 0], px, poly, len(pred), ne_full,
               pred["poly_id"].nunique(), icc)
    panel_neff_by_class(axes[1, 1], [m for m in S.MODEL_ORDER
                                     if m in set(pd.read_csv(
                                         C.TABLES / "olofsson_per_class.csv"
                                     ).model)])
    axes[1, 1].legend(ncol=2, fontsize=6.2, loc="upper left",
                      handletextpad=0.2, columnspacing=0.6)
    fig.text(0.5, 0.005, f"panel (c) uses {model}, the highest-OA model; the "
             "other five give the same shape", ha="center", fontsize=6.2,
             color="#555555")
    return fig


def fig_reference():
    """Allocation of the blind reference sample.

    Proportional allocation would give Plantation 35 points, which cannot
    support a user's accuracy to better than about +/- 16 points. The
    allocation actually used raises every rare class to a floor, at a small
    cost in the precision of the overall estimate.
    """
    p = C.TABLES / "reference_sample_design.csv"
    S.require(p)
    d = pd.read_csv(p).sort_values("class_id")
    y = np.arange(len(d))
    colors = [C.CLASS_COLORS[int(c)] for c in d["class_id"]]

    fig, axes = plt.subplots(1, 3, figsize=(S.DOUBLE, S.DOUBLE * 0.26),
                             gridspec_kw=dict(width_ratios=[1, 1, 0.9],
                                              wspace=0.38))

    ax = axes[0]
    ax.barh(y, d["W_i"], color=colors, edgecolor="#333333", lw=0.3,
            height=0.7)
    for i, (w, a) in enumerate(zip(d["W_i"], d["map_area_km2"])):
        ax.text(w, i, f"  {a:.0f} km$^2$", fontsize=6.2, va="center",
                color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(d["class"])
    ax.invert_yaxis()
    ax.set_xlim(0, d["W_i"].max() * 1.45)
    ax.set_xlabel("stratum weight W$_i$")
    S.panel_title(ax, "a", "map strata")
    ax.grid(axis="x", color=S.LIGHT, lw=0.4)
    ax.set_axisbelow(True)

    ax = axes[1]
    h = 0.34
    ax.barh(y - h / 2, d["proportional_n"], height=h, color="#bbbbbb",
            edgecolor="none", label="proportional")
    # The allocated bars carry the class palette, so the legend swatch for them
    # is drawn neutral - a blue swatch would read as "Water" rather than as
    # "allocated".
    ax.barh(y + h / 2, d["allocated_n"], height=h, color=colors,
            edgecolor="#333333", lw=0.3)
    ax.barh([np.nan], [np.nan], height=h, color="white", edgecolor="#333333",
            lw=0.5, label="allocated (class colour)")
    for i, (pn, an) in enumerate(zip(d["proportional_n"], d["allocated_n"])):
        if an > pn:
            ax.text(an, i + h / 2, f"  +{an - pn:.0f}", fontsize=6.2,
                    va="center", color="#b2182b")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlim(0, d["allocated_n"].max() * 1.28)
    ax.set_xlabel(f"reference points (total {int(d['allocated_n'].sum())})")
    ax.legend(loc="lower right", fontsize=6.2, handletextpad=0.4)
    S.panel_title(ax, "b", "allocation raises the rare classes")
    ax.grid(axis="x", color=S.LIGHT, lw=0.4)
    ax.set_axisbelow(True)

    # (c) what the allocation buys: the width of a UA interval at the prior
    # accuracy, under each allocation. Wald half-width, 95%.
    ax = axes[2]
    def halfwidth(n, p):
        return 1.96 * np.sqrt(np.clip(p * (1 - p), 1e-9, None) / np.maximum(n, 1))
    hp = halfwidth(d["proportional_n"].to_numpy(), d["prior_UA"].to_numpy())
    ha = halfwidth(d["allocated_n"].to_numpy(), d["prior_UA"].to_numpy())
    ax.barh(y - h / 2, 100 * hp, height=h, color="#bbbbbb", edgecolor="none")
    ax.barh(y + h / 2, 100 * ha, height=h, color=colors, edgecolor="#333333",
            lw=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("95% half-width of user's accuracy (pp)")
    S.panel_title(ax, "c", "precision gained")
    ax.grid(axis="x", color=S.LIGHT, lw=0.4)
    ax.set_axisbelow(True)
    return fig


def main():
    S.use()
    print("fig_sampling_design")
    S.save(fig_design(), "fig_sampling_design", dpi=S.DPI_RASTER)
    print("fig_reference_design")
    S.save(fig_reference(), "fig_reference_design")


if __name__ == "__main__":
    main()
