"""Classified-map figures.

Produces
    fig_lulc_panel      six classified maps, one per model, shared legend
    fig_lulc_best       the leading model on its own, with an area inset
    fig_model_agreement per-pixel agreement across the six maps
    fig_study_area      Sentinel-2 true colour with the city outline

Replaces the ArcGIS-drawn maps in the submitted manuscript. Colours come
straight from config.CLASS_COLORS, i.e. the palette the GEE script uses, so
the maps are the same ones the Code Editor draws.

    python src/figures/fig_maps.py [--models A,B] [--no-agreement]
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import style as S  # noqa: E402
import maptools as M  # noqa: E402


def raster_path(model):
    return C.EXPORTS / f"taipei_lulc_{S.MODEL_SLUG[model]}.tif"


def available_models(requested=None):
    """Models that both exist on disk and are ranked in model_comparison.csv."""
    import pandas as pd

    cmp_path = C.TABLES / "model_comparison.csv"
    ranked = (list(pd.read_csv(cmp_path).sort_values("OA", ascending=False)
                   ["model"]) if cmp_path.exists() else S.MODEL_ORDER)
    have = [m for m in ranked if raster_path(m).exists()]
    if requested:
        want = [m.strip() for m in requested.split(",")]
        unknown = [m for m in want if m not in have]
        if unknown:
            sys.exit(f"No raster for {unknown}. Available: {have}")
        return want
    if not have:
        sys.exit("No classified rasters in data/exports/. "
                 "Run `python src/classify_map.py` first.")
    return have


def areas_by_model():
    p = C.TABLES / "map_areas.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    out = {}
    for model, block in (raw.get("models") or {}).items():
        a = block.get("areas_km2", block)
        out[model] = {int(k): float(v) for k, v in a.items()}
    return out


# --------------------------------------------------------------------------


def fig_panel(models, boundary):
    """Six maps in a 2 x 3 grid, one shared legend beneath.

    Each panel is titled with the model and its hold-out OA, because the
    point of the figure is that maps this visibly different come from models
    whose accuracies are statistically indistinguishable.
    """
    import pandas as pd

    cmp_path = C.TABLES / "model_comparison.csv"
    oa = (pd.read_csv(cmp_path).set_index("model")["OA"].to_dict()
          if cmp_path.exists() else {})
    areas = areas_by_model()

    cmap, norm = M.class_cmap()
    ncol = 3
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(S.DOUBLE, S.DOUBLE * 0.78))
    axes = np.atleast_1d(axes).ravel()

    for ax, model in zip(axes, models):
        arr, extent, _ = M.read_class_raster(raster_path(model))
        ax.imshow(arr, extent=extent, cmap=cmap, norm=norm,
                  interpolation="nearest")
        M.tidy_map_axes(ax, boundary)
        title = model if model not in oa else f"{model}   OA {oa[model]:.3f}"
        ax.set_title(title, pad=2)
        if model in areas:
            built = areas[model].get(2)
            if built is not None:
                # Top right, not bottom right. The city narrows towards the
                # south-east, so the bottom-right corner is where the mapped
                # coastline and the scale bar both sit; the label was landing
                # on the map. The top-right corner is empty in every panel.
                ax.text(0.985, 0.985, f"Built-up {built:.1f} km$^2$",
                        transform=ax.transAxes, fontsize=6.2, va="top",
                        ha="right", color="#222222")
    for ax in axes[len(models):]:
        ax.axis("off")

    M.scalebar(axes[0], 5000, loc=(0.05, 0.05))
    # Dropped below the area label, which now owns the top-right corner.
    M.north_arrow(axes[0], loc=(0.93, 0.70))

    for i, ax in enumerate(axes[:len(models)]):
        S.panel_label(ax, f"({chr(97 + i)})")

    fig.legend(handles=S.class_legend_handles(), loc="lower center",
               ncol=5, bbox_to_anchor=(0.5, -0.005), handlelength=1.1,
               columnspacing=1.2, handletextpad=0.4)
    fig.subplots_adjust(wspace=0.03, hspace=0.10, bottom=0.06)
    return fig


def fig_best(model, boundary):
    """The leading model alone, with its class areas beside it.

    The bar chart sits in its own axes rather than as an inset over the map:
    Taipei's outline reaches every corner of its bounding box, so an inset
    would have to cover mapped pixels wherever it was put.
    """
    cmap, norm = M.class_cmap()
    arr, extent, _ = M.read_class_raster(raster_path(model))
    areas = areas_by_model().get(model, {})

    fig, (ax, bar) = plt.subplots(
        1, 2, figsize=(S.ONE_HALF, S.ONE_HALF * 0.80),
        gridspec_kw=dict(width_ratios=[1, 0.55], wspace=0.28))

    ax.imshow(arr, extent=extent, cmap=cmap, norm=norm,
              interpolation="nearest")
    M.tidy_map_axes(ax, boundary)
    M.scalebar(ax, 5000, loc=(0.04, 0.04))
    M.north_arrow(ax, loc=(0.92, 0.85))

    if areas:
        ids = [i for i in S.CLASS_IDS if i in areas]
        vals = [areas[i] for i in ids]
        bar.barh(range(len(ids)), vals,
                 color=[S.CLASS_COLORS[i] for i in ids],
                 edgecolor="#333333", lw=0.3, height=0.7)
        bar.set_yticks(range(len(ids)))
        bar.set_yticklabels([S.CLASS_NAMES[i] for i in ids])
        bar.invert_yaxis()
        bar.set_xlabel("mapped area (km$^2$)")
        for k, v in enumerate(vals):
            bar.text(v, k, f" {v:.1f}", va="center", fontsize=6.2)
        bar.set_xlim(0, max(vals) * 1.30)
        bar.grid(axis="x", color=S.LIGHT, lw=0.4)
        bar.set_axisbelow(True)
        bar.set_title(f"total {sum(vals):.1f} km$^2$", fontsize=6.2, pad=3)
    else:
        bar.axis("off")

    S.panel_label(ax, "(a)")
    S.panel_label(bar, "(b)")
    fig.legend(handles=S.class_legend_handles(), loc="lower center",
               ncol=5, bbox_to_anchor=(0.5, -0.02), handlelength=1.1,
               columnspacing=1.0, handletextpad=0.4)
    fig.suptitle(f"{model}", fontsize=7, y=0.99)
    return fig


def fig_agreement(models, boundary):
    """How many of the six models assign a pixel its modal class.

    This is the spatial counterpart of the area-range table: a pixel at 6/6
    is unambiguous, one at 3/6 or below is a place where the choice of
    classifier decides the answer. Because the maps disagree on area far more
    than their accuracies differ, the disagreement must be spatially
    structured rather than scattered noise, and this figure shows where.
    """
    stack, extent = [], None
    for model in models:
        arr, extent, _ = M.read_class_raster(raster_path(model))
        stack.append(arr.filled(0))
    stack = np.stack(stack)          # (n_models, rows, cols)
    valid = (stack > 0).all(0)

    # Modal class and its count, computed with bincount over the class axis.
    counts = np.zeros((len(S.CLASS_IDS) + 1,) + stack.shape[1:], dtype="uint8")
    for cid in S.CLASS_IDS:
        counts[cid] = (stack == cid).sum(0)
    n_agree = counts.max(0)
    modal = counts.argmax(0)
    # argmax breaks ties toward the lowest class id, which is not neutral here:
    # a few per cent of the city is tied, and since Water is id 1 and Swamp is
    # id 5, every tie between those two would be handed to Water - exactly the
    # pair the models disagree about. Tied pixels are identified and held out
    # of the per-class breakdown rather than silently attributed to one side.
    tied = (counts[S.CLASS_IDS] == n_agree).sum(0) > 1

    agree = np.ma.masked_where(~valid, n_agree)
    n = len(models)

    fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, S.DOUBLE * 0.46),
                             gridspec_kw=dict(width_ratios=[1, 1.15]))

    from matplotlib.colors import ListedColormap, BoundaryNorm
    ramp = ["#67001f", "#d6604d", "#f4a582", "#d1e5f0", "#4393c3", "#053061"]
    ramp = ramp[-n:] if n <= len(ramp) else ramp
    cmap = ListedColormap(ramp)
    cmap.set_bad(alpha=0)
    # The floor is whatever actually occurs, not n - len(ramp) + 1: with six
    # models over five classes the pigeonhole principle puts at least two on
    # the modal class, so a "1/6" level would be a legend entry for an empty
    # set.
    lo = int(n_agree[valid].min()) if valid.any() else 1
    ramp = ramp[-(n - lo + 1):]
    cmap = ListedColormap(ramp)
    cmap.set_bad(alpha=0)
    norm = BoundaryNorm(np.arange(lo - 0.5, n + 1), cmap.N)

    im = axes[0].imshow(agree, extent=extent, cmap=cmap, norm=norm,
                        interpolation="nearest")
    M.tidy_map_axes(axes[0], boundary)
    M.scalebar(axes[0], 5000, loc=(0.05, 0.05))
    M.north_arrow(axes[0])
    cb = fig.colorbar(im, ax=axes[0], fraction=0.030, pad=0.02, shrink=0.75,
                      ticks=np.arange(lo, n + 1))
    cb.set_label(f"models agreeing (of {n})", fontsize=6.2)
    cb.ax.tick_params(labelsize=6.2, length=1.5)
    cb.outline.set_linewidth(0.4)

    # Right panel: for each class, the share of its modal-class pixels at
    # each agreement level. Classes whose bars sit low are the ones whose
    # mapped area is really a modelling choice.
    rows = []
    unique = valid & ~tied
    for cid in S.CLASS_IDS:
        m = unique & (modal == cid)
        tot = int(m.sum())
        if not tot:
            continue
        for k in range(lo, n + 1):
            rows.append((cid, k, float(((n_agree == k) & m).sum()) / tot))
    pct_tied = 100.0 * tied[valid].sum() / max(valid.sum(), 1)

    import pandas as pd
    d = pd.DataFrame(rows, columns=["cid", "k", "share"])
    piv = d.pivot(index="cid", columns="k", values="share").fillna(0)
    left = np.zeros(len(piv))
    ax = axes[1]
    for j, k in enumerate(piv.columns):
        ax.barh(range(len(piv)), piv[k], left=left, height=0.68,
                color=ramp[j], edgecolor="white", lw=0.3,
                label=f"{k}/{n}")
        left = left + piv[k].to_numpy()
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([S.CLASS_NAMES[i] for i in piv.index])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the class's mapped pixels")
    # Below the axis, not as a title: panel_label() puts a bold "(b)" at the
    # top-left corner of these same axes and a left-aligned title runs into it.
    ax.text(0.0, -0.145, f"a further {pct_tied:.1f}% of the city has no single "
            "modal class and is excluded here", transform=ax.transAxes,
            fontsize=6.2, color="#555555", ha="left", va="top")
    ax.grid(axis="x", color=S.LIGHT, lw=0.4)
    ax.set_axisbelow(True)
    ax.legend(title="agreement", ncol=1, fontsize=6.2, title_fontsize=6.2,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    S.panel_label(axes[0], "(a)")
    S.panel_label(axes[1], "(b)")
    fig.subplots_adjust(wspace=0.58)
    return fig


def fig_study_area(boundary):
    """Sentinel-2 true colour with the city outline - the study-area panel.

    Not the submitted Figure 1. The published location map
    (fig_study_area_published.jpg, carried over from the first submission)
    stays as Figure 1: it carries the hillshade, the road network and the
    Taiwan locator inset, none of which are reproducible from the four bands
    exported here. This function is kept as a quick check that the composite
    downloaded correctly and is written under a name that cannot be mistaken
    for the manuscript figure.
    """
    comp = C.EXPORTS / "s2_composite_2020.tif"
    if not comp.exists():
        print("  skipped composite check: s2_composite_2020.tif not found")
        return None
    rgb, extent = M.read_rgb_composite(comp)

    fig, ax = plt.subplots(figsize=(S.ONE_HALF, S.ONE_HALF * 0.92))
    ax.imshow(rgb, extent=extent, interpolation="nearest")
    M.tidy_map_axes(ax, boundary, bcolor="#ffe100", blw=0.9)
    M.scalebar(ax, 5000, loc=(0.04, 0.04), color="white")
    M.north_arrow(ax, loc=(0.93, 0.85), color="white")
    # Caption below the axes, not over the image: the city outline reaches
    # the corners of its own bounding box, so any white text placed inside
    # would sit half over imagery and half over blank page.
    ax.set_xlabel(
        "Sentinel-2 MSI Level-1C top-of-atmosphere reflectance, median composite "
        f"{C.DATE_START} to {C.DATE_END}, cloud cover < {C.MAX_CLOUD_PCT}%\n"
        f"R/G/B = B4/B3/B2, {C.SCALE} m, {C.EXPORT_CRS}. "
        "Yellow line: Taipei City administrative boundary.",
        fontsize=6.2, labelpad=4)
    return fig


# --------------------------------------------------------------------------


def main(argv=None):
    # argv defaults to [] rather than to sys.argv so that make_figures.py can
    # import this module and call main() without its own group arguments being
    # parsed here and rejected.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", help="comma-separated subset to draw")
    ap.add_argument("--no-agreement", action="store_true",
                    help="skip the agreement figure (it is the slow one)")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    S.use()
    models = available_models(args.models)
    print(f"Drawing {len(models)} model(s): {', '.join(models)}")
    boundary = M.load_boundary()
    if boundary is None:
        print("  note: no city boundary found in data/raw/, outline omitted")

    print("composite_check (not a manuscript figure)")
    f = fig_study_area(boundary)
    if f is not None:
        S.save(f, "composite_check", dpi=S.DPI_RASTER)

    print("fig_lulc_panel")
    S.save(fig_panel(models, boundary), "fig_lulc_panel", dpi=S.DPI_RASTER)

    print(f"fig_lulc_best ({models[0]})")
    S.save(fig_best(models[0], boundary), "fig_lulc_best", dpi=S.DPI_RASTER)

    if not args.no_agreement and len(models) > 1:
        print("fig_model_agreement")
        S.save(fig_agreement(models, boundary), "fig_model_agreement",
               dpi=S.DPI_RASTER)


if __name__ == "__main__":
    main()
