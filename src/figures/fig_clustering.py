"""Why the model comparison has to be cluster-aware.

Produces
    fig_clustering   three panels: spectral near-duplication inside polygons,
                     the effective sample size that follows from it, and the
                     false-positive rate of the two McNemar variants

    python src/figures/fig_clustering.py

This figure carries the methodological claim the whole paper rests on. The
argument runs in three steps and the panels are those steps:

(a) Pixels drawn from one digitised polygon are near-duplicates of one
    another. Measured, not asserted: the spectral distance between two pixels
    of the same class is several times larger when they come from different
    polygons than when they come from the same one.
(b) So 1,829 validation pixels are not 1,829 independent observations. The
    Kish design effect converts them to the tens.
(c) So a test that assumes independence over-rejects. Under a null with two
    equally accurate models, the naive McNemar test's false-positive rate
    climbs with the intra-cluster correlation while the clustered test stays
    at or below nominal.

Panel (c) reads the simulation written by src/simulate_clustered_null.py
rather than re-running it, so the figure and the number quoted in the text
cannot disagree.
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

SIM_CSV = C.TABLES / "clustered_null_simulation.csv"
OLOFSSON_CSV = C.TABLES / "olofsson_overall.csv"
SAMPLE_CSV = C.SAMPLE_CSV

# Pairs sampled per class for panel (a). Large enough that the medians are
# stable to the third decimal, small enough to stay instant.
N_PAIRS = 20000
SEED = 0


def spectral_pairs(bands=None):
    """Distances between same-class pixel pairs, split by same/other polygon.

    The same-class restriction is what makes the comparison fair. Comparing
    any two pixels would mostly measure the distance between classes, which is
    not the question; holding class fixed isolates the polygon effect, which
    is.
    """
    d = pd.read_csv(SAMPLE_CSV)
    bands = bands or list(C.BANDS_PUBLISHED)
    X = d[bands].to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)

    within, between = [], []
    for _, g in d.groupby("Class"):
        idx = g.index.to_numpy()
        pid = g["poly_id"].to_numpy()
        if len(idx) < 2 or len(np.unique(pid)) < 2:
            continue
        i = rng.integers(0, len(idx), N_PAIRS)
        j = rng.integers(0, len(idx), N_PAIRS)
        keep = i != j
        i, j = i[keep], j[keep]
        dist = np.linalg.norm(X[idx[i]] - X[idx[j]], axis=1)
        same = pid[i] == pid[j]
        within.append(dist[same])
        between.append(dist[~same])

    return np.concatenate(within), np.concatenate(between)


def panel_a(ax):
    """Spectral distance within a polygon versus between polygons."""
    within, between = spectral_pairs()

    # Log x: the two distributions differ by a factor, not an offset, and on a
    # linear axis the within-polygon mass collapses onto the origin.
    lo = max(1e-4, min(within.min(), between.min()))
    hi = max(within.max(), between.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), 60)

    for v, lab, col in ((within, "same polygon", "#4c78a8"),
                        (between, "different polygon,\nsame class", "#e45756")):
        ax.hist(v, bins=bins, density=True, histtype="stepfilled",
                alpha=0.45, color=col, lw=0.7, edgecolor=col, label=lab)
        ax.axvline(np.median(v), color=col, lw=1.0, ls="--")

    ratio = np.median(between) / np.median(within)
    ax.set_xscale("log")
    ax.set_xlabel("Euclidean distance in 4-band reflectance space")
    ax.set_ylabel("density")
    ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=6.2, handlelength=1.2)
    ax.text(0.97, 0.60,
            f"median distance is\n{ratio:.1f}$\\times$ larger between\npolygons",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
            color=S.GREY)
    S.panel_title(ax, "a", "pixels inside one polygon are near-duplicates",
                  fontsize=7.5)
    return ratio


def panel_b(ax):
    """1,829 pixels, but only tens of effective observations."""
    ol = pd.read_csv(OLOFSSON_CSV).set_index("model")
    models = [m for m in S.MODEL_ORDER if m in ol.index]
    y = np.arange(len(models))
    n_eff = ol.loc[models, "n_eff_total"].to_numpy()
    n_tot = float(ol["n_total"].iloc[0])

    ax.barh(y, [n_tot] * len(models), color=S.LIGHT, height=0.66,
            label="pixels counted")
    ax.barh(y, n_eff, color=[S.MODEL_COLORS[m] for m in models], height=0.66,
            label="effective observations")

    for yi, m, ne in zip(y, models, n_eff):
        ax.text(ne + 28, yi, f"{ne:.0f}  (deff {n_tot / ne:.0f})",
                va="center", fontsize=6.5, color=S.GREY)

    ax.set_yticks(y)
    ax.set_yticklabels([S.MODEL_SHORT[m] for m in models], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(0, n_tot * 1.02)
    ax.set_xlabel("observations in the hold-out set")
    # Every row is a full-width grey bar, so an unframed legend anywhere in
    # the axes sits on top of one. A white frame is the cheap way out.
    ax.legend(loc="lower right", fontsize=6.2, handlelength=1.2, frameon=True,
              framealpha=0.92, edgecolor="none", borderpad=0.4)
    S.panel_title(ax, "b", f"{n_tot:,.0f} pixels, tens of effective "
                  f"observations", fontsize=7.5)


def panel_c(ax):
    """False-positive rate of both McNemar variants against the ICC."""
    sim = pd.read_csv(SIM_CSV)
    alpha = float(sim["alpha"].iloc[0])
    icc_lo = float(sim["icc_observed_min"].iloc[0])
    icc_hi = float(sim["icc_observed_max"].iloc[0])
    icc_med = float(sim["icc_observed_median"].iloc[0])

    ax.axhspan(0, 0, color="none")
    ax.axvspan(icc_lo, icc_hi, color="#f0f0f0", zorder=0)
    ax.text((icc_lo + icc_hi) / 2, 0.97, "ICC observed\nhere",
            ha="center", va="top", fontsize=6.2, color=S.GREY,
            transform=ax.get_xaxis_transform())

    for name, col, mark in (("naive", "#e45756", "o"),
                            ("clustered", "#4c78a8", "s")):
        s = sim[sim.test == name].sort_values("icc")
        ax.plot(s.icc, s.false_positive_rate, marker=mark, ms=2.6, lw=1.1,
                color=col, label=f"{name} McNemar", zorder=3)
        ax.fill_between(s.icc, s.ci_low, s.ci_high, color=col, alpha=0.18,
                        lw=0, zorder=2)

    ax.axhline(alpha, color=S.GREY, lw=0.8, ls=":", zorder=1)
    ax.text(0.985, alpha + 0.02, f"nominal $\\alpha$ = {alpha:g}", ha="right",
            va="bottom", fontsize=6.2, color=S.GREY,
            transform=ax.get_yaxis_transform())

    hit = sim[(sim.test == "naive") & np.isclose(sim.icc, icc_med)]
    if len(hit):
        r = float(hit.false_positive_rate.iloc[0])
        ax.annotate(f"{r:.0%}", xy=(icc_med, r), xytext=(icc_med - 0.30, r),
                    fontsize=6.5, color="#e45756", va="center",
                    arrowprops=dict(arrowstyle="->", color="#e45756", lw=0.7))

    ax.set_xlim(0, 0.92)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("intra-cluster correlation of correctness")
    ax.set_ylabel("false-positive rate")
    ax.legend(loc="center right", fontsize=6.2, handlelength=1.4)
    S.panel_title(ax, "c", "so the naive test invents significance",
                  fontsize=7.5)


def fig_clustering():
    fig, axes = plt.subplots(1, 3, figsize=(S.DOUBLE, S.DOUBLE * 0.31))
    ratio = panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])
    fig.subplots_adjust(wspace=0.30)
    fig.tight_layout(pad=0.4, w_pad=1.6)
    return fig, ratio


def main():
    S.use()
    S.require(SAMPLE_CSV, OLOFSSON_CSV, SIM_CSV)
    print("fig_clustering")
    fig, ratio = fig_clustering()
    print(f"  within/between spectral distance ratio: {ratio:.2f}x")
    S.save(fig, "fig_clustering")


if __name__ == "__main__":
    main()
