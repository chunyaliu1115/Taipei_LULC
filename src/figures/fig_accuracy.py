"""Accuracy comparison and pairwise significance.

Produces
    fig_accuracy_ci   hold-out and area-weighted OA with confidence intervals
    fig_mcnemar       Holm-adjusted McNemar p-values, naive vs cluster-corrected
    fig_perclass_ua_pa per-class user's and producer's accuracy by model

The pairing of the two top figures is the argument of the revision: the
models' intervals overlap, and once the polygon clustering is accounted for
not one of the fifteen pairwise differences survives.

    python src/figures/fig_accuracy.py
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


def fig_accuracy_ci():
    """Two forest plots: hold-out OA and kappa, plus area-weighted OA.

    The hold-out interval is a bootstrap over validation pixels and therefore
    treats those pixels as independent, which they are not. The area-weighted
    interval uses the effective sample size after the polygon design effect,
    and is three to four times wider. Showing them side by side is the point:
    the apparent ranking of the models lives entirely inside the narrower
    interval that the design does not license.
    """
    cmp_path = C.TABLES / "model_comparison.csv"
    S.require(cmp_path)
    d = pd.read_csv(cmp_path).sort_values("OA")

    olof_path = C.TABLES / "olofsson_overall.csv"
    o = (pd.read_csv(olof_path).set_index("model")
         if olof_path.exists() else None)

    # Constrained layout rather than a fixed wspace: panel (c)'s left-aligned
    # title and its x-label are both wider than the third of the figure they
    # sit in, and with a fixed layout they ran off the right edge of the
    # canvas and were cut off in the saved file.
    fig, axes = plt.subplots(1, 3, figsize=(S.DOUBLE, S.DOUBLE * 0.30),
                             layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.01, wspace=0.04)
    y = np.arange(len(d))

    def forest(ax, lo, mid, hi, letter, title, xlabel):
        for i, (m, a, b, c) in enumerate(zip(d["model"], lo, mid, hi)):
            col = S.MODEL_COLORS.get(m, "#444444")
            ax.plot([a, c], [i, i], color=col, lw=1.4, solid_capstyle="butt",
                    zorder=2)
            ax.plot([a, a, np.nan, c, c], [i - .16, i + .16, np.nan,
                                           i - .16, i + .16],
                    color=col, lw=0.7, zorder=2)
            ax.plot(b, i, "o", ms=3.0, color=col, mec="white", mew=0.4,
                    zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(d["model"])
        ax.set_ylim(-0.6, len(d) - 0.4)
        ax.set_xlabel(xlabel)
        S.panel_title(ax, letter, title)
        ax.grid(axis="x", color=S.LIGHT, lw=0.4)
        ax.set_axisbelow(True)

    forest(axes[0], d["OA_CI_low"], d["OA"], d["OA_CI_high"], "a",
           "hold-out overall accuracy", "OA (bootstrap 95% CI)")
    forest(axes[1], d["kappa_CI_low"], d["kappa"], d["kappa_CI_high"], "b",
           "hold-out kappa", "kappa (bootstrap 95% CI)")

    ax = axes[2]
    if o is not None:
        oo = o.reindex(d["model"])
        for i, m in enumerate(d["model"]):
            r = oo.loc[m]
            col = S.MODEL_COLORS.get(m, "#444444")
            ax.plot([r["OA_CI_low"], min(r["OA_CI_high"], 1.0)], [i, i],
                    color=col, lw=1.4, zorder=2)
            ax.plot(r["OA"], i, "o", ms=3.0, color=col, mec="white", mew=0.4,
                    zorder=3)
            ax.text(1.004, i, f"n$_{{eff}}$ {r['n_eff_total']:.0f}",
                    fontsize=6.2, va="center")
        ax.set_yticks(y)
        ax.set_yticklabels(d["model"])
        ax.set_ylim(-0.6, len(d) - 0.4)
        # Kept short on purpose: at 7 pt a longer label is wider than the
        # third of the figure this panel occupies, and the overhang is
        # cropped away when the figure is saved. The design-effect
        # qualification lives in the caption instead.
        ax.set_xlabel("OA (95% CI on n$_{eff}$)")
        S.panel_title(ax, "c", "area-weighted overall accuracy")
        ax.axvline(1.0, color="#999999", lw=0.4, ls=":")
        ax.set_xlim(right=1.05)
        ax.grid(axis="x", color=S.LIGHT, lw=0.4)
        ax.set_axisbelow(True)
    else:
        ax.axis("off")

    return fig


def fig_mcnemar():
    """Holm-adjusted McNemar p-values: naive below, cluster-corrected above.

    Splitting the two corrections across the diagonal of one matrix makes the
    comparison unavoidable - the same pair of models sits at mirrored
    positions, so a dark cell below the diagonal with a pale cell above it is
    a difference that only the naive test believes in.
    """
    p = C.TABLES / "mcnemar_pairwise.csv"
    S.require(p)
    d = pd.read_csv(p)
    cmp_path = C.TABLES / "model_comparison.csv"
    order = (list(pd.read_csv(cmp_path).sort_values("OA", ascending=False)
                  ["model"]) if cmp_path.exists() else S.MODEL_ORDER)
    order = [m for m in order if m in set(d.model_A) | set(d.model_B)]
    idx = {m: i for i, m in enumerate(order)}
    n = len(order)

    grid = np.full((n, n), np.nan)
    for _, r in d.iterrows():
        i, j = idx[r.model_A], idx[r.model_B]
        lo, hi = (i, j) if i < j else (j, i)
        grid[hi, lo] = r["p_holm_naive"]        # lower triangle
        grid[lo, hi] = r["p_holm_clustered"]    # upper triangle

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(S.DOUBLE, S.DOUBLE * 0.36),
        gridspec_kw=dict(width_ratios=[1, 1.05], wspace=0.55))

    from matplotlib.colors import LogNorm
    m = np.ma.masked_invalid(grid)
    im = ax.imshow(m, cmap="RdYlBu", norm=LogNorm(vmin=1e-16, vmax=1.0))
    for i in range(n):
        for j in range(n):
            if np.isnan(grid[i, j]):
                continue
            v = grid[i, j]
            txt = "<0.001" if v < 1e-3 else f"{v:.3f}".lstrip("0")
            # RdYlBu is not monotone in lightness - its middle is pale yellow -
            # so the label colour comes from the cell's own luminance rather
            # than from the p-value.
            r, g, b, _ = im.cmap(im.norm(v))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.2,
                    color="black" if lum > 0.55 else "white")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short = [S.MODEL_SHORT.get(m, m) for m in order]
    ax.set_xticklabels(short, rotation=0)
    ax.set_yticklabels(short)
    ax.set_xticks(np.arange(-.5, n), minor=True)
    ax.set_yticks(np.arange(-.5, n), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    S.panel_title(ax, "a", "Holm-adjusted McNemar p\nlower: naive   "
                  "upper: polygon-cluster corrected")
    cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
    cb.set_label("p (log scale)", fontsize=6.2)
    cb.ax.tick_params(labelsize=6.2, length=1.5)
    cb.outline.set_linewidth(0.4)

    # Right panel: the same information as a paired dot plot, which makes the
    # size of the shift explicit rather than leaving it to a colour scale.
    d2 = d.sort_values("p_holm_naive").reset_index(drop=True)
    yy = np.arange(len(d2))
    ax2.hlines(yy, d2["p_holm_naive"].clip(lower=1e-17),
               d2["p_holm_clustered"], color=S.LIGHT, lw=0.8, zorder=1)
    ax2.plot(d2["p_holm_naive"].clip(lower=1e-17), yy, "o", ms=2.6,
             color="#b2182b", label="naive", zorder=2)
    ax2.plot(d2["p_holm_clustered"], yy, "o", ms=2.6, color="#2166ac",
             label="cluster-corrected", zorder=2)
    ax2.axvline(0.05, color="#444444", lw=0.6, ls="--")
    ax2.text(0.05, len(d2) - 0.2, " p = 0.05", fontsize=6.2, va="top")
    ax2.set_xscale("log")
    ax2.set_yticks(yy)
    ax2.set_yticklabels(
        [f"{S.MODEL_SHORT.get(a, a)} vs {S.MODEL_SHORT.get(b, b)}"
         for a, b in zip(d2.model_A, d2.model_B)], fontsize=6.2)
    ax2.set_ylim(-0.6, len(d2) - 0.4)
    ax2.set_xlabel("Holm-adjusted p")
    ax2.grid(axis="x", color=S.LIGHT, lw=0.4)
    ax2.set_axisbelow(True)
    ax2.legend(loc="lower left", fontsize=6.2)

    n_naive = int((d["p_holm_naive"] < 0.05).sum())
    n_clu = int((d["p_holm_clustered"] < 0.05).sum())
    S.panel_title(ax2, "b", f"{n_naive} of {len(d)} pairs significant "
                  f"naively, {n_clu} of {len(d)} after clustering")
    return fig


def fig_perclass():
    """User's and producer's accuracy per class, all models, with CIs."""
    p = C.TABLES / "olofsson_per_class.csv"
    S.require(p)
    d = pd.read_csv(p)
    order = [m for m in S.MODEL_ORDER if m in set(d.model)]
    classes = sorted(d.class_id.unique())

    fig, axes = plt.subplots(1, 2, figsize=(S.DOUBLE, S.DOUBLE * 0.30),
                             sharey=True, gridspec_kw=dict(wspace=0.06))
    off = np.linspace(-0.3, 0.3, len(order))

    for ax, (col, lo, hi, name) in zip(
            axes, [("UA", "UA_CI_low", "UA_CI_high", "user's accuracy"),
                   ("PA", "PA_CI_low", "PA_CI_high", "producer's accuracy")]):
        for k, m in enumerate(order):
            sub = d[d.model == m].set_index("class_id").reindex(classes)
            x = np.arange(len(classes)) + off[k]
            colr = S.MODEL_COLORS.get(m, "#444444")
            ax.errorbar(x, sub[col],
                        yerr=[np.clip(sub[col] - sub[lo], 0, None),
                              np.clip(sub[hi] - sub[col], 0, None)],
                        fmt="o", ms=2.4, lw=0.7, capsize=1.2, color=colr,
                        label=m if col == "UA" else None, zorder=3)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([C.CLASS_NAMES[c] for c in classes], rotation=18,
                           ha="right")
        ax.set_ylabel(name if col == "UA" else "")
        ax.set_ylim(0, 1.04)
        ax.grid(axis="y", color=S.LIGHT, lw=0.4)
        ax.set_axisbelow(True)
        # Class colour behind each group, so the reader keeps the map palette.
        for i, c in enumerate(classes):
            ax.axvspan(i - 0.45, i + 0.45, color=C.CLASS_COLORS[c], alpha=0.07,
                       lw=0, zorder=0)

    axes[0].legend(ncol=3, fontsize=6.2, loc="lower left",
                   handletextpad=0.3, columnspacing=0.8)
    S.panel_title(axes[0], "a", "user's accuracy")
    S.panel_title(axes[1], "b", "producer's accuracy")
    return fig


def main():
    S.use()
    print("fig_accuracy_ci")
    S.save(fig_accuracy_ci(), "fig_accuracy_ci")
    print("fig_mcnemar")
    S.save(fig_mcnemar(), "fig_mcnemar")
    print("fig_perclass_ua_pa")
    S.save(fig_perclass(), "fig_perclass_ua_pa")


if __name__ == "__main__":
    main()
