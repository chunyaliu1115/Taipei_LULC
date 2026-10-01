"""Between-model area uncertainty.

Produces
    fig_area_uncertainty  mapped-area spread, area estimates with CIs, and
                          the spread measured against those CIs

The argument: six classifiers whose accuracies cannot be told apart produce
class areas that differ by up to a factor of three (Swamp, 10.9 to 32.2 km2),
and for Built-up, Swamp and Water the between-model spread is wider than any
single model's confidence interval. An area figure quoted from one classifier,
with a CI from that same classifier, therefore understates the real
uncertainty for those three classes.

Panel (c) is the test, so it is worth being precise about what it compares.
Forest/Vegetation (0.5x) and Plantation (0.5x) fall the other way: for those
two the models agree with each other more closely than the sampling interval,
and a single-model area with its own CI is an honest summary. The figure is
not the claim that model choice always dominates - it is the claim that which
of the two dominates has to be checked class by class rather than assumed.

    python src/figures/fig_area_uncertainty.py
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


def load():
    per = C.TABLES / "olofsson_per_class.csv"
    bym = C.TABLES / "map_areas_by_model.csv"
    S.require(per, bym)
    d = pd.read_csv(per)
    a = pd.read_csv(bym)
    a = a[a["class"] != "TOTAL"].copy()
    return d, a


def fig_area(d, a):
    # Class-id order, not the CSV's alphabetical order, so the rows line up
    # with the legend of every map figure and with the other analysis panels.
    present = set(a["class"])
    classes = [C.CLASS_NAMES[i] for i in sorted(C.CLASS_NAMES)
               if C.CLASS_NAMES[i] in present]
    models = [m for m in S.MODEL_ORDER if m in a.columns]
    y = np.arange(len(classes))

    fig, axes = plt.subplots(1, 3, figsize=(S.DOUBLE, S.DOUBLE * 0.33),
                             gridspec_kw=dict(width_ratios=[1, 1, 0.85],
                                              wspace=0.42))

    # ---- (a) mapped area, one dot per model ------------------------------
    ax = axes[0]
    hi = 0.0
    for i, cls in enumerate(classes):
        vals = [float(a.loc[a["class"] == cls, m].iloc[0]) for m in models]
        hi = max(hi, max(vals))
        ax.plot([min(vals), max(vals)], [i, i], color=S.LIGHT, lw=3.2,
                solid_capstyle="round", zorder=1)
        for m, v in zip(models, vals):
            ax.plot(v, i, "o", ms=3.0, color=S.MODEL_COLORS[m], mec="white",
                    mew=0.4, zorder=3)
        # To the right of the dots, not above them. Placed above the row the
        # label sits inside the next row's band for the classes whose dots are
        # tightly bunched, and for the bottom row it lands on the dots.
        ax.text(max(vals), i, f"  {max(vals) / max(min(vals), 1e-9):.1f}x",
                fontsize=6.2, ha="left", va="center", color="#555555")
    ax.set_xlim(0, hi * 1.20)
    ax.set_yticks(y)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()
    ax.set_xlabel("mapped area (km$^2$)")
    S.panel_title(ax, "a", "area of the classified map")
    ax.grid(axis="x", color=S.LIGHT, lw=0.4)
    ax.set_axisbelow(True)

    # ---- (b) design-based area estimate with CI --------------------------
    ax = axes[1]
    off = np.linspace(-0.32, 0.32, len(models))
    for k, m in enumerate(models):
        sub = d[d.model == m].set_index("class")
        for i, cls in enumerate(classes):
            if cls not in sub.index:
                continue
            r = sub.loc[cls]
            ax.plot([r.area_CI_low, r.area_CI_high], [i + off[k]] * 2,
                    color=S.MODEL_COLORS[m], lw=0.8, zorder=2)
            ax.plot(r.area_estimate, i + off[k], "o", ms=2.4,
                    color=S.MODEL_COLORS[m], mec="white", mew=0.3, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("area estimate (km$^2$), 95% CI")
    S.panel_title(ax, "b", "design-based (Olofsson) area estimate")
    ax.grid(axis="x", color=S.LIGHT, lw=0.4)
    ax.set_axisbelow(True)
    for i in range(len(classes)):
        ax.axhline(i + 0.5, color="#eeeeee", lw=0.4, zorder=0)

    # ---- (c) spread against interval width -------------------------------
    # The comparison that matters: half the between-model range against the
    # median half-width of a single model's interval. A bar pair where the
    # left bar is taller means model choice, not sampling, dominates.
    ax = axes[2]
    spread, margin = [], []
    for cls in classes:
        vals = [float(a.loc[a["class"] == cls, m].iloc[0]) for m in models]
        spread.append((max(vals) - min(vals)) / 2)
        sub = d[d["class"] == cls]
        margin.append(float(sub["area_margin"].median()) if len(sub) else np.nan)
    h = 0.36
    ax.barh(y - h / 2, spread, height=h, color="#b2182b", edgecolor="none")
    ax.barh(y + h / 2, margin, height=h, color="#2166ac", edgecolor="none")
    # The two bars are labelled in place rather than in a legend box, which
    # would have to sit on top of the shorter bars. The label row is whichever
    # class has the widest pair - on a narrow class the words are clipped.
    j = int(np.nanargmax([min(s_, m_) for s_, m_ in zip(spread, margin)]))
    ax.text(0.35, j - h / 2, "between-model", fontsize=6.2, va="center",
            color="white")
    ax.text(0.35, j + h / 2, "sampling CI", fontsize=6.2, va="center",
            color="white")
    for i, (s_, m_) in enumerate(zip(spread, margin)):
        if np.isfinite(m_) and m_ > 0:
            ax.text(max(s_, m_) * 1.03, i, f"  {s_ / m_:.1f}x", fontsize=6.2,
                    va="center", color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    # The two series are defined precisely in the figure caption
    # (results/figures/FIGURE_CAPTIONS.md); the in-bar labels only have to be
    # enough to tell them apart at a glance.
    ax.set_xlabel("km$^2$")
    S.panel_title(ax, "c", "which uncertainty is larger")
    ax.grid(axis="x", color=S.LIGHT, lw=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(max(spread), np.nanmax(margin)) * 1.30)

    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=2.8,
                                   color=S.MODEL_COLORS[m], label=m)
                        for m in models],
               ncol=len(models), fontsize=6.2, loc="lower center",
               bbox_to_anchor=(0.5, -0.17), handletextpad=0.25,
               columnspacing=1.2)
    return fig


def main():
    S.use()
    d, a = load()
    print("fig_area_uncertainty")
    S.save(fig_area(d, a), "fig_area_uncertainty")


if __name__ == "__main__":
    main()
