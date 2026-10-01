"""Where the errors are, and why they are not random.

Produces
    fig_error_concentration  recurring confusions, concentration in single
                             polygons, and the class-pair axes those errors
                             lie along
    fig_separability         Jeffries-Matusita distance between class pairs

The four panels answer the editor's question about misclassification with
something more specific than a confusion matrix: the same two class pairs
account for nearly all of it, the errors sit inside a handful of training
polygons rather than scattering, and those polygons lie on the axis between
the two class centres in feature space - the signature of overlapping class
definitions rather than of mislabelled ground truth.

    python src/figures/fig_error_concentration.py
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

NAME_TO_ID = {v: k for k, v in C.CLASS_NAMES.items()}


def centroid_axis(df, name_a, name_b, feats):
    """Project every pixel of class a onto the a -> b centroid axis.

    t = ((P - c_a) . v) / (v . v)  with  v = c_b - c_a

    t = 0 is the centre of the labelled class, t = 1 the centre of the class
    it is confused with. A training polygon whose pixels sit near t = 1 is
    spectrally indistinguishable from the other class, whatever its label
    says. Features are z-scored first so that no band dominates the axis
    purely through its range.
    """
    X = df[feats].to_numpy(dtype="float64")
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    # The sample CSV carries the polygon's own free-text label ('Forest',
    # 'Builtup'); the canonical names live in config, keyed by class id. Go
    # through the id so the two spellings can never drift apart.
    lab = df[C.CLASS_PROPERTY].map(C.CLASS_NAMES).to_numpy()
    ca, cb = X[lab == name_a].mean(0), X[lab == name_b].mean(0)
    v = cb - ca
    t = ((X - ca) @ v) / (v @ v)
    return pd.DataFrame({"t": t, "Class": lab,
                         "poly_id": df["poly_id"].to_numpy()})


def panel_confusions(ax):
    p = C.TABLES / "confusion_pairs.csv"
    S.require(p)
    # confusion_pairs.csv is ordered by median pixel count, but the bars show
    # the percentage share. Re-sort so the longest bar is the top row -
    # otherwise the second bar is visibly longer than the first.
    d = (pd.read_csv(p).sort_values("pct_of_class_median", ascending=False)
         .head(8).iloc[::-1])
    y = np.arange(len(d))
    colors = [C.CLASS_COLORS.get(NAME_TO_ID.get(t, 0), "#888888")
              for t in d["true"]]
    ax.barh(y, d["pct_of_class_median"], color=colors, edgecolor="#333333",
            lw=0.3, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a} $\\rightarrow$ {b}" for a, b in
                        zip(d["true"], d["predicted"])], fontsize=6.2)
    n_models = len([m for m in S.MODEL_ORDER])
    for i, (v, n, k) in enumerate(zip(d["pct_of_class_median"],
                                      d["median_n_px"],
                                      d["n_models_affected"])):
        ax.text(v, i, f"  {n:.0f} px, {k}/{n_models} models", fontsize=6.2,
                va="center", color="#333333")
    ax.set_xlim(0, d["pct_of_class_median"].max() * 1.85)
    ax.set_xlabel("median share of the true class misassigned (%)")
    S.panel_title(ax, "a", "recurring confusions")
    ax.grid(axis="x", color=S.LIGHT, lw=0.4)
    ax.set_axisbelow(True)


def panel_concentration(ax):
    p = C.TABLES / "error_concentration.csv"
    S.require(p)
    d = pd.read_csv(p)
    classes = [C.CLASS_NAMES[i] for i in sorted(C.CLASS_NAMES)
               if C.CLASS_NAMES[i] in set(d["class"])]
    rng = np.random.default_rng(1)
    for i, cls in enumerate(classes):
        sub = d[d["class"] == cls]
        # "Share of the class's errors" is undefined for a model that made no
        # errors in that class - the table stores 0.0 as a filler, but plotting
        # it would say the errors were perfectly spread out when in fact there
        # were none. Those models are excluded from the points and from the
        # median, and counted in the annotation instead. Without this,
        # Plantation (5 of 6 models error-free) plots a median of 0.00 against
        # a true value of 1.00 and reads as the least concentrated class.
        has_err = sub[sub["n_errors"] > 0]
        jitter = rng.uniform(-0.16, 0.16, len(has_err))
        for (_, r), j in zip(has_err.iterrows(), jitter):
            ax.plot(i + j, r["share_in_worst_polygon"], "o", ms=3.0,
                    color=S.MODEL_COLORS.get(r["model"], "#666666"),
                    mec="white", mew=0.3, alpha=0.95, zorder=3)
        if len(has_err):
            med = has_err["share_in_worst_polygon"].median()
            ax.plot([i - 0.3, i + 0.3], [med, med], color="#333333", lw=0.8,
                    zorder=4)
        n_zero = len(sub) - len(has_err)
        # Two short lines, not one long one: at five classes across a
        # half-width panel, anything wider than about eight characters runs
        # into the neighbouring class's annotation.
        note = (f"{int(sub['n_val_polygons'].iloc[0])} polys\n"
                f"{len(has_err)}/{len(sub)} err")
        ax.text(i, 1.08, note, ha="center", fontsize=6.2, color="#555555",
                linespacing=1.35)
        if n_zero == len(sub):
            ax.text(i, 0.5, "no errors", ha="center", fontsize=6.2,
                    color="#999999", rotation=90, va="center")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=18, ha="right")
    ax.set_ylim(0, 1.32)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("share of the class's errors\nin its single worst polygon")
    S.panel_title(ax, "b", "errors concentrate in few polygons")
    ax.grid(axis="y", color=S.LIGHT, lw=0.4)
    ax.set_axisbelow(True)
    for i, c in enumerate(sorted(C.CLASS_NAMES)):
        if C.CLASS_NAMES[c] in classes:
            ax.axvspan(classes.index(C.CLASS_NAMES[c]) - 0.45,
                       classes.index(C.CLASS_NAMES[c]) + 0.45,
                       color=C.CLASS_COLORS[c], alpha=0.07, lw=0, zorder=0)


def panel_axis(ax, df, feats, name_a, name_b, letter):
    """Per-polygon position on the a -> b centroid axis."""
    t = centroid_axis(df, name_a, name_b, feats)
    rng = np.random.default_rng(2)
    rows = []
    for cls, col in ((name_a, C.CLASS_COLORS[NAME_TO_ID[name_a]]),
                     (name_b, C.CLASS_COLORS[NAME_TO_ID[name_b]])):
        g = t[t["Class"] == cls].groupby("poly_id")
        med = g["t"].median()
        n = g["t"].size()
        yy = (0 if cls == name_a else 1) + rng.uniform(-0.17, 0.17, len(med))
        ax.scatter(med, yy, s=np.clip(n / 6.0, 2, 26), color=col,
                   edgecolor="#333333", linewidth=0.25, alpha=0.9, zorder=3)
        rows.append((cls, med, n, yy))

    # Flag polygons of class a that sit past the halfway point: their pixels
    # are closer to the other class centre than to their own.
    cls, med, n, yy = rows[0]
    over = med[med > 0.5]
    for pid, val in over.items():
        ax.annotate(f"poly {pid}", (val, yy[list(med.index).index(pid)]),
                    textcoords="offset points", xytext=(0, 6), fontsize=6.2,
                    ha="center", color="#b2182b")
    ax.axvline(0.5, color="#b2182b", lw=0.6, ls="--")
    ax.text(0.5, 1.62, " halfway", fontsize=6.2, color="#b2182b", va="top")
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"labelled\n{name_a}", f"labelled\n{name_b}"],
                       fontsize=6.2)
    ax.set_ylim(-0.55, 1.75)
    # The full "position on the X -> Y axis" wording overflows a half-width
    # panel and collides with the neighbouring one; the axis is named in the
    # panel title, so the label only has to explain the units.
    ax.set_xlabel("0 = own class centre,  1 = other class centre")
    S.panel_title(ax, letter, f"{name_a} vs {name_b}")
    ax.grid(axis="x", color=S.LIGHT, lw=0.4)
    ax.set_axisbelow(True)
    return len(over)


def fig_errors():
    S.require(C.SAMPLE_CSV)
    df = pd.read_csv(C.SAMPLE_CSV)
    feats = [b for b in C.BANDS_PUBLISHED if b in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(S.DOUBLE, S.DOUBLE * 0.62),
                             gridspec_kw=dict(wspace=0.30, hspace=0.55))
    panel_confusions(axes[0, 0])
    panel_concentration(axes[0, 1])
    panel_axis(axes[1, 0], df, feats, "Forest/Vegetation", "Plantation", "c")
    panel_axis(axes[1, 1], df, feats, "Swamp", "Water", "d")

    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=2.8,
                                   color=S.MODEL_COLORS[m], label=m)
                        for m in S.MODEL_ORDER],
               ncol=6, fontsize=6.2, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), handletextpad=0.25,
               columnspacing=1.2, title="panel (b) markers",
               title_fontsize=6.2)
    return fig


def fig_separability():
    """Jeffries-Matusita distance for every class pair.

    JM is bounded at 2; pairs below about 1.9 are conventionally read as not
    reliably separable with the given feature set. Only one pair falls short,
    which is why the paper's misclassification is concentrated rather than
    spread across the matrix.
    """
    p = C.TABLES / "separability_jm.csv"
    S.require(p)
    d = pd.read_csv(p)
    ids = sorted(C.CLASS_NAMES)
    n = len(ids)
    idx = {c: i for i, c in enumerate(ids)}
    g = np.full((n, n), np.nan)
    for _, r in d.iterrows():
        i, j = idx[int(r.class_a)], idx[int(r.class_b)]
        g[i, j] = g[j, i] = r.jeffries_matusita

    # Drawn at exactly one column width. The earlier 1.25x version had to be
    # scaled to 0.8 to fit the column, which took its 5 pt cell labels below
    # 4 pt on the page.
    fig, ax = plt.subplots(figsize=(S.SINGLE, S.SINGLE * 0.90))
    im = ax.imshow(np.ma.masked_invalid(g), cmap="viridis", vmin=1.8, vmax=2.0)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "-", ha="center", va="center", fontsize=6.2,
                        color="#888888")
                continue
            v = g[i, j]
            r_, g_, b_, _ = im.cmap(im.norm(v))
            lum = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.2,
                    color="black" if lum > 0.55 else "white")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([C.CLASS_NAMES[c] for c in ids], rotation=30,
                       ha="right")
    ax.set_yticklabels([C.CLASS_NAMES[c] for c in ids])
    # Colour the tick labels with the map palette so the reader can carry the
    # classes across from the map figures without a legend.
    for lab, c in zip(ax.get_xticklabels(), ids):
        lab.set_color(C.CLASS_COLORS[c])
    for lab, c in zip(ax.get_yticklabels(), ids):
        lab.set_color(C.CLASS_COLORS[c])
    ax.set_xticks(np.arange(-.5, n), minor=True)
    ax.set_yticks(np.arange(-.5, n), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("Jeffries-Matusita distance (max 2)", fontsize=6.2)
    cb.ax.tick_params(labelsize=6.2, length=1.5)
    cb.outline.set_linewidth(0.4)
    worst = d.sort_values("jeffries_matusita").iloc[0]
    ax.set_title(f"only {worst.name_a} vs {worst.name_b} falls short "
                 f"(JM = {worst.jeffries_matusita:.3f})", pad=4, fontsize=6.5)
    # The rotated tick labels overhang the axes, and S.save() crops to the
    # ink, so without this the saved PDF comes out wider than the figure and
    # LaTeX has to scale it down again. tight_layout pulls the overhang back
    # inside, which keeps the file at one column width and the text at the
    # size it was drawn.
    fig.tight_layout(pad=0.25)
    return fig


def main():
    S.use()
    print("fig_error_concentration")
    S.save(fig_errors(), "fig_error_concentration")
    print("fig_separability")
    S.save(fig_separability(), "fig_separability")


if __name__ == "__main__":
    main()
