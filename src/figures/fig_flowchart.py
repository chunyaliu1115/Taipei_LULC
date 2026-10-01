"""Workflow and classification-scheme diagrams.

Produces
    fig_workflow              end-to-end methodology flowchart
    fig_classification_scheme feature vector, grouped CV, six classifiers

Counts in the boxes are read from the analysis outputs rather than typed in,
so the diagram cannot drift away from the numbers in the tables after a
re-run.

    python src/figures/fig_flowchart.py
"""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import style as S  # noqa: E402
import flowtools as F  # noqa: E402

BLUE = "#dbe7f3"


def facts():
    """Pull the numbers the diagram quotes out of the pipeline outputs."""
    f = dict(n_px=None, n_poly=None, n_train=None, n_val=None,
             poly_train=None, poly_val=None, models=S.MODEL_ORDER,
             n_ref=None)
    if C.SAMPLE_CSV.exists():
        d = pd.read_csv(C.SAMPLE_CSV)
        f["n_px"] = len(d)
        f["n_poly"] = d["poly_id"].nunique()
        vc = d["split"].value_counts()
        f["n_train"], f["n_val"] = int(vc.get("train", 0)), int(vc.get("val", 0))
        pv = d.groupby("split")["poly_id"].nunique()
        f["poly_train"], f["poly_val"] = int(pv.get("train", 0)), int(pv.get("val", 0))
    cmp_path = C.TABLES / "model_comparison.csv"
    if cmp_path.exists():
        f["models"] = list(pd.read_csv(cmp_path)
                           .sort_values("OA", ascending=False)["model"])
    ref = C.SAMPLES / "reference_points_blank.csv"
    if ref.exists():
        f["n_ref"] = len(pd.read_csv(ref))
    return f


# --------------------------------------------------------------------------


def fig_scheme(f):
    """How one pixel becomes one label, and where the grouping enters."""
    fig, ax = F.canvas((S.DOUBLE, S.DOUBLE * 0.42))

    # (a) feature vector
    ax.text(0.005, 0.97, "(a) per-pixel feature vector", fontsize=7,
            fontweight="bold", va="top")
    bands = C.BANDS_PUBLISHED
    x0, w = 0.03, 0.052
    for i, b in enumerate(bands):
        F.box(ax, x0 + i * (w + 0.012) + w / 2, 0.72, w, 0.13, b,
              face="#dbe7f3")
    ax.text(x0 + len(bands) * (w + 0.012) - 0.006, 0.72,
            "  reflectance, scaled to 0-1", fontsize=6.2, va="center")
    ax.text(x0, 0.60, f"one row per labelled pixel; label from the polygon's "
            f"'{C.CLASS_PROPERTY}' property", fontsize=6.2, va="center")

    # (b) grouped split
    ax.text(0.005, 0.50, "(b) split and cross-validation are grouped by "
            "polygon", fontsize=7, fontweight="bold", va="top")
    ax.text(0.03, 0.40,
            "Pixels inside one training polygon are near-duplicates. A "
            "pixel-level split puts copies of\nthe same observation on both "
            "sides of the divide, so the hold-out score measures memory\n"
            "rather than generalisation. Every split here - the "
            f"{100 * (1 - C.TEST_SIZE):.0f}/{100 * C.TEST_SIZE:.0f} hold-out "
            f"and the {C.CV_FOLDS} tuning\nfolds - keeps a polygon whole.",
            fontsize=6.2, va="top", linespacing=1.5)

    # Polygon-vs-pixel cartoon. The axes are 0-1 in both directions on a
    # non-square figure, so box width is scaled by the figure aspect to keep
    # the polygons looking like polygons rather than letterboxes.
    import numpy as np
    aspect = fig.get_figheight() / fig.get_figwidth()
    bw, bh = 0.11 * aspect, 0.075
    rng = np.random.default_rng(0)
    for cx, tag, col in [(0.640, "train", "#4c78a8"),
                         (0.640 + bw + 0.055, "val", "#e45756")]:
        for j in range(3):
            cy = 0.44 - j * (bh + 0.018)
            ax.add_patch(plt.Rectangle((cx, cy - bh / 2), bw, bh,
                                       facecolor="none", edgecolor=col,
                                       lw=0.6, zorder=2))
            px = rng.uniform(cx + 0.1 * bw, cx + 0.9 * bw, 9)
            py = rng.uniform(cy - 0.4 * bh, cy + 0.4 * bh, 9)
            ax.scatter(px, py, s=1.4, color=col, zorder=3)
        ax.text(cx + bw / 2, 0.495, tag, ha="center", fontsize=6.2, color=col)
    ax.text(0.640, 0.205, "whole polygons go to one side; dots are pixels",
            fontsize=6.2, va="center")

    # (c) models
    ax.text(0.005, 0.235, "(c) classifiers compared", fontsize=7,
            fontweight="bold", va="top")
    n = len(f["models"])
    bw = 0.90 / n
    for i, m in enumerate(f["models"]):
        F.box(ax, 0.045 + bw / 2 + i * bw, 0.115, bw * 0.88, 0.10, m,
              face=S.MODEL_COLORS.get(m, "#eeeeee") + "44",
              edge=S.MODEL_COLORS.get(m, "#444444"), title_size=6.2)
    ax.text(0.045, 0.030, "identical features, identical split, identical "
            "tuning budget - the only difference between the six maps is the "
            "decision rule", fontsize=6.2, va="center")
    return fig


import matplotlib.pyplot as plt  # noqa: E402  (used inside fig_scheme)


def main():
    S.use()
    f = facts()
    # The workflow diagram is drawn by Graphviz, which also writes the
    # editable draw.io .xml. The earlier hand-placed matplotlib version was
    # removed: two sources for one figure is how a diagram and its caption
    # drift apart.
    import fig_workflow_gv as GV
    GV.main()
    print("fig_classification_scheme")
    S.save(fig_scheme(f), "fig_classification_scheme")


if __name__ == "__main__":
    main()
