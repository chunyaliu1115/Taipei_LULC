"""Shared styling for every manuscript figure.

One place for the palette, the page geometry and the save routine, so that
figures drawn by different scripts are visually consistent and so that a
change of journal template is a one-file edit.

Elsevier artwork sizing (https://www.elsevier.com/artworkinstructions):
    single column   90 mm
    1.5 column     140 mm
    double column  190 mm
Line art is submitted at 600 dpi and combination/halftone at 300 dpi. Maps
carry a resampled raster so they go out at 300; everything else at 600.
"""

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

# --------------------------------------------------------------------------
# Page geometry
# --------------------------------------------------------------------------
MM = 1 / 25.4
SINGLE = 90 * MM      # inches
ONE_HALF = 140 * MM
DOUBLE = 190 * MM

DPI_LINE = 600        # vector-like content
DPI_RASTER = 300      # anything containing an imshow of a map

FIGDIR = C.FIGURES

# --------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------
# The class palette is NOT redefined here. It is read from config so the
# figures match the GEE script's `palette` argument exactly - the same colours
# the reviewers saw in the Code Editor and in the submitted maps.
CLASS_COLORS = C.CLASS_COLORS
CLASS_NAMES = C.CLASS_NAMES
CLASS_IDS = sorted(CLASS_NAMES)

# Model colours. Qualitative, colour-blind safe (Okabe-Ito), deliberately
# unrelated to the class palette so a model legend can never be mistaken for
# a class legend.
MODEL_ORDER = ["SVM (RBF)", "MLP (deep)", "Random Forest",
               "LightGBM", "XGBoost", "CART"]
MODEL_COLORS = {
    "SVM (RBF)": "#0072B2",
    "MLP (deep)": "#D55E00",
    "Random Forest": "#009E73",
    "LightGBM": "#CC79A7",
    "XGBoost": "#E69F00",
    "CART": "#56B4E9",
}
# Short forms, for axes where six full model names will not fit.
MODEL_SHORT = {
    "SVM (RBF)": "SVM",
    "MLP (deep)": "MLP",
    "Random Forest": "RF",
    "LightGBM": "LGBM",
    "XGBoost": "XGB",
    "CART": "CART",
}

# File-name form used by classify_map.py / train_compare.py outputs.
MODEL_SLUG = {m: m.replace(" ", "_").replace("(", "").replace(")", "")
              for m in MODEL_ORDER}

GREY = "#4d4d4d"
LIGHT = "#d9d9d9"

# --------------------------------------------------------------------------
# rcParams
# --------------------------------------------------------------------------
RC = {
    "figure.dpi": 150,
    "savefig.dpi": DPI_LINE,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.frameon": False,
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.4,
    "lines.linewidth": 0.9,
    "patch.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,       # embed as TrueType, not Type 3 - Elsevier requires it
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def use():
    """Apply the manuscript rcParams. Call once at the top of each script."""
    mpl.rcParams.update(RC)


def class_legend_handles(ids=None):
    """Patch handles for the LULC classes, in class-id order."""
    ids = ids or CLASS_IDS
    return [Patch(facecolor=CLASS_COLORS[i], edgecolor="none",
                  label=f"{i}  {CLASS_NAMES[i]}") for i in ids]


def panel_label(ax, text, dx=0.0, dy=0.0, **kw):
    """(a), (b), ... in the top-left corner, outside the data area."""
    kw.setdefault("fontsize", 8)
    kw.setdefault("fontweight", "bold")
    kw.setdefault("va", "bottom")
    kw.setdefault("ha", "left")
    return ax.text(0.0 + dx, 1.01 + dy, text, transform=ax.transAxes, **kw)


def panel_title(ax, letter, text, **kw):
    """Left-aligned '(a) title'.

    Preferred over panel_label() on narrow analysis panels, where a corner
    label and a centred title collide.
    """
    kw.setdefault("loc", "left")
    kw.setdefault("pad", 3)
    return ax.set_title(f"({letter}) {text}", **kw)


def save(fig, name, dpi=DPI_LINE, formats=("png", "pdf")):
    """Write the figure to results/figures/ in every requested format.

    Both a raster and a vector copy are produced: the PNG for quick viewing
    and for journals that want a bitmap, the PDF for submission.
    """
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in formats:
        p = FIGDIR / f"{name}.{ext}"
        fig.savefig(p, dpi=dpi, format=ext)
        out.append(p)
    plt.close(fig)
    sizes = ", ".join(f"{p.name} ({p.stat().st_size / 1024:.0f} KB)"
                      for p in out)
    print(f"  wrote {sizes}")
    return out


def require(*paths):
    """Exit with a useful message when an input table is missing."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        sys.exit("Missing input(s):\n  " + "\n  ".join(missing) +
                 "\n\nRun the analysis pipeline first "
                 "(see docs/01_import_gee_locally.md section 6).")
