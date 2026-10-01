"""Minimal box-and-arrow drawing for the workflow diagrams.

Matplotlib rather than Graphviz so the diagrams need no system package, come
out as true vector PDF at the journal's column width, and use the same fonts
and line weights as every other figure.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(ax, x, y, w, h, title, body=None, face="#f2f2f2", edge="#444444",
        lw=0.6, title_size=6.5, body_size=5.5, radius=0.012, title_weight="bold"):
    """Rounded box with a bold title and optional smaller body text.

    Coordinates are the box centre in axes units; w and h are full width and
    height. Returns (x, y, w, h) so anchors can be computed by the caller.
    """
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       facecolor=face, edgecolor=edge, lw=lw, zorder=2)
    ax.add_patch(p)
    if body:
        ax.text(x, y + h * 0.22, title, ha="center", va="center",
                fontsize=title_size, fontweight=title_weight, zorder=3)
        ax.text(x, y - h * 0.20, body, ha="center", va="center",
                fontsize=body_size, color="#333333", zorder=3,
                linespacing=1.35)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=title_size,
                fontweight=title_weight, zorder=3)
    return (x, y, w, h)


def anchor(b, side):
    """Edge midpoint of a box: 'l', 'r', 't' or 'b'."""
    x, y, w, h = b
    return {"l": (x - w / 2, y), "r": (x + w / 2, y),
            "t": (x, y + h / 2), "b": (x, y - h / 2)}[side]


def arrow(ax, a, b, from_side="b", to_side="t", color="#444444", lw=0.7,
          style="-|>", rad=0.0, label=None, label_dx=0.0, label_dy=0.0,
          dashed=False):
    """Arrow between two boxes, from one edge midpoint to another."""
    p0 = anchor(a, from_side) if isinstance(a, tuple) and len(a) == 4 else a
    p1 = anchor(b, to_side) if isinstance(b, tuple) and len(b) == 4 else b
    ap = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=6,
                         color=color, lw=lw, zorder=1,
                         linestyle="--" if dashed else "-",
                         connectionstyle=f"arc3,rad={rad}",
                         shrinkA=1.0, shrinkB=1.0)
    ax.add_patch(ap)
    if label:
        mx, my = (p0[0] + p1[0]) / 2 + label_dx, (p0[1] + p1[1]) / 2 + label_dy
        ax.text(mx, my, label, ha="center", va="center", fontsize=6.2,
                color=color, zorder=3,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    return ap


def band(ax, y0, y1, label, color="#e8eef5", x0=0.005, x1=0.995,
         fontsize=6.2, alpha=1.0):
    """Shaded horizontal lane with a rotated label on the left."""
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color,
                               edgecolor="none", zorder=0, alpha=alpha))
    ax.text(x0 + 0.008, (y0 + y1) / 2, label, rotation=90, ha="left",
            va="center", fontsize=fontsize, color="#3d556e",
            fontweight="bold", zorder=1)


def canvas(figsize):
    """A blank 0-1 axes with no decoration, ready to draw a diagram on."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.995)
    return fig, ax
