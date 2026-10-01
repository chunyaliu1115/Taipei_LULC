"""Map furniture: scale bar, north arrow, class raster rendering.

Kept apart from style.py because these helpers know about coordinate systems
and rasters, whereas style.py only knows about ink.
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
import style as S  # noqa: E402


def class_cmap():
    """Discrete colormap for class ids 1..N, with 0 (nodata) transparent.

    A ListedColormap plus BoundaryNorm is used rather than a continuous
    colormap so that each class gets exactly its palette colour and no
    interpolation happens at class edges.
    """
    ids = S.CLASS_IDS
    cmap = ListedColormap([S.CLASS_COLORS[i] for i in ids])
    cmap.set_bad(alpha=0.0)
    norm = BoundaryNorm([ids[0] - 0.5] + [i + 0.5 for i in ids], cmap.N)
    return cmap, norm


def read_class_raster(path):
    """Read a classified GeoTIFF as a masked array plus its plotting extent.

    Nodata (0, written by classify_map.py for everything outside the city)
    is masked so the page shows through instead of a black frame.
    """
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read(1)
        nodata = src.nodata if src.nodata is not None else 0
        b = src.bounds
        crs = src.crs
    arr = np.ma.masked_equal(arr, nodata)
    return arr, (b.left, b.right, b.bottom, b.top), crs


def read_rgb_composite(path, bands=("B4", "B3", "B2"), pct=(2, 98)):
    """True-colour stretch of the Sentinel-2 composite for a context panel.

    Bands are looked up by their descriptions rather than by index, because
    the export band order follows config.BANDS_PUBLISHED and would silently
    reorder if that list ever changed.
    """
    import rasterio

    with rasterio.open(path) as src:
        names = list(src.descriptions)
        idx = [names.index(b) + 1 for b in bands]
        stack = np.stack([src.read(i).astype("float32") for i in idx], -1)
        nodata = src.nodata
        b = src.bounds
    if nodata is not None:
        stack[stack == nodata] = np.nan
    lo, hi = np.nanpercentile(stack, pct)
    rgb = np.clip((stack - lo) / (hi - lo), 0, 1)
    alpha = (~np.isnan(stack).any(-1)).astype("float32")
    rgb = np.nan_to_num(rgb)
    return np.dstack([rgb, alpha]), (b.left, b.right, b.bottom, b.top)


def _halo(width=1.6, fg="white"):
    """Outline so map furniture stays legible over both imagery and page.

    Everything drawn on a map has to survive being placed over a dark
    Sentinel-2 composite in one figure and over blank page in the next; a
    contrasting stroke removes the need to hand-pick a colour per figure.
    """
    import matplotlib.patheffects as pe

    return [pe.withStroke(linewidth=width, foreground=fg)]


def scalebar(ax, length_m=5000, loc=(0.06, 0.06), height=0.012, label=None,
             color="black", fontsize=6.2, halo=True):
    """Two-tone scale bar in axes coordinates, sized in map units.

    Assumes the axes are in a projected CRS whose unit is the metre - true
    for EPSG:32651, which every raster here is in.
    """
    x0, x1 = ax.get_xlim()
    frac = length_m / abs(x1 - x0)
    fx, fy = loc
    n = 4
    seg = frac / n
    fx_pe = _halo(2.2, "white" if color == "black" else "black")
    for k in range(n):
        r = plt.Rectangle(
            (fx + k * seg, fy), seg, height, transform=ax.transAxes,
            facecolor=("black" if k % 2 == 0 else "white"),
            edgecolor=color, lw=0.4, zorder=5, clip_on=False)
        if halo:
            r.set_path_effects(fx_pe)
        ax.add_patch(r)
    label = label or (f"{length_m / 1000:g} km")
    kw = dict(transform=ax.transAxes, fontsize=fontsize, color=color,
              zorder=6, path_effects=fx_pe if halo else None)
    ax.text(fx + frac / 2, fy + height + 0.008, label, ha="center",
            va="bottom", **kw)
    ax.text(fx, fy - 0.004, "0", ha="center", va="top", **kw)


def north_arrow(ax, loc=(0.93, 0.86), size=0.075, color="black", fontsize=7,
                halo=True):
    """Simple filled triangle with an N above it.

    The rasters are in a UTM zone, so grid north and true north differ by the
    meridian convergence - under half a degree over a 21 km city, which is
    well below the width of the arrow.
    """
    fx, fy = loc
    pe = _halo(2.2, "white" if color == "black" else "black") if halo else None
    a = ax.annotate("", xy=(fx, fy + size), xytext=(fx, fy),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8,
                                    mutation_scale=7), zorder=5)
    if pe:
        a.arrow_patch.set_path_effects(pe)
    ax.text(fx, fy + size + 0.01, "N", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=fontsize, fontweight="bold", color=color,
            zorder=6, path_effects=pe)


def tidy_map_axes(ax, boundary=None, bcolor="#222222", blw=0.5, crop=True,
                  pad=400):
    """Strip ticks, keep a hairline frame, optionally overlay the city outline.

    `crop` trims the view to the boundary's bounding box. The exported
    rasters cover a rectangular bounding box that is roughly twice the area
    of the city, so without this the panels are mostly empty page.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.4)
        s.set_color("#888888")
    ax.set_aspect("equal")
    if boundary is not None:
        boundary.boundary.plot(ax=ax, color=bcolor, lw=blw, zorder=4)
        if crop:
            x0, y0, x1, y1 = boundary.total_bounds
            ax.set_xlim(x0 - pad, x1 + pad)
            ax.set_ylim(y0 - pad, y1 + pad)


def load_boundary(crs="EPSG:32651"):
    """City boundary as a GeoDataFrame in the raster CRS, or None if absent."""
    import geopandas as gpd

    for name in ("Taipei_City.gpkg", "Taipei_City.geojson"):
        p = C.RAW / name
        if p.exists():
            return gpd.read_file(p).to_crs(crs)
    return None
