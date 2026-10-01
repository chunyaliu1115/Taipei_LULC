"""Full-page classified map in the layout of the published Figure 3a.

Produces
    fig_lulc_detail   SVM (RBF) map, district lines and names, three zoom
                      insets, legend, north arrow, scale bar

The published Figure 3a was made in ArcGIS, which is no longer available. This
rebuilds the same layout from the GeoTIFF: district outlines and names over the
classification, and three enlarged windows in the margins - Beitou/Shilin in
the north, the Zhongshan-Datong-Songshan core, and Wenshan in the south - each
tied to its locator box on the main map.

Only SVM (RBF) is drawn. It is the most accurate of the six (OA 0.956) and the
point of this figure is to show the classification in detail; the comparison
between models is Figure 4 (fig_lulc_panel) and Figure 6
(fig_area_uncertainty).

Three departures from the original, all deliberate:

  - The locators are rectangles, not circles. A circular locator over a
    rectangular inset misstates what the inset shows; the reader has to guess
    which corners of the circle's bounding box made it in.
  - No coordinate graticule. Figure 1 already carries the graticule and the
    Taiwan locator, so repeating it here only buys a wider frame of blank
    page; the room it frees goes to the insets instead.
  - The raster is still warped to EPSG:4326 rather than left in UTM, so north
    on the page is true north everywhere and the north arrow is honest.

    python src/figures/fig_lulc_detail.py

Needs data/raw/taipei_districts.geojson for the district outlines; run
gee_repo/export_taipei_districts.js to produce it. Without that file the map
still draws, with the district names placed from the fallback table below and
no boundary lines - the figure says so in the legend rather than silently
dropping the entry.
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import style as S  # noqa: E402
import maptools as M  # noqa: E402

MODEL = "SVM (RBF)"

# Fallback district label positions, used only when the district polygons are
# absent. Taken from the label placement of the published Figure 1, so the two
# figures name the same places in the same spots.
DISTRICTS = {
    "Beitou": (121.501, 25.132),
    "Shilin": (121.553, 25.114),
    "Neihu": (121.594, 25.083),
    "Nangang": (121.607, 25.038),
    "Songshan": (121.558, 25.058),
    "Zhongshan": (121.531, 25.070),
    "Datong": (121.513, 25.063),
    "Wanhua": (121.499, 25.028),
    "Zhongzheng": (121.518, 25.032),
    "Da'an": (121.543, 25.026),
    "Xinyi": (121.571, 25.031),
    "Wenshan": (121.570, 24.989),
}

# Inset windows: label, centre, half-height in degrees of latitude, and the
# position of the inset box in axes fraction. The half-width is not given -
# it is derived from the box's own aspect so the window is never stretched.
# Half-heights are 0.018 deg, about 4 km top to bottom. Wider windows put the
# locator box outside the city on at least one corner, so the inset came back
# with a wedge of blank page in it and the reader has to work out whether that
# is nodata or a real gap in the classification.
INSETS = [
    ("A", "Beitou / Shilin", (121.522, 25.128), 0.018, (0.004, 0.522)),
    ("B", "Zhongshan / Datong / Songshan", (121.535, 25.048), 0.018,
     (0.004, 0.020)),
    ("C", "Wenshan / Xinyi", (121.572, 25.000), 0.018, (0.764, 0.522)),
]
# Taller boxes than before: dropping the graticule freed the frame, and the
# left column now runs almost the full height of the page with no gap between
# A and B.
INSET_W, INSET_H = 0.232, 0.458

# Free frame on each side, as a fraction of the axes width. The three insets
# and the legend all sit inside it, so nothing is drawn over the city.
MARGIN = 0.245

WHITE = "#ffffff"
INSET_EDGE = "#ffffff"


# --------------------------------------------------------------------------
# data


def reproject_to_wgs84(path, dst_res=0.0001):
    """Classified raster warped to EPSG:4326 so the graticule is axis-aligned.

    Plotting in the native UTM zone and labelling the frame in degrees would
    be a lie of half a degree of meridian convergence - small, but the whole
    point of a graticule is that the reader can take coordinates off it.
    Warping once, with nearest-neighbour so no class is invented at a
    boundary, removes the question.
    """
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    with rasterio.open(path) as src:
        dst_crs = "EPSG:4326"
        transform, w, h = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=dst_res)
        out = np.zeros((h, w), dtype=src.dtypes[0])
        reproject(source=rasterio.band(src, 1), destination=out,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=dst_crs,
                  resampling=Resampling.nearest, dst_nodata=0)
    left, top = transform.c, transform.f
    right, bottom = left + w * transform.a, top + h * transform.e
    return np.ma.masked_equal(out, 0), (left, right, bottom, top)


def load_districts():
    """District polygons in EPSG:4326, or None if the export has not been run."""
    import geopandas as gpd

    for name in ("taipei_districts.geojson", "taipei_districts.gpkg"):
        p = C.RAW / name
        if p.exists():
            return gpd.read_file(p).to_crs("EPSG:4326")
    return None


def district_labels(gdf):
    """(name, lon, lat) per district, from the polygons if we have them.

    representative_point rather than centroid: Beitou and Wenshan are concave
    enough that their centroids fall outside the polygon, which would put the
    name over a neighbour.
    """
    if gdf is None:
        return [(n, x, y) for n, (x, y) in DISTRICTS.items()]
    for col in ("shapeName", "ADM2_NAME", "TOWNENG", "TOWNNAME", "NAME_2",
                "name"):
        if col in gdf.columns and gdf[col].notna().any():
            out = []
            for _, r in gdf.iterrows():
                p = r.geometry.representative_point()
                out.append((str(r[col]).replace(" District", ""), p.x, p.y))
            return out
    return [(n, x, y) for n, (x, y) in DISTRICTS.items()]


# --------------------------------------------------------------------------
# frame


def neat_frame(ax):
    """Thin border, no ticks and no coordinate labels.

    The graticule that used to sit here is Figure 1's job. Keeping the border
    is worth it even without ticks: it tells the reader where the page ends
    and stops the two outer insets from floating.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.7)
        s.set_color("#333333")


# --------------------------------------------------------------------------
# figure


def draw_map(ax, arr, extent, boundary, districts, labels, lat0,
             names=True, lw_city=1.0, lw_dist=0.45, label_size=6.2):
    cmap, norm = M.class_cmap()
    ax.imshow(arr, extent=extent, cmap=cmap, norm=norm,
              interpolation="nearest", zorder=1)
    if districts is not None:
        districts.boundary.plot(ax=ax, color=WHITE, lw=lw_dist, zorder=3,
                                alpha=0.85)
    if boundary is not None:
        boundary.boundary.plot(ax=ax, color="#111111", lw=lw_city, zorder=4)
    if names:
        pe = M._halo(1.5, "#00000088")
        for name, x, y in labels:
            ax.text(x, y, name, fontsize=label_size, color="white",
                    ha="center", va="center", zorder=6, path_effects=pe)
    ax.set_aspect(1.0 / np.cos(np.radians(lat0)))


def fig_detail():
    path = C.EXPORTS / f"taipei_lulc_{S.MODEL_SLUG[MODEL]}.tif"
    S.require(path)
    arr, extent = reproject_to_wgs84(path)

    import geopandas as gpd
    boundary = gpd.read_file(C.RAW / "Taipei_City.geojson").to_crs("EPSG:4326")
    districts = load_districts()
    labels = district_labels(districts)

    bx0, by0, bx1, by1 = boundary.total_bounds
    lat0 = (by0 + by1) / 2

    # The page shape is derived, not chosen. Latitude range is fixed by the
    # data; MARGIN fixes how much free frame the insets and legend need on
    # each side, which fixes the longitude range; the equal-ground-scale
    # constraint then fixes the height. Choosing the figure aspect by hand
    # instead meant the margin came out to whatever was left over, and the
    # insets kept landing on the western edge of the city.
    pad_y = (by1 - by0) * 0.012
    y0, y1 = by0 - pad_y, by1 + pad_y
    span_x = (bx1 - bx0) / (1 - 2 * MARGIN)
    # Near full bleed. With the tick labels gone there is nothing to leave
    # room for outside the frame, so the axes takes the page.
    rect = [0.006, 0.006, 0.988, 0.988]
    w_ax_in = S.DOUBLE * rect[2]
    h_ax_in = w_ax_in * (y1 - y0) / (span_x * np.cos(np.radians(lat0)))

    fig = plt.figure(figsize=(S.DOUBLE, h_ax_in / rect[3]))
    ax = fig.add_axes(rect)
    cx = (bx0 + bx1) / 2
    ax.set_xlim(cx - span_x / 2, cx + span_x / 2)
    ax.set_ylim(y0, y1)
    w_in, h_in = w_ax_in, h_ax_in

    draw_map(ax, arr, extent, boundary, districts, labels, lat0)
    neat_frame(ax)

    # ---- insets ----------------------------------------------------------
    ax_w_in, ax_h_in = w_in, h_in
    for tag, _, (clon, clat), half_h, (ix, iy) in INSETS:
        sub = ax.inset_axes([ix, iy, INSET_W, INSET_H])
        box_w_in, box_h_in = INSET_W * ax_w_in, INSET_H * ax_h_in
        half_w = half_h * (box_w_in / box_h_in) / np.cos(np.radians(clat))
        sub.set_xlim(clon - half_w, clon + half_w)
        sub.set_ylim(clat - half_h, clat + half_h)
        draw_map(sub, arr, extent, boundary, districts, labels, clat,
                 names=False, lw_city=0.8, lw_dist=0.35)
        sub.set_xticks([])
        sub.set_yticks([])
        for s in sub.spines.values():
            s.set_linewidth(0.9)
            s.set_color("#222222")
        sub.text(0.045, 0.955, tag, transform=sub.transAxes, fontsize=7,
                 fontweight="bold", va="top", ha="left", color="white",
                 zorder=8, path_effects=M._halo(1.8, "#000000"))

        # Locator box only, no leader lines. Connecting each box to its inset
        # drew three long diagonals straight across the classification, and
        # the thing they were disambiguating - which box goes with which
        # inset - is already carried by the letter at both ends.
        # Haloed, because a white locator crossing the city edge is invisible
        # for the half of its perimeter that lies on blank page.
        box = Rectangle(
            (clon - half_w, clat - half_h), 2 * half_w, 2 * half_h,
            facecolor="none", edgecolor=INSET_EDGE, lw=0.9, zorder=7)
        box.set_path_effects(M._halo(2.0, "#000000"))
        ax.add_patch(box)
        ax.text(clon - half_w, clat + half_h, f"{tag} ", fontsize=6.5,
                fontweight="bold", color="white", va="bottom", ha="right",
                zorder=8, path_effects=M._halo(1.8, "#000000"))

    # ---- legend ----------------------------------------------------------
    handles = [Patch(facecolor=S.CLASS_COLORS[i], edgecolor="#333333",
                     lw=0.3, label=f"{i}  {C.CLASS_NAMES[i]}")
               for i in S.CLASS_IDS]
    if districts is not None:
        handles.append(Line2D([], [], color="#999999", lw=0.8,
                              label="District boundary"))
    handles.append(Line2D([], [], color="#111111", lw=1.2,
                          label="City boundary"))
    # The right column below inset C carries the legend, then the furniture,
    # then the provenance note, in that order down the page.
    leg = ax.legend(handles=handles, loc="upper left",
                    bbox_to_anchor=(0.766, 0.470), fontsize=6.2,
                    handlelength=1.0, handleheight=0.9, handletextpad=0.5,
                    borderpad=0.5, labelspacing=0.40, frameon=True)
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_edgecolor("#333333")
    leg.set_zorder(9)

    # ---- furniture -------------------------------------------------------
    M.north_arrow(ax, loc=(0.952, 0.115), size=0.058, fontsize=8)
    _km_scalebar(ax, lat0, km=4, loc=(0.772, 0.118))

    ax.text(0.766, 0.020,
            f"{MODEL}, Sentinel-2 median composite\n"
            f"{C.DATE_START} to {C.DATE_END}, {C.SCALE} m\n"
            "WGS 1984, geographic",
            transform=ax.transAxes, fontsize=6.2, color="#333333",
            va="bottom", ha="left", linespacing=1.5, zorder=9)
    return fig


def _km_scalebar(ax, lat0, km=4, loc=(0.772, 0.118)):
    """Scale bar in kilometres for an axes whose units are degrees.

    maptools.scalebar assumes metres, which is right for every other map in
    the paper and wrong here. Rather than teach it about latitude, this draws
    the four-segment bar of the published figure directly.
    """
    x0, x1 = ax.get_xlim()
    deg_per_km = 1.0 / (111.320 * np.cos(np.radians(lat0)))
    frac = km * deg_per_km / (x1 - x0)
    fx, fy, h, n = loc[0], loc[1], 0.011, 4
    seg = frac / n
    pe = M._halo(2.0, "white")
    for k in range(n):
        r = Rectangle((fx + k * seg, fy), seg, h, transform=ax.transAxes,
                      facecolor="black" if k % 2 == 0 else "white",
                      edgecolor="black", lw=0.4, zorder=9)
        r.set_path_effects(pe)
        ax.add_patch(r)
    kw = dict(transform=ax.transAxes, fontsize=6.2, zorder=10,
              path_effects=pe)
    for k in range(n + 1):
        ax.text(fx + k * seg, fy + h + 0.006, f"{int(k * km / n)}",
                ha="center", va="bottom", **kw)
    ax.text(fx + frac / 2, fy - 0.008, "Kilometers", ha="center", va="top",
            **kw)


def main():
    S.use()
    print("fig_lulc_detail")
    S.save(fig_detail(), "fig_lulc_detail", dpi=S.DPI_RASTER)


if __name__ == "__main__":
    main()
