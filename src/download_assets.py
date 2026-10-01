"""Pull the GEE assets down to local disk.

Three things, three mechanisms:

  vectors  ROI_TAIPEI polygons + Taipei_City boundary -> GeoPackage/GeoJSON
           Small, come down directly through the API.

  raster   Sentinel-2 composite -> GeoTIFF
           Too big for the 32 MB direct-download ceiling at 10 m, so this uses
           geemap's tiled downloader (geedim under the hood). Kept as scaled
           int16 reflectance rather than float32, which halves the file size.

  table    Band values at every sample pixel -> CSV
           This is the one you actually need for the modelling. Produced by
           src/gee_export_samples.py, not here.

Usage:
    python src/download_assets.py                # vectors only (fast)
    python src/download_assets.py --raster       # vectors + S2 composite
    python src/download_assets.py --raster --extended   # 9-band version
    python src/download_assets.py --estimate     # size check, downloads nothing
"""

import argparse
import sys

import ee

import config as C


def init_ee():
    try:
        ee.Initialize(project=C.EE_PROJECT)
    except Exception:
        print("Authenticating with Earth Engine (a browser window will open)...")
        ee.Authenticate()
        ee.Initialize(project=C.EE_PROJECT)
    print(f"Earth Engine ready (project: {C.EE_PROJECT})")


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------
def download_vectors():
    import geemap

    targets = {
        "roi_polygons": C.RAW / "ROI_TAIPEI.geojson",
        "taipei_boundary": C.RAW / "Taipei_City.geojson",
    }

    for key, out in targets.items():
        asset = C.ASSETS[key]
        fc = ee.FeatureCollection(asset)
        n = fc.size().getInfo()
        print(f"\n{key}: {asset}  ({n} features)")

        geemap.ee_to_geojson(fc, filename=str(out))
        print(f"  -> {out}")

        # Also write a GeoPackage, which QGIS handles better than GeoJSON
        try:
            import geopandas as gpd
            gdf = gpd.read_file(out)

            # GEE writes system:index out as a lowercase 'id' field, while the
            # class label lives in 'Id'. GeoPackage column names are
            # case-insensitive, so the two collide and the write fails with
            # "Error adding field 'Id' to layer". Rename the GEE one.
            if "id" in gdf.columns and "Id" in gdf.columns:
                gdf = gdf.rename(columns={"id": "system_index"})

            gpkg = out.with_suffix(".gpkg")
            gdf.to_file(gpkg, driver="GPKG")
            print(f"  -> {gpkg}")

            if C.CLASS_PROPERTY in gdf.columns:
                # Features vs parts matters here. This asset stores one
                # MultiPolygon per class, so the feature count (5) says nothing
                # about how many independent training areas exist (110).
                parts = gdf.explode(index_parts=False)
                tab = gdf.groupby(C.CLASS_PROPERTY).size().rename("features").to_frame()
                tab["parts"] = parts.groupby(C.CLASS_PROPERTY).size()
                if "Class" in gdf.columns:
                    tab.insert(0, "name", gdf.groupby(C.CLASS_PROPERTY)["Class"].first())
                print(f"  per class:\n{tab.to_string()}")
                if (tab["parts"] > tab["features"]).any():
                    print("  NOTE: multipart geometries. The train/val split is "
                          "made at PART level; see src/gee_export_samples.py")
        except Exception as exc:
            print(f"  (GeoPackage step skipped: {exc})")


# ---------------------------------------------------------------------------
# Raster
# ---------------------------------------------------------------------------
def mask_s2_clouds(image):
    qa = image.select("QA60")
    mask = (qa.bitwiseAnd(1 << 10).eq(0)
            .And(qa.bitwiseAnd(1 << 11).eq(0)))
    return image.updateMask(mask).copyProperties(image, ["system:time_start"])


def build_composite(extended=False):
    """Rebuild the composite used in the GEE script.

    Reflectance is left in native scaled integer units (0-10000). Dividing by
    10000 is deferred to the local side so the download stays int16.
    """
    aoi = ee.FeatureCollection(C.ASSETS["taipei_boundary"]).geometry()

    col = (ee.ImageCollection(C.S2_COLLECTION)
           .filterDate(C.DATE_START, C.DATE_END)
           .filterBounds(aoi)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", C.MAX_CLOUD_PCT))
           .map(mask_s2_clouds))

    print(f"  scenes in composite: {col.size().getInfo()}")
    composite = col.median().clip(aoi)

    bands = C.BANDS_EXTENDED if extended else C.BANDS_PUBLISHED
    img = composite.select(bands)

    if extended:
        # Indices are computed locally after download to avoid mixing scales;
        # here we just carry the SWIR bands needed to derive them.
        pass

    return img.toInt16(), aoi, bands


def estimate_size(aoi, n_bands, scale):
    """Rough uncompressed size of the download, in MB."""
    area_m2 = aoi.area(maxError=10).getInfo()
    n_px = area_m2 / (scale ** 2)
    mb = n_px * n_bands * 2 / 1e6          # int16 = 2 bytes
    bounds_m2 = aoi.bounds(maxError=10).area(maxError=10).getInfo()
    mb_bbox = bounds_m2 / (scale ** 2) * n_bands * 2 / 1e6
    print(f"  AOI area: {area_m2/1e6:,.1f} sq.km  ({n_px/1e6:.2f} M pixels)")
    print(f"  estimated size: ~{mb:.0f} MB clipped, ~{mb_bbox:.0f} MB over the "
          f"bounding box ({n_bands} bands, int16, {scale} m)")
    if mb_bbox > 32:
        print("  > exceeds the 32 MB direct-download limit, tiled download required")
    return mb_bbox


def download_raster(extended=False, estimate_only=False):
    img, aoi, bands = build_composite(extended=extended)
    print(f"\nSentinel-2 composite bands: {bands}")

    size_mb = estimate_size(aoi, len(bands), C.SCALE)
    if estimate_only:
        return

    suffix = "extended" if extended else "published"
    out = C.RAW / f"s2_composite_2020_{suffix}.tif"

    import geemap

    # geemap.download_ee_image tiles automatically and has no 32 MB ceiling.
    try:
        geemap.download_ee_image(
            image=img,
            filename=str(out),
            region=aoi,
            crs="EPSG:32651",       # UTM 51N, appropriate for Taipei
            scale=C.SCALE,
            dtype="int16",
        )
        print(f"  -> {out}")
        print("  NOTE: values are scaled reflectance. Divide by 10000 before use.")
        return
    except AttributeError:
        print("  geemap.download_ee_image unavailable - falling back")
    except Exception as exc:
        print(f"  tiled download failed: {exc}")

    if size_mb < 32:
        geemap.ee_export_image(img, filename=str(out), scale=C.SCALE,
                               region=aoi, crs="EPSG:32651", file_per_band=False)
        print(f"  -> {out}")
    else:
        print("\n  Too large for a direct pull. Starting a Drive export instead;\n"
              "  watch the Tasks tab at https://code.earthengine.google.com,\n"
              f"  then move the GeoTIFF into {C.RAW}")
        task = ee.batch.Export.image.toDrive(
            image=img,
            description=f"s2_composite_2020_{suffix}",
            folder="GEE_exports",
            region=aoi,
            scale=C.SCALE,
            crs="EPSG:32651",
            maxPixels=int(1e13),
        )
        task.start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster", action="store_true",
                    help="also download the Sentinel-2 composite")
    ap.add_argument("--extended", action="store_true",
                    help="6-band composite instead of 4-band")
    ap.add_argument("--estimate", action="store_true",
                    help="report the raster size and exit without downloading")
    args = ap.parse_args()

    init_ee()

    if args.estimate:
        download_raster(extended=args.extended, estimate_only=True)
        return

    download_vectors()

    if args.raster:
        download_raster(extended=args.extended)
    else:
        print("\nSkipped the raster (pass --raster if you want it). "
              "It is not needed for the model comparison - only for "
              "producing classified maps locally.")

    print(f"\nDone. Files in {C.RAW}")
    print("Next: python src/gee_export_samples.py")


if __name__ == "__main__":
    main()
