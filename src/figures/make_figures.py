"""Regenerate every manuscript figure.

    python src/figures/make_figures.py            # everything
    python src/figures/make_figures.py maps       # one group
    python src/figures/make_figures.py --list

Each group is a module in this directory with a main() of its own, so a single
figure can still be re-run on its own while iterating. Groups are independent;
a group whose input tables are missing reports the fact and the rest continue,
which matters because the map figures need the exported GeoTIFFs and the
analysis figures only need results/tables/.

Figures are written to results/figures/ as PNG (600 dpi line art, 300 dpi for
anything containing a map raster) and PDF.
"""

import argparse
import importlib
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import style as S  # noqa: E402

GROUPS = {
    "maps": ("fig_maps", "study area, per-model LULC maps, model agreement"),
    "detail": ("fig_lulc_detail",
               "full-page SVM map with district names and three zoom insets"),
    "flow": ("fig_flowchart", "workflow and classification-scheme diagrams"),
    "accuracy": ("fig_accuracy", "OA/kappa CIs, McNemar, per-class UA and PA"),
    "area": ("fig_area_uncertainty", "between-model area spread vs CIs"),
    "errors": ("fig_error_concentration",
               "recurring confusions, error concentration, separability"),
    "design": ("fig_sampling_design",
               "polygon sampling design, n_eff, reference allocation"),
}


def run(group):
    mod_name, _ = GROUPS[group]
    print(f"\n=== {group}  ({mod_name})")
    t0 = time.time()
    try:
        mod = importlib.import_module(mod_name)
        # fig_maps.main() takes its own CLI flags. Hand it an empty argv so it
        # parses defaults rather than this runner's group names.
        try:
            mod.main([])
        except TypeError:
            mod.main()
    except SystemExit as e:
        # style.require() exits when an input table is missing. That is a
        # missing prerequisite, not a bug, so report it and keep going.
        print(f"  skipped: {e}")
        return False
    except Exception:
        traceback.print_exc()
        print(f"  FAILED: {group}")
        return False
    print(f"  {group} done in {time.time() - t0:.1f}s")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # No `choices=` here: argparse validates the *default* of a nargs="*"
    # positional against choices, so an empty default is rejected before the
    # parse even begins. Validate by hand instead.
    ap.add_argument("groups", nargs="*", default=[],
                    help=f"groups to build, any of {', '.join(GROUPS)} "
                         "(default: all)")
    ap.add_argument("--list", action="store_true", help="list groups and exit")
    a = ap.parse_args()

    bad = [g for g in a.groups if g not in GROUPS]
    if bad:
        ap.error(f"unknown group(s): {', '.join(bad)}. "
                 f"Choose from: {', '.join(GROUPS)}")

    if a.list:
        for k, (m, desc) in GROUPS.items():
            print(f"  {k:<10} {desc}")
        return

    S.use()
    todo = a.groups or list(GROUPS)
    ok = [g for g in todo if run(g)]

    figs = sorted(C.FIGURES.glob("*.pdf"))
    print(f"\n{len(ok)}/{len(todo)} groups built; "
          f"{len(figs)} figures now in {C.FIGURES}")
    for p in figs:
        png = p.with_suffix(".png")
        size = f"{png.stat().st_size / 1024:.0f} KB" if png.exists() else "-"
        print(f"  {p.stem:<28} pdf {p.stat().st_size / 1024:>5.0f} KB   "
              f"png {size:>9}")
    if len(ok) != len(todo):
        sys.exit(1)


if __name__ == "__main__":
    main()
