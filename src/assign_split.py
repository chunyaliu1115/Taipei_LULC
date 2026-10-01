"""Assign the train/val split to a samples CSV that came back via Drive.

The direct path (`python src/gee_export_samples.py`) already does this. Use
this script only after a `--to-drive` export, once you have moved the CSV into
data/samples/.

    python src/assign_split.py
    python src/assign_split.py --train-frac 0.6

The split is made at PART level and stratified by class, so pixels from one
drawn polygon never straddle the divide and no class can end up with an empty
validation side.
"""

import argparse
import sys

import pandas as pd

import config as C
from gee_export_samples import assign_split, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(C.SAMPLE_CSV))
    ap.add_argument("--train-frac", type=float, default=1 - C.TEST_SIZE)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing split column")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Read {len(df)} rows from {args.csv}")

    if "poly_id" not in df.columns:
        sys.exit(
            "No 'poly_id' column. This CSV was not produced by the current "
            "version of gee_export_samples.py, so pixels cannot be grouped by "
            "polygon and any split would leak. Re-export."
        )

    if "split" in df.columns and not args.force:
        sys.exit("A 'split' column already exists. Pass --force to redo it.")

    df = assign_split(df, train_frac=args.train_frac)
    df.to_csv(args.csv, index=False)
    print(f"Wrote split -> {args.csv}")
    report(df)


if __name__ == "__main__":
    main()
