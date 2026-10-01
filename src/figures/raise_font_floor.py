"""Raise every hardcoded font size in the figure scripts to a legible floor.

    python src/figures/raise_font_floor.py --check
    python src/figures/raise_font_floor.py --apply

A reviewer complained that the figure text is unreadable, and it is. Two
things compound:

* The scripts set annotation text as low as 4.6 pt.
* Six figures drawn at 157-190 mm were being placed in an 89 mm column, which
  scales everything by 0.53-0.57 and takes 5 pt text down to under 3 pt.

The placement half is fixed in the manuscript (those figures are now
full-width floats at close to their natural size, so the scale factor is
within a few percent of 1.0). This script fixes the other half by lifting
every literal font size below FLOOR up to FLOOR. FLOOR is 6.2 rather than 6.0
because the two widest figures are set at about 0.97 of their natural size,
and 6.2 x 0.97 still clears 6 pt on the page.

Only keyword arguments that unambiguously set type size are touched --
fontsize, labelsize, title_fontsize, label_size, title_size. A bare `size=`
is left alone: in these scripts it also means arrow size and marker size.
"""

import argparse
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
FLOOR = 6.2

KEYS = ("fontsize", "labelsize", "title_fontsize", "label_size", "title_size",
        "titlesize")
PAT = re.compile(r"\b(" + "|".join(KEYS) + r")=(\d+(?:\.\d+)?)\b")

# Graphviz sizes are in points but inside a string template, and style.py
# holds the rcParams defaults, which are already at or above the floor.
SKIP = {"raise_font_floor.py", "fig_workflow_gv.py"}


def rewrite(text):
    hits = []

    def repl(m):
        key, val = m.group(1), float(m.group(2))
        if val >= FLOOR:
            return m.group(0)
        hits.append((key, val))
        return f"{key}={FLOOR}"

    return PAT.sub(repl, text), hits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="write the changes (default is a dry run)")
    args = ap.parse_args()

    total = 0
    for path in sorted(HERE.glob("*.py")):
        if path.name in SKIP:
            continue
        text = path.read_text()
        new, hits = rewrite(text)
        if not hits:
            continue
        total += len(hits)
        sizes = ", ".join(f"{v:g}" for _, v in sorted(hits, key=lambda t: t[1]))
        print(f"{path.name:32s} {len(hits):3d} raised  ({sizes})")
        if args.apply:
            path.write_text(new)

    print(f"\n{total} font sizes below {FLOOR} pt"
          + (" raised." if args.apply else "; rerun with --apply."))


if __name__ == "__main__":
    main()
