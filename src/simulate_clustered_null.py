"""Check that the clustered McNemar test holds its nominal size on this design.

    python src/simulate_clustered_null.py
    python src/simulate_clustered_null.py --reps 5000 --seed 7

The manuscript rests on a negative claim: once the polygon structure of the
validation set is accounted for, none of the fifteen pairwise model
differences survives. A negative claim from a corrected test is only worth
anything if the correction itself is calibrated, so this script measures the
calibration directly rather than asserting it.

The experiment is a null experiment. Two classifiers are simulated with
*identical* expected accuracy, so every rejection is a false positive by
construction, and the rejection rate at alpha = 0.05 should come out at 0.05.
Anything much above that is a test that manufactures significance.

Generative model
----------------
The polygon sizes, the polygon count and the mean accuracy are taken from the
real validation set, so the simulated design is the design of this study and
not a generic one. Within that skeleton, each (polygon, model) pair draws its
own error rate

    q_km ~ Beta(a, b),    mean(q) = 1 - accuracy,    a + b = 1/ICC - 1,

and each pixel in polygon k is then misclassified independently with
probability q_km. Two properties of this construction matter:

* Drawing q per polygon (rather than per pixel) reproduces the intra-cluster
  correlation of correctness that is actually observed here - a polygon is
  easy or hard as a whole. For a Beta-Bernoulli hierarchy the induced ICC is
  1/(a + b + 1), which is what fixes a + b above.
* Drawing q independently for the two models, from the same distribution,
  is what makes the null interesting. The models agree on average but
  disagree polygon by polygon, so the discordant pixels arrive in runs that
  all favour the same model. That is the exact structure the naive test
  assumes away, and the reason it fails.

A version where both models share one q per polygon would also satisfy the
null, but discordances would then be direction-symmetric within a polygon and
the naive test would look nearly fine - which is not the situation in real
model comparisons, where different decision rules genuinely carve different
polygons well.

The intra-cluster correlation is swept rather than fixed at one value. A
single number invites the question "why that ICC?", and the honest answer is
that the six fitted models span a range (0.60 to 0.82 on this hold-out). The
sweep shows the whole relationship and lets the reader locate the observed
range on it.

Output
------
results/tables/clustered_null_simulation.csv, one row per (ICC, test variant)
with the false-positive rate and a binomial confidence interval on it. The
manuscript quotes the row at the observed median ICC, and check_numbers.py
holds the prose to it.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from mcnemar_test import mcnemar, mcnemar_clustered  # noqa: E402
from olofsson import icc_oneway  # noqa: E402

PRED_CSV = C.INTERIM / "test_predictions.csv"
OUT = C.TABLES / "clustered_null_simulation.csv"

NON_MODEL_COLS = {"y_true", "poly_id", "split", "class", "Id", "lon", "lat"}


def observed_design():
    """Polygon sizes, mean accuracy and the ICC range from the real hold-out.

    The median across the six fitted models is used as the headline ICC rather
    than any single model's value: the simulation is meant to describe the
    design, and a design-level quantity should not inherit one classifier's
    quirks. The min and max travel alongside it so the figure can shade the
    range the real models actually occupy.
    """
    df = pd.read_csv(PRED_CSV)
    y = df["y_true"].to_numpy()
    models = [c for c in df.columns if c not in NON_MODEL_COLS]

    sizes = df.groupby("poly_id").size().to_numpy()
    accs, iccs = [], []
    for m in models:
        ok = (df[m].to_numpy() == y).astype(float)
        accs.append(ok.mean())
        icc, _ = icc_oneway(ok, df["poly_id"].to_numpy())
        iccs.append(icc)

    return dict(sizes=sizes,
                accuracy=float(np.mean(accs)),
                icc=float(np.median(iccs)),
                icc_min=float(np.min(iccs)),
                icc_max=float(np.max(iccs)),
                n_px=int(sizes.sum()),
                n_poly=int(len(sizes)),
                models=models)


def simulate_once(rng, sizes, accuracy, icc):
    """One replicate: two equally accurate models on one clustered sample.

    Returns (p_naive, p_clustered).
    """
    q_bar = 1.0 - accuracy
    # Beta-Bernoulli: ICC = 1 / (a + b + 1). An ICC of 0 would mean pixels are
    # independent, which the concentration below cannot represent, so it is
    # floored away from the degenerate case.
    conc = 1.0 / max(icc, 1e-6) - 1.0
    a, b = q_bar * conc, (1.0 - q_bar) * conc

    n_poly = len(sizes)
    # One error rate per polygon per model. Independent across models: same
    # expected accuracy, different polygon-level strengths.
    q = rng.beta(a, b, size=(n_poly, 2))

    clusters = np.repeat(np.arange(n_poly), sizes)
    q_px = np.repeat(q, sizes, axis=0)
    wrong = rng.random(q_px.shape) < q_px

    # McNemar only ever looks at correctness, so a two-symbol alphabet is
    # enough: truth is 0 everywhere and a wrong prediction is 1.
    y_true = np.zeros(q_px.shape[0], dtype=int)
    pred_a = wrong[:, 0].astype(int)
    pred_b = wrong[:, 1].astype(int)

    _, _, _, p_naive, _ = mcnemar(y_true, pred_a, pred_b)
    _, p_clust, _ = mcnemar_clustered(y_true, pred_a, pred_b, clusters)
    return p_naive, p_clust


def sweep_point(rng, d, icc, reps, alpha):
    """False-positive rate of both tests at one ICC."""
    pn = np.empty(reps)
    pc = np.empty(reps)
    for i in range(reps):
        pn[i], pc[i] = simulate_once(rng, d["sizes"], d["accuracy"], icc)

    out = []
    for name, p in (("naive", pn), ("clustered", pc)):
        k = int((p < alpha).sum())
        lo, hi = stats.binomtest(k, reps).proportion_ci()
        out.append({"icc": round(icc, 4), "test": name,
                    "n_reject": k, "false_positive_rate": round(k / reps, 4),
                    "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=2000,
                    help="null replicates per ICC (default 2000)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not PRED_CSV.exists():
        raise SystemExit(f"{PRED_CSV} not found. Run src/train_compare.py first.")

    d = observed_design()
    print(f"Design taken from {PRED_CSV.name}: "
          f"{d['n_px']} pixels in {d['n_poly']} polygons, "
          f"mean accuracy {d['accuracy']:.4f}, "
          f"ICC {d['icc_min']:.2f}-{d['icc_max']:.2f} (median {d['icc']:.3f})")

    # A geometric-ish ladder: the interesting behaviour is all at small ICC,
    # where the naive rate climbs steeply, so the grid is dense there. The
    # observed median is inserted so the quoted number is a simulated point
    # and not an interpolation.
    grid = sorted({0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                   0.6, 0.7, 0.8, 0.9, round(d["icc"], 4)})

    print(f"Simulating {args.reps} null replicates at each of "
          f"{len(grid)} ICC values...")

    rng = np.random.default_rng(args.seed)
    rows = []
    for icc in grid:
        rows.extend(sweep_point(rng, d, icc, args.reps, args.alpha))

    res = pd.DataFrame(rows)
    res["alpha"] = args.alpha
    res["reps"] = args.reps
    res["n_pixels"] = d["n_px"]
    res["n_polygons"] = d["n_poly"]
    res["accuracy"] = round(d["accuracy"], 4)
    res["icc_observed_median"] = round(d["icc"], 4)
    res["icc_observed_min"] = round(d["icc_min"], 4)
    res["icc_observed_max"] = round(d["icc_max"], 4)
    res["seed"] = args.seed

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    print()
    print(res.pivot(index="icc", columns="test",
                    values="false_positive_rate").to_string())
    at = res[np.isclose(res.icc, round(d["icc"], 4))]
    print(f"\nAt the observed median ICC of {d['icc']:.2f}: "
          + ", ".join(f"{r.test} {r.false_positive_rate:.1%}"
                      for r in at.itertuples()))
    print(f"Saved -> {OUT}")
    print(f"\nA calibrated test rejects at {args.alpha:.0%} under this null. "
          f"The gap between the two\ncurves is the amount of significance the "
          f"naive test invents by counting\ncorrelated pixels as independent "
          f"evidence.")


if __name__ == "__main__":
    main()
