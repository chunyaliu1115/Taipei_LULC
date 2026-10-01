"""Step 3: statistical significance testing between classifiers.

Addresses editor comment 2 ("...statistical significance tests, e.g. McNemar's
test, to demonstrate whether the observed performance differences are
statistically meaningful").

Runs pairwise McNemar tests on the shared test set, applies a Holm-Bonferroni
correction for multiple comparisons, and writes a publication-ready matrix.

Two versions of the test are reported side by side, and the difference between
them is the point of the exercise. The standard McNemar test treats the 1,999
validation pixels as 1,999 independent observations. They are not: they come
from 30 polygons, and the intra-polygon correlation measured in olofsson.py is
0.29-0.86, which puts the effective sample size near 53. A test that assumes
independence under those conditions will return significance for differences
that a replication with new polygons would not reproduce.

The clustered variant is the statistic of Durkalski, Palesch, Lipsitz & Rust
(2003), "Analysis of clustered matched-pair data", Statistics in Medicine
22:2417-2428. Discordant pairs are summed within polygon first,

    X2 = (sum_k d_k)^2 / sum_k d_k^2 ,  d_k = n01_k - n10_k,

so a polygon where one model beats the other on 80 pixels contributes roughly
what a polygon that wins on 1 pixel contributes, rather than 80 times as much.
Report the clustered column; the naive column is there to show the reviewer
how much of the original significance was an artefact of pixel counting.

Usage:
    python src/mcnemar_test.py
    python src/mcnemar_test.py --cluster-col poly_id
"""

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy import stats

import config as C

PRED_CSV = C.INTERIM / "test_predictions.csv"

# Columns in test_predictions.csv that are not model predictions.
NON_MODEL_COLS = {"y_true", "poly_id", "split", "class", "Id", "lon", "lat"}


def mcnemar(y_true, pred_a, pred_b, exact_threshold=25):
    """Return (n01, n10, statistic, p_value, test_used).

    n01 = A wrong / B right ; n10 = A right / B wrong.
    Uses the exact binomial test when the discordant count is small,
    otherwise the chi-square form with Edwards' continuity correction.
    """
    a_ok = pred_a == y_true
    b_ok = pred_b == y_true
    n01 = int(np.sum(~a_ok & b_ok))
    n10 = int(np.sum(a_ok & ~b_ok))
    n = n01 + n10

    if n == 0:
        return n01, n10, 0.0, 1.0, "identical"

    if n < exact_threshold:
        p = stats.binomtest(min(n01, n10), n=n, p=0.5).pvalue
        return n01, n10, float(min(n01, n10)), float(p), "exact binomial"

    stat = (abs(n01 - n10) - 1) ** 2 / n
    p = float(stats.chi2.sf(stat, df=1))
    return n01, n10, float(stat), p, "chi2 (corrected)"


def mcnemar_clustered(y_true, pred_a, pred_b, clusters):
    """Durkalski et al. (2003) McNemar test for clustered matched pairs.

    Returns (statistic, p_value, n_informative_clusters). Discordances are
    aggregated within cluster before being squared, so a polygon on which one
    model wins 80 pixels counts as one strongly-favouring polygon rather than
    as 80 independent wins.
    """
    a_ok = pred_a == y_true
    b_ok = pred_b == y_true
    d = (~a_ok & b_ok).astype(int) - (a_ok & ~b_ok).astype(int)

    df = pd.DataFrame({"cluster": clusters, "d": d})
    dk = df.groupby("cluster")["d"].sum().to_numpy(dtype=float)

    denom = float(np.sum(dk ** 2))
    n_inf = int(np.sum(dk != 0))
    if denom == 0:
        return 0.0, 1.0, n_inf

    stat = float(np.sum(dk) ** 2 / denom)
    return stat, float(stats.chi2.sf(stat, df=1)), n_inf


def holm_bonferroni(pvals):
    """Return Holm-adjusted p-values, preserving input order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cluster-col", default="poly_id",
                    help="column identifying the sampling cluster (default poly_id)")
    ap.add_argument("--no-cluster-correction", action="store_true",
                    help="report only the naive pixel-level test")
    args = ap.parse_args()

    if not PRED_CSV.exists():
        raise SystemExit(f"{PRED_CSV} not found. Run src/train_compare.py first.")

    df = pd.read_csv(PRED_CSV)
    y_true = df["y_true"].to_numpy()

    # Only actual prediction columns. poly_id in particular travels in this
    # file so the tests can be clustered; treating it as a model would both
    # produce nonsense rows and inflate the Holm family size, making every
    # real comparison look less significant than it is.
    models = [c for c in df.columns if c not in NON_MODEL_COLS]

    clusters = None
    if not args.no_cluster_correction and args.cluster_col in df.columns:
        clusters = df[args.cluster_col].to_numpy()

    print(f"Test set n = {len(y_true)}; models = {models}")
    if clusters is None:
        print("Clustering: OFF - p-values assume 1,999 independent pixels\n")
    else:
        print(f"Clustering: {args.cluster_col}, "
              f"{len(np.unique(clusters))} polygons\n")

    rows = []
    for a, b in itertools.combinations(models, 2):
        pa, pb = df[a].to_numpy(), df[b].to_numpy()
        n01, n10, stat, p, test = mcnemar(y_true, pa, pb)
        acc_a = float(np.mean(pa == y_true))
        acc_b = float(np.mean(pb == y_true))
        row = {
            "model_A": a, "model_B": b,
            "OA_A": round(acc_a, 4), "OA_B": round(acc_b, 4),
            "delta_OA": round(acc_a - acc_b, 4),
            "n_A_wrong_B_right": n01,
            "n_A_right_B_wrong": n10,
            "statistic": round(stat, 4),
            "p_naive": p,
            "test": test,
        }
        if clusters is not None:
            cstat, cp, n_inf = mcnemar_clustered(y_true, pa, pb, clusters)
            row["clustered_statistic"] = round(cstat, 4)
            row["p_clustered"] = cp
            row["n_discordant_polygons"] = n_inf
        rows.append(row)

    res = pd.DataFrame(rows)

    # Holm is applied separately to each family; mixing the two would be
    # correcting the same 15 comparisons twice.
    res["p_holm_naive"] = holm_bonferroni(res["p_naive"].to_numpy())
    if clusters is not None:
        res["p_holm_clustered"] = holm_bonferroni(res["p_clustered"].to_numpy())

    lead = "p_holm_clustered" if clusters is not None else "p_holm_naive"
    res["significant_005"] = res[lead] < 0.05

    for col in ("p_naive", "p_clustered", "p_holm_naive", "p_holm_clustered"):
        if col in res.columns:
            res[col] = res[col].map(lambda v: f"{v:.4g}")

    res.to_csv(C.TABLES / "mcnemar_pairwise.csv", index=False)
    print(res.to_string(index=False))

    # Compact matrix of Holm-adjusted p-values for the manuscript
    mat = pd.DataFrame("-", index=models, columns=models)
    for r in res.itertuples():
        mark = "*" if r.significant_005 else ""
        val = f"{getattr(r, lead)}{mark}"
        mat.loc[r.model_A, r.model_B] = val
        mat.loc[r.model_B, r.model_A] = val
    mat.to_csv(C.TABLES / "mcnemar_matrix.csv")

    label = "cluster-corrected" if clusters is not None else "naive"
    print(f"\nHolm-adjusted p-value matrix ({label}, * = p < 0.05):\n"
          f"{mat.to_string()}")
    print(f"\nSaved -> {C.TABLES / 'mcnemar_pairwise.csv'}")

    n_sig = int(res["significant_005"].sum())
    print(f"\n{n_sig} of {len(res)} pairwise comparisons significant after correction.")

    if clusters is not None:
        n_naive = int((res["p_holm_naive"].astype(float) < 0.05).sum())
        print(f"({n_naive} of {len(res)} would be significant if the pixels were "
              f"treated as independent.)")
        if n_naive > n_sig:
            print("The gap between those two counts is the answer to the editor's\n"
                  "question: some of the apparent model differences are an artefact\n"
                  "of counting correlated pixels as independent evidence.")

    if n_sig == 0:
        print("Note: with a small effective sample, real differences can fail to\n"
              "reach significance. Report this honestly - it is a defensible\n"
              "finding and directly answers the editor's question.")


if __name__ == "__main__":
    main()
