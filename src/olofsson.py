"""Area-weighted accuracy and area estimation, after Olofsson et al. (2014).

Olofsson, P., Foody, G.M., Herold, M., Stehman, S.V., Woodcock, C.E. and Wulder,
M.A. (2014). "Good practices for estimating area and assessing accuracy of land
change." Remote Sensing of Environment 148: 42-57.

WHY THIS IS NEEDED HERE
-----------------------
The validation pixels are wildly disproportionate to the classes' actual
extents. Forest contributes 1,048 of 1,999 validation pixels; swamp contributes
135. A plain confusion matrix therefore reports the accuracy of a *sample* whose
class mix nobody chose deliberately, and the resulting overall accuracy is
dominated by whichever class happened to have the most polygons drawn over it.

Olofsson's estimator fixes this by reweighting each map class by the fraction of
the map it actually occupies (W_i), which turns the confusion matrix into an
estimate of accuracy over the map rather than over the sample. The same
machinery then yields area estimates with confidence intervals, which is what
the Results section should report instead of raw pixel counts.

WHAT THIS SCRIPT DOES NOT FIX
-----------------------------
Olofsson's estimator assumes the sample within each stratum is a PROBABILITY
sample - random points drawn from the mapped stratum. Ours is not: it is every
pixel inside a handful of hand-drawn polygons. Two consequences:

  1. The sample is spatially clustered, so the nominal n_i massively overstates
     the information content. `--cluster-col poly_id` (on by default) replaces
     n_i with an effective sample size n_i / deff_i, where the design effect is
     estimated from the intra-polygon correlation. On this dataset that turns
     1,999 pixels into roughly 50 independent observations.

  2. The sample is not random within the stratum - polygons were placed on
     unambiguous, spectrally pure examples. No statistical correction repairs
     that. It biases accuracy upward and the bias cannot be quantified from the
     data itself.

So: report these numbers as the best available estimate from the existing data,
state both caveats explicitly in Limitations, and draw a genuine stratified
random reference sample (src/draw_reference_sample.py) for the final version.

Usage:
    python src/olofsson.py                       # all models in the pred file
    python src/olofsson.py --model RF
    python src/olofsson.py --areas results/tables/map_areas.json
    python src/olofsson.py --no-cluster-correction   # nominal n, optimistic SEs
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

import config as C

Z95 = 1.959963984540054


def wilson_ci(p, n, z=Z95):
    """Wilson score interval - does not collapse when p is 0 or 1."""
    if not np.isfinite(p) or n <= 0:
        return np.nan, np.nan
    d = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


# ---------------------------------------------------------------------------
# Design effect
# ---------------------------------------------------------------------------
def icc_oneway(values, clusters):
    """One-way random-effects ICC for unequal cluster sizes.

    `values` is typically a 0/1 correctness indicator. Returns (icc, m0), and
    (0.0, 1.0) when the design cannot support an estimate - fewer than two
    clusters, or no variance to partition.
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)

    uniq = np.unique(clusters)
    k = len(uniq)
    n = len(values)
    if k < 2 or n <= k or values.std() == 0:
        return 0.0, 1.0

    grand = values.mean()
    sizes = np.array([(clusters == c).sum() for c in uniq], dtype=float)
    means = np.array([values[clusters == c].mean() for c in uniq])

    ss_between = float((sizes * (means - grand) ** 2).sum())
    ss_within = float(sum(((values[clusters == c] - m) ** 2).sum()
                          for c, m in zip(uniq, means)))

    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)

    # m0: the "average" cluster size that makes the ANOVA estimator unbiased
    # under unequal sizes (Donner 1986).
    m0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    if m0 <= 1 or (ms_between + (m0 - 1) * ms_within) == 0:
        return 0.0, 1.0

    icc = (ms_between - ms_within) / (ms_between + (m0 - 1) * ms_within)
    return float(np.clip(icc, 0.0, 1.0)), float(m0)


def design_effects(df, map_col, correct_col, cluster_col):
    """Kish design effect per map stratum: deff = 1 + (m0 - 1) * ICC.

    Degenerate case worth being careful about: when a stratum has no errors at
    all, the correctness indicator has zero variance, the ICC is undefined, and
    a naive deff of 1.0 would declare all 951 forest pixels independent. That
    is exactly backwards - a stratum classified perfectly tells you nothing
    about whether its pixels are independent, and for polygon-derived data the
    prior should be that they are heavily correlated. So the degenerate case
    falls back to deff = m0, i.e. n_eff = the number of polygons, which is the
    ICC -> 1 limit and the conservative end of the range.
    """
    out = {}
    for cls, grp in df.groupby(map_col):
        if cluster_col not in grp.columns or grp[cluster_col].nunique() < 2:
            out[int(cls)] = float(max(1.0, len(grp) / 1.0)) \
                if grp[cluster_col].nunique() == 1 else 1.0
            continue
        vals = grp[correct_col].to_numpy()
        icc, m0 = icc_oneway(vals, grp[cluster_col].to_numpy())
        if vals.std() == 0:
            k = grp[cluster_col].nunique()
            out[int(cls)] = float(max(1.0, len(grp) / k))
            continue
        out[int(cls)] = float(max(1.0, 1.0 + (m0 - 1.0) * icc))
    return out


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------
def olofsson(y_map, y_ref, areas, labels=None, n_eff=None, total_area=None):
    """Stratified estimator of accuracy and area.

    Rows i index the MAP class (the stratum). Columns j index the REFERENCE
    class. This orientation is not arbitrary: the sample is drawn from the map,
    so the map class is what defines the stratum and carries the known weight.

    Parameters
    ----------
    y_map, y_ref : array-like of int
        Predicted (map) and reference class per sample unit.
    areas : dict {class_id: mapped area in the same unit throughout}
        Typically sq.km from src/classify_map.py.
    n_eff : dict {class_id: effective sample size}, optional
        Substituted for n_i in every variance term. Defaults to the nominal n_i.
    total_area : float, optional
        Defaults to sum(areas.values()).

    Returns
    -------
    dict with keys 'overall', 'per_class', 'matrix_counts', 'matrix_proportions'
    """
    y_map = np.asarray(y_map, dtype=int)
    y_ref = np.asarray(y_ref, dtype=int)

    if labels is None:
        labels = sorted(set(y_map) | set(y_ref) | set(areas))
    labels = list(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    q = len(labels)

    # Counts n_ij
    n_ij = np.zeros((q, q), dtype=float)
    for m, r in zip(y_map, y_ref):
        if m in idx and r in idx:
            n_ij[idx[m], idx[r]] += 1
    n_i = n_ij.sum(axis=1)

    A = np.array([float(areas.get(lab, 0.0)) for lab in labels])
    A_tot = float(total_area) if total_area else float(A.sum())
    if A_tot <= 0:
        sys.exit("Total mapped area is zero - check the --areas file.")
    W = A / A_tot

    # Effective sample size drives every variance. Nominal n_i unless told
    # otherwise; see the module docstring for why nominal is wrong here.
    ne = np.array([float(n_eff.get(lab, n_i[i])) if n_eff else n_i[i]
                   for i, lab in enumerate(labels)])
    ne = np.maximum(ne, 2.0)          # variance formulas need n_i - 1 > 0

    empty = [labels[i] for i in range(q) if n_i[i] == 0]
    if empty:
        print(f"  NOTE: map class(es) {empty} have no sample units. They are "
              f"carried at zero and their accuracy is undefined.")

    # p_ij = W_i * n_ij / n_i   (estimated proportion of area)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(n_i[:, None] > 0, n_ij / n_i[:, None], 0.0)
    p_ij = W[:, None] * rate

    # --- Overall accuracy -------------------------------------------------
    OA = float(np.trace(p_ij))
    U = np.where(n_i > 0, np.diag(n_ij) / np.where(n_i > 0, n_i, 1), np.nan)
    var_OA = float(np.nansum(W ** 2 * U * (1 - U) / (ne - 1)))
    se_OA = float(np.sqrt(max(var_OA, 0.0)))

    # --- User's accuracy (row) -------------------------------------------
    var_U = U * (1 - U) / (ne - 1)
    se_U = np.sqrt(np.clip(var_U, 0, None))

    # Olofsson's variance is the Wald form, which degenerates at the boundary:
    # a stratum with no errors gets UA = 100% +/- 0.00, which no reviewer should
    # accept. Wilson score intervals stay sensible there, so they are used for
    # the reported UA interval whenever UA hits 0 or 1. The SE column is left as
    # the Wald value because that is what feeds V(OA) and V(PA) in the paper's
    # formulas.
    ua_lo, ua_hi = np.clip(U - Z95 * se_U, 0, 1), np.clip(U + Z95 * se_U, 0, 1)
    for i in range(q):
        if np.isfinite(U[i]) and (U[i] <= 0 or U[i] >= 1):
            ua_lo[i], ua_hi[i] = wilson_ci(U[i], ne[i])

    # --- Producer's accuracy (column), Olofsson eq. 7 ---------------------
    # N_i is the mapped pixel/area total of stratum i; using area works because
    # only ratios of N enter.
    N = A
    N_hat_j = (N[:, None] * rate).sum(axis=0)          # estimated area of ref class j
    P = np.full(q, np.nan)
    se_P = np.full(q, np.nan)
    for j in range(q):
        if N_hat_j[j] <= 0:
            continue
        P[j] = N[j] * rate[j, j] / N_hat_j[j] if N[j] > 0 else 0.0
        term_j = (N[j] ** 2 * (1 - P[j]) ** 2 * U[j] * (1 - U[j]) / (ne[j] - 1)
                  if n_i[j] > 0 else 0.0)
        term_rest = 0.0
        for i in range(q):
            if i == j:
                continue
            r = rate[i, j]
            term_rest += N[i] ** 2 * r * (1 - r) / (ne[i] - 1)
        var_P = (term_j + P[j] ** 2 * term_rest) / (N_hat_j[j] ** 2)
        se_P[j] = float(np.sqrt(max(var_P, 0.0)))

    # Same boundary problem as UA: a class no other stratum ever claimed gets
    # PA = 100% +/- 0.00. Fall back to Wilson on the stratum's effective n.
    pa_lo = np.clip(P - Z95 * se_P, 0, 1)
    pa_hi = np.clip(P + Z95 * se_P, 0, 1)
    for j in range(q):
        if np.isfinite(P[j]) and (P[j] <= 0 or P[j] >= 1):
            pa_lo[j], pa_hi[j] = wilson_ci(P[j], ne[j])

    # --- Area estimates ---------------------------------------------------
    p_dot_j = p_ij.sum(axis=0)
    se_p_j = np.sqrt(np.clip(
        (W[:, None] ** 2 * rate * (1 - rate) / (ne[:, None] - 1)).sum(axis=0),
        0, None))
    area_hat = A_tot * p_dot_j
    area_se = A_tot * se_p_j

    per_class = pd.DataFrame({
        "class_id": labels,
        "class": [C.CLASS_NAMES.get(l, str(l)) for l in labels],
        "map_area": A,
        "W_i": W,
        "n_i": n_i.astype(int),
        "n_eff_i": np.round(ne, 1),
        "UA": U,
        "UA_SE": se_U,
        "UA_CI_low": ua_lo,
        "UA_CI_high": ua_hi,
        "PA": P,
        "PA_SE": se_P,
        "PA_CI_low": pa_lo,
        "PA_CI_high": pa_hi,
        "area_estimate": area_hat,
        "area_SE": area_se,
        "area_CI_low": np.clip(area_hat - Z95 * area_se, 0, None),
        "area_CI_high": area_hat + Z95 * area_se,
        "area_margin": Z95 * area_se,
    })

    overall = {
        "OA": OA,
        "OA_SE": se_OA,
        "OA_CI_low": max(OA - Z95 * se_OA, 0.0),
        "OA_CI_high": min(OA + Z95 * se_OA, 1.0),
        "n_total": int(n_i.sum()),
        "n_eff_total": float(ne.sum()),
        "total_area": A_tot,
    }

    names = [C.CLASS_NAMES.get(l, str(l)) for l in labels]
    return {
        "overall": overall,
        "per_class": per_class,
        "matrix_counts": pd.DataFrame(n_ij.astype(int), index=names, columns=names),
        "matrix_proportions": pd.DataFrame(p_ij, index=names, columns=names),
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Area weights
# ---------------------------------------------------------------------------
def load_areas(path, labels, sample_counts=None, model=None):
    """Mapped area per class, in sq.km.

    The weights W_i must come from the map being assessed, so when
    map_areas.json carries a per-model block (classify_map.py writes one for
    every classifier) the block matching `model` is used. Falling back to the
    top-level areas for a different model would weight, say, LightGBM's
    confusion matrix by SVM's map - and the two disagree by a factor of three
    on swamp, which is more than enough to move the area estimates outside
    their own confidence intervals.

    Falls back to sample proportions if no area file exists, which makes the
    estimator collapse back to the ordinary unweighted one - useful only as a
    smoke test.
    """
    if path:
        raw = json.loads(open(path).read())
        if raw.get("PLACEHOLDER"):
            sys.exit(
                f"{path} is the placeholder written during development, not "
                f"real mapped areas.\nRun `python src/classify_map.py` to "
                f"overwrite it with the real ones."
            )

        block, src = raw, f"mapped areas from {path}"
        per_model = raw.get("models") or {}
        if model is not None and model in per_model:
            block = per_model[model]
            src = f"mapped areas from {path} [{model}]"
        elif model is not None and per_model:
            sys.exit(
                f"{path} has no mapped areas for '{model}'. It covers "
                f"{list(per_model)}.\nRe-run `python src/classify_map.py` so "
                f"every model in the prediction table has its own map."
            )

        areas = {int(k): float(v) for k, v in
                 (block.get("areas_km2", block)).items()}
        missing = [l for l in labels if l not in areas]
        if missing:
            sys.exit(f"{path} has no area for class(es) {missing}.")
        return areas, src

    print("\n*** NO MAP AREAS SUPPLIED ***\n"
          "    Falling back to sample proportions as weights, which makes this\n"
          "    identical to the ordinary unweighted confusion matrix and\n"
          "    defeats the point of the estimator.\n"
          "    Run `python src/classify_map.py` first to produce\n"
          "    results/tables/map_areas.json.\n")
    counts = sample_counts if sample_counts is not None else {}
    return {int(k): float(v) for k, v in counts.items()}, "SAMPLE PROPORTIONS (placeholder)"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def fmt_pct(x):
    return "  n/a " if not np.isfinite(x) else f"{100*x:6.2f}"


def report(res, model, weight_source, clustered):
    o = res["overall"]
    print(f"\n{'='*74}\n{model} - Olofsson et al. (2014) stratified estimator\n{'='*74}")
    print(f"Weights: {weight_source}")
    print(f"Total mapped area: {o['total_area']:.2f} sq.km")
    print(f"Sample units: {o['n_total']}"
          + (f"   effective: {o['n_eff_total']:.0f}" if clustered else ""))
    print(f"\nOverall accuracy: {100*o['OA']:.2f}% "
          f"+/- {100*Z95*o['OA_SE']:.2f}  "
          f"(95% CI {100*o['OA_CI_low']:.2f} - {100*o['OA_CI_high']:.2f})")

    t = res["per_class"]

    def rng(lo, hi):
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return f"{'n/a':>15}"
        return f"{100*lo:6.1f}-{100*hi:<6.1f}".rjust(15)

    print("\nPer class:")
    print(f"{'class':<20}{'W_i':>7}{'n_i':>6}{'n_eff':>7}"
          f"{'UA %':>7}{'95% CI':>15}{'PA %':>7}{'95% CI':>15}"
          f"{'area':>8}{'+/-':>7}")
    for _, r in t.iterrows():
        print(f"{r['class']:<20}{r['W_i']:>7.3f}{int(r['n_i']):>6}"
              f"{r['n_eff_i']:>7.1f}"
              f"{fmt_pct(r['UA']):>7}{rng(r['UA_CI_low'], r['UA_CI_high'])}"
              f"{fmt_pct(r['PA']):>7}{rng(r['PA_CI_low'], r['PA_CI_high'])}"
              f"{r['area_estimate']:>8.2f}{r['area_margin']:>7.2f}")

    print("\nNote: Olofsson et al. recommend against reporting Kappa - it is\n"
          "      not informative once accuracies are area-weighted, and its\n"
          "      chance-correction baseline is not meaningful for maps.")

    thin = t[t["n_eff_i"] < 10]
    if len(thin):
        print(f"\nWARNING: {list(thin['class'])} have an effective sample size\n"
              f"         below 10. Their PA/UA intervals are so wide as to be\n"
              f"         uninformative. This is the quantitative case for\n"
              f"         enlarging the reference sample.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(C.INTERIM / "test_predictions.csv"))
    ap.add_argument("--areas", default=None,
                    help="JSON of mapped area per class, from classify_map.py "
                         "(default: results/tables/map_areas.json if present)")
    ap.add_argument("--model", default=None,
                    help="restrict to one model column (default: all)")
    ap.add_argument("--cluster-col", default="poly_id")
    ap.add_argument("--no-cluster-correction", action="store_true",
                    help="use nominal n_i. Standard errors will be far too "
                         "narrow on polygon-derived samples.")
    ap.add_argument("--reference", default=None,
                    help="completed reference_points CSV from "
                         "draw_reference_sample.py (columns map_class, "
                         "reference_class). This is the real protocol: an "
                         "unclustered probability sample, so no design-effect "
                         "correction is applied.")
    args = ap.parse_args()

    if args.reference:
        df = pd.read_csv(args.reference)
        need = {"map_class", "reference_class"}
        if not need <= set(df.columns):
            sys.exit(f"{args.reference} must have columns {sorted(need)}.")
        df = df[df["reference_class"].notna()]
        df = df[df["reference_class"].astype(str).str.strip() != ""]
        if df.empty:
            sys.exit("No interpreted points - reference_class is empty "
                     "everywhere. Fill it in first.")
        df = df.rename(columns={"reference_class": "y_true",
                                "map_class": "map"})
        df["y_true"] = df["y_true"].astype(int)
        df["map"] = df["map"].astype(int)
        args.preds = args.reference
        # A stratified random sample has one unit per cluster by construction.
        args.no_cluster_correction = True
        print(f"Reference sample: {len(df)} interpreted points "
              f"(probability sample - no design-effect correction needed)")
    else:
        try:
            df = pd.read_csv(args.preds)
        except FileNotFoundError:
            sys.exit(f"{args.preds} not found. "
                     f"Run `python src/train_compare.py` first.")

    if "y_true" not in df.columns:
        sys.exit(f"{args.preds} has no y_true column.")

    reserved = {"y_true", "poly_id", "lon", "lat", "x", "y", "row", "col",
                "point_id", "map_class_name", "interpreter", "confidence",
                "notes"}
    models = ([args.model] if args.model
              else [c for c in df.columns if c not in reserved])
    models = [m for m in models if pd.api.types.is_numeric_dtype(df[m])]
    if not models:
        sys.exit("No model prediction columns found.")

    areas_path = args.areas
    if areas_path is None:
        default = C.TABLES / "map_areas.json"
        areas_path = str(default) if default.exists() else None

    labels = sorted(set(df["y_true"]) | set(pd.unique(df[models].values.ravel())))
    fallback_counts = df["y_true"].value_counts().to_dict()

    clustered = (not args.no_cluster_correction
                 and args.cluster_col in df.columns)
    if not clustered and not args.no_cluster_correction:
        print(f"NOTE: no '{args.cluster_col}' column in {args.preds}; "
              f"variances will assume independent pixels.\n"
              f"      Re-run train_compare.py to regenerate it with poly_id.")

    all_rows, summary = [], []
    weight_source = None
    for model in models:
        areas, weight_source = load_areas(areas_path, labels,
                                          fallback_counts, model=model)
        work = df[["y_true", model]].copy()
        if clustered:
            work[args.cluster_col] = df[args.cluster_col]
            work["correct"] = (work["y_true"] == work[model]).astype(int)
            deff = design_effects(work, model, "correct", args.cluster_col)
            n_i = work.groupby(model).size().to_dict()
            n_eff = {int(k): max(2.0, n_i[k] / deff.get(int(k), 1.0))
                     for k in n_i}
        else:
            n_eff = None

        res = olofsson(work[model], work["y_true"], areas,
                       labels=labels, n_eff=n_eff)
        report(res, model, weight_source, clustered)

        tbl = res["per_class"].copy()
        tbl.insert(0, "model", model)
        all_rows.append(tbl)
        summary.append({"model": model, **res["overall"]})

        safe = model.replace(" ", "_").replace("(", "").replace(")", "")
        res["matrix_proportions"].to_csv(
            C.TABLES / f"olofsson_matrix_{safe}.csv")

    pd.concat(all_rows).to_csv(C.TABLES / "olofsson_per_class.csv", index=False)
    pd.DataFrame(summary).to_csv(C.TABLES / "olofsson_overall.csv", index=False)
    print(f"\nSaved -> {C.TABLES}/olofsson_*.csv")
    if weight_source and weight_source.startswith("SAMPLE"):
        print("These numbers are NOT usable in the paper until real map areas "
              "exist. Run: python src/classify_map.py")


if __name__ == "__main__":
    main()
