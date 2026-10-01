"""Step 5: where the errors actually are, and why.

Addresses the editor's request for a deeper discussion of misclassification
rather than a bare confusion matrix. Three questions are answered:

1. **Are the errors spread out or concentrated?** If a class's errors all come
   from one or two polygons, the honest description is not "the classifier
   confuses swamp with water" but "one particular swamp polygon is spectrally
   water". Those call for completely different sentences in the Discussion, and
   only the second one is fixable by relabelling.

2. **Which class pairs are separable at all with four bands?** Jeffries-Matusita
   distance on the training spectra gives a model-free answer. JM is bounded on
   [0, 2]; the conventional reading is that above ~1.9 a pair is separable,
   1.0-1.9 is marginal, and below 1.0 no classifier will reliably split them.
   If a pair scores low, the fault is in the band set or the class definitions,
   not in the choice of CART versus XGBoost - which is worth establishing given
   that no classifier significantly outperformed any other.

3. **Are any training polygons mislabelled?** Each polygon's mean spectrum is
   compared, by Mahalanobis distance, against every class centroid. A polygon
   that sits closer to another class than to its own is flagged. This is a
   suggestion for visual re-inspection, not an automatic correction.

Inputs:  data/samples/taipei_samples.csv, data/interim/test_predictions.csv
Outputs: results/tables/error_concentration.csv
         results/tables/separability_jm.csv
         results/tables/suspect_polygons.csv
         results/tables/confusion_pairs.csv
         results/figures/separability_heatmap.png (if matplotlib is available)

Usage:
    python src/error_analysis.py
    python src/error_analysis.py --model "SVM (RBF)"
"""

import argparse
import itertools

import numpy as np
import pandas as pd

import config as C

NON_MODEL_COLS = {"y_true", "poly_id", "split", "class", "Id", "lon", "lat"}


# ----------------------------------------------------------------- separability

def bhattacharyya(mu1, cov1, mu2, cov2):
    """Bhattacharyya distance between two multivariate normals.

    B = 1/8 (m1-m2)' S^-1 (m1-m2) + 1/2 ln( |S| / sqrt(|S1||S2|) ),
    with S the average covariance. The first term measures separation of the
    means, the second separation of the shapes, so two classes with identical
    means but different spread still register as partly separable.
    """
    d = np.asarray(mu1, float) - np.asarray(mu2, float)
    S = (np.asarray(cov1, float) + np.asarray(cov2, float)) / 2.0

    # Ridge the covariances: with four highly correlated Sentinel-2 bands the
    # pooled matrix can be near-singular, and an unregularised inverse turns a
    # marginal pair into a spuriously perfect one.
    eps = 1e-8 * np.trace(S) / S.shape[0]
    S = S + eps * np.eye(S.shape[0])
    c1 = np.asarray(cov1, float) + eps * np.eye(S.shape[0])
    c2 = np.asarray(cov2, float) + eps * np.eye(S.shape[0])

    term1 = float(d @ np.linalg.solve(S, d)) / 8.0
    sign_s, logdet_s = np.linalg.slogdet(S)
    sign_1, logdet_1 = np.linalg.slogdet(c1)
    sign_2, logdet_2 = np.linalg.slogdet(c2)
    if min(sign_s, sign_1, sign_2) <= 0:
        return float(term1)
    term2 = 0.5 * (logdet_s - 0.5 * (logdet_1 + logdet_2))
    return float(term1 + term2)


def jeffries_matusita(b):
    """JM = 2(1 - exp(-B)). Saturates at 2, so it does not reward a pair for
    being astronomically separable - which is what makes it readable."""
    return float(2.0 * (1.0 - np.exp(-b)))


def separability_table(df, feats, class_col="Id"):
    stats = {}
    for cls, grp in df.groupby(class_col):
        X = grp[feats].to_numpy(float)
        if len(X) < len(feats) + 1:
            continue
        stats[int(cls)] = (X.mean(0), np.cov(X, rowvar=False))

    rows = []
    for a, b in itertools.combinations(sorted(stats), 2):
        bd = bhattacharyya(*stats[a], *stats[b])
        jm = jeffries_matusita(bd)
        rows.append({
            "class_a": a, "name_a": C.CLASS_NAMES.get(a, a),
            "class_b": b, "name_b": C.CLASS_NAMES.get(b, b),
            "bhattacharyya": round(bd, 4),
            "jeffries_matusita": round(jm, 4),
            "verdict": ("separable" if jm >= 1.9
                        else "marginal" if jm >= 1.0
                        else "not separable"),
        })
    return pd.DataFrame(rows).sort_values("jeffries_matusita")


# ---------------------------------------------------------- error concentration

def error_concentration(val, models, class_col="Id", cluster_col="poly_id"):
    """For each (class, model): how many polygons hold the errors, and what
    share of them sits in the single worst polygon."""
    rows = []
    for cls, grp in val.groupby(class_col):
        for m in models:
            wrong = grp[grp[m] != grp[class_col]]
            n_err = len(wrong)
            if n_err == 0:
                rows.append({
                    "class_id": int(cls), "class": C.CLASS_NAMES.get(int(cls), cls),
                    "model": m, "n_val_px": len(grp),
                    "n_val_polygons": grp[cluster_col].nunique(),
                    "n_errors": 0, "n_polygons_with_errors": 0,
                    "worst_polygon": "", "share_in_worst_polygon": 0.0,
                    "dominant_confusion": "",
                })
                continue
            by_poly = wrong.groupby(cluster_col).size().sort_values(ascending=False)
            top_conf = wrong[m].value_counts().idxmax()
            rows.append({
                "class_id": int(cls), "class": C.CLASS_NAMES.get(int(cls), cls),
                "model": m, "n_val_px": len(grp),
                "n_val_polygons": grp[cluster_col].nunique(),
                "n_errors": n_err,
                "n_polygons_with_errors": int(by_poly.size),
                "worst_polygon": int(by_poly.index[0]),
                "share_in_worst_polygon": round(by_poly.iloc[0] / n_err, 3),
                "dominant_confusion": C.CLASS_NAMES.get(int(top_conf), top_conf),
            })
    return pd.DataFrame(rows)


def confusion_pairs(val, models, class_col="Id"):
    """Which class pairs are confused, aggregated over every model, so a
    reviewer can see that a confusion is systematic rather than one model's
    quirk."""
    rows = []
    for a, b in itertools.permutations(sorted(val[class_col].unique()), 2):
        sub = val[val[class_col] == a]
        counts = [int((sub[m] == b).sum()) for m in models]
        if sum(counts) == 0:
            continue
        rows.append({
            "true": C.CLASS_NAMES.get(int(a), a),
            "predicted": C.CLASS_NAMES.get(int(b), b),
            "n_models_affected": int(sum(c > 0 for c in counts)),
            "median_n_px": float(np.median(counts)),
            "min_n_px": min(counts), "max_n_px": max(counts),
            "pct_of_class_median": round(100 * np.median(counts) / len(sub), 2),
        })
    return (pd.DataFrame(rows)
            .sort_values("median_n_px", ascending=False)
            .reset_index(drop=True))


# ------------------------------------------------------------ suspect polygons

def suspect_polygons(df, feats, class_col="Id", cluster_col="poly_id"):
    """Flag polygons whose mean spectrum is closer to another class centroid.

    Mahalanobis rather than Euclidean, because the bands are strongly
    correlated and a Euclidean nearest centroid would mostly rank polygons by
    overall brightness.
    """
    cent, icov = {}, {}
    for cls, grp in df.groupby(class_col):
        X = grp[feats].to_numpy(float)
        if len(X) < len(feats) + 1:
            continue
        cent[int(cls)] = X.mean(0)
        cov = np.cov(X, rowvar=False)
        cov = cov + 1e-8 * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        icov[int(cls)] = np.linalg.pinv(cov)

    rows = []
    for (cls, pid), grp in df.groupby([class_col, cluster_col]):
        cls = int(cls)
        if cls not in cent:
            continue
        mu = grp[feats].to_numpy(float).mean(0)
        dists = {c: float(np.sqrt(max(0.0, (mu - cent[c]) @ icov[c] @ (mu - cent[c]))))
                 for c in cent}
        nearest = min(dists, key=dists.get)
        rows.append({
            "poly_id": int(pid),
            "labelled_class": C.CLASS_NAMES.get(cls, cls),
            "n_px": len(grp),
            "split": grp["split"].iloc[0] if "split" in grp else "",
            "d_own": round(dists[cls], 2),
            "nearest_class": C.CLASS_NAMES.get(nearest, nearest),
            "d_nearest": round(dists[nearest], 2),
            "flagged_wrong_class": nearest != cls,
        })
    out = pd.DataFrame(rows)

    # A polygon can drive most of a class's errors without ever being closer to
    # another centroid: if its own class has a wide covariance, that class will
    # happily absorb an outlier. Polygon 77 (Forest) is the case in point - it
    # produces 97% of the forest errors yet stays nominally nearest to forest.
    # So flag within-class outliers separately, relative to the class median.
    med = out.groupby("labelled_class")["d_own"].transform("median")
    out["d_own_vs_class_median"] = (out["d_own"] / med.replace(0, np.nan)).round(2)
    out["rank_in_class"] = (out.groupby("labelled_class")["d_own"]
                            .rank(ascending=False, method="min").astype(int))
    out["flagged_outlier"] = out["d_own_vs_class_median"] >= 2.0
    out["flagged"] = out["flagged_wrong_class"] | out["flagged_outlier"]

    return out.sort_values(["flagged", "d_own"], ascending=[False, False])


# ---------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None,
                    help="restrict the concentration table to one model")
    args = ap.parse_args()

    df = pd.read_csv(C.SAMPLE_CSV)
    feats = [c for c in df.columns
             if c not in NON_MODEL_COLS | {"Class", "split", "poly_id", "lon", "lat"}]
    print(f"Features: {feats}")

    pred_csv = C.INTERIM / "test_predictions.csv"
    if not pred_csv.exists():
        raise SystemExit(f"{pred_csv} not found. Run src/train_compare.py first.")
    preds = pd.read_csv(pred_csv)

    val = df[df["split"] == "val"].reset_index(drop=True)
    if len(val) != len(preds):
        raise SystemExit(
            f"row mismatch: {len(val)} validation rows vs {len(preds)} predictions. "
            "Re-run src/train_compare.py so the two files agree.")
    if not (val[C.CLASS_PROPERTY].to_numpy() == preds["y_true"].to_numpy()).all():
        raise SystemExit("y_true in test_predictions.csv does not match the "
                         "validation labels - the files are out of sync.")

    models = [c for c in preds.columns if c not in NON_MODEL_COLS]
    if args.model:
        models = [m for m in models if m == args.model] or models
    val = val.join(preds[models])

    # 1. separability, computed on the TRAINING split only so it describes the
    #    class definitions rather than the validation outcome
    train = df[df["split"] == "train"]
    sep = separability_table(train, feats, C.CLASS_PROPERTY)
    sep.to_csv(C.TABLES / "separability_jm.csv", index=False)
    print("\n=== Spectral separability (Jeffries-Matusita, training split) ===")
    print(sep.to_string(index=False))
    weak = sep[sep["jeffries_matusita"] < 1.9]
    if len(weak):
        print(f"\n{len(weak)} of {len(sep)} class pairs fall below the "
              f"conventional JM = 1.9 separability threshold.")

    # 2. where the errors sit
    conc = error_concentration(val, models, C.CLASS_PROPERTY)
    conc.to_csv(C.TABLES / "error_concentration.csv", index=False)
    print("\n=== Error concentration ===")
    print(conc.to_string(index=False))

    focal = conc[(conc.n_errors > 0) & (conc.share_in_worst_polygon >= 0.8)]
    if len(focal):
        print("\nErrors concentrated in a single polygon (>=80% of the class's "
              "errors):")
        for r in focal.itertuples():
            print(f"  {r._2:18s} {r.model:15s} {r.n_errors:3d} errors, "
                  f"{r.share_in_worst_polygon:.0%} in polygon {r.worst_polygon} "
                  f"-> {r.dominant_confusion}")

    # 3. systematic confusions across models
    pairs = confusion_pairs(val, models, C.CLASS_PROPERTY)
    pairs.to_csv(C.TABLES / "confusion_pairs.csv", index=False)
    print("\n=== Confusions ranked by median count across models ===")
    print(pairs.to_string(index=False))

    # 4. candidate mislabelled polygons
    susp = suspect_polygons(df, feats, C.CLASS_PROPERTY)
    susp.to_csv(C.TABLES / "suspect_polygons.csv", index=False)
    flagged = susp[susp.flagged]
    n_wrong = int(susp.flagged_wrong_class.sum())
    n_out = int(susp.flagged_outlier.sum())
    print(f"\n=== Spectrally anomalous polygons ({len(flagged)} of {len(susp)}: "
          f"{n_wrong} nearer another class, {n_out} outliers within their own) ===")
    print((flagged if len(flagged) else susp.head()).to_string(index=False))
    if len(flagged):
        print("\nThese are candidates for visual re-inspection in the Code "
              "Editor, not automatic relabelling. A genuinely transitional\n"
              "surface - a flooded paddy, a reservoir margin - can legitimately "
              "sit between two centroids.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = sorted(train[C.CLASS_PROPERTY].unique())
        names = [C.CLASS_NAMES.get(int(c), c) for c in labels]
        M = np.full((len(labels), len(labels)), np.nan)
        idx = {c: i for i, c in enumerate(labels)}
        for r in sep.itertuples():
            M[idx[r.class_a], idx[r.class_b]] = r.jeffries_matusita
            M[idx[r.class_b], idx[r.class_a]] = r.jeffries_matusita
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(M, vmin=0, vmax=2, cmap="viridis")
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names)
        for i in range(len(names)):
            for j in range(len(names)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            color="w" if M[i, j] < 1.4 else "k", fontsize=9)
        ax.set_title("Jeffries-Matusita separability (training spectra)")
        fig.colorbar(im, label="JM distance (2 = fully separable)")
        fig.tight_layout()
        out = C.FIGURES / "separability_heatmap.png"
        fig.savefig(out, dpi=200)
        print(f"\nFigure -> {out}")
    except ImportError:
        print("\n(matplotlib not installed - skipping the heatmap)")

    print(f"\nTables -> {C.TABLES}")


if __name__ == "__main__":
    main()
