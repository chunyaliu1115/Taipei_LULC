"""Step 2 of the revision pipeline: classifier benchmark with tuning.

Addresses editor comments 1 and 2:
  - documents the train/test split and per-class sample counts
  - runs an explicit hyperparameter search for SVM, RF, CART
  - adds XGBoost, LightGBM and an MLP as modern comparators
  - reports OA, Kappa, per-class PA/UA with bootstrap confidence intervals
  - saves per-sample test predictions so src/mcnemar_test.py can run

Usage:
    python src/train_compare.py
    python src/train_compare.py --input data/samples/taipei_samples.csv
    python src/train_compare.py --quick     # smaller search, for a dry run
"""

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix)
from sklearn.model_selection import (GroupShuffleSplit, RandomizedSearchCV,
                                     StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config as C

warnings.filterwarnings("ignore", category=UserWarning)

FEATURE_EXCLUDE = {C.CLASS_PROPERTY, "Class", "split", "lon", "lat",
                   "system:index", "system_index", "id", ".geo",
                   "poly_id", C.GROUP_COLUMN, "random"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    feats = [c for c in df.columns if c not in FEATURE_EXCLUDE]
    feats = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    df = df.dropna(subset=feats + [C.CLASS_PROPERTY]).reset_index(drop=True)
    X = df[feats].to_numpy(dtype=float)
    y = df[C.CLASS_PROPERTY].to_numpy(dtype=int)
    groups = (df["poly_id"].to_numpy() if "poly_id" in df.columns
              else np.arange(len(df)))
    return X, y, groups, feats, df


def split_data(X, y, groups, df):
    """Honour the part-level split in the CSV; otherwise build one.

    Never falls back to a plain random split when polygon ids are available -
    that is the leak this whole exercise exists to remove.
    """
    if "split" in df.columns and set(df["split"].unique()) >= {"train", "val"}:
        m = (df["split"] == "train").to_numpy()
        return (X[m], X[~m], y[m], y[~m], groups[m], groups[~m],
                "part-level train/val split, stratified by class "
                "(assigned by gee_export_samples.assign_split)")

    if "poly_id" in df.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=C.TEST_SIZE,
                                random_state=C.RANDOM_SEED)
        tr, te = next(gss.split(X, y, groups))
        return (X[tr], X[te], y[tr], y[te], groups[tr], groups[te],
                f"polygon-grouped {int((1-C.TEST_SIZE)*100)}/"
                f"{int(C.TEST_SIZE*100)} split")

    print("WARNING: no poly_id column - falling back to a pixel-level split.\n"
          "         If these samples came from polygons, this leaks "
          "near-duplicate pixels across the split and will overstate accuracy.")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_SEED
    )
    n_tr = len(ytr)
    return (Xtr, Xte, ytr, yte, np.arange(n_tr), np.arange(len(yte)),
            f"pixel-level stratified {int((1-C.TEST_SIZE)*100)}/"
            f"{int(C.TEST_SIZE*100)} split (NOT group-aware)")


# ---------------------------------------------------------------------------
# Models and search spaces
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier as _XGBBase
except ImportError:                                   # optional dependency
    _XGBBase = None

if _XGBBase is not None:
    class XGBLabelSafe(_XGBBase):
        """XGBClassifier that tolerates class labels 1..5.

        Since 2.0, XGBoost rejects any target whose classes are not exactly
        0..k-1. Our labels are the GEE 'Id' values 1..5, so it refuses to fit
        with "Invalid classes inferred from unique values of `y`".

        Relabelling the data globally would be the wrong fix: config.CLASS_NAMES,
        the confusion matrices, mcnemar_test.py and olofsson.py all key off
        1..5, and a 0-based y would silently shift every class name by one -
        water would be reported as built-up. So the encoding is hidden inside
        the estimator instead: y is mapped to 0..k-1 on fit and predictions are
        mapped back on predict. XGBoost's own hyperparameter names stay intact,
        so the search grid needs no prefixing.

        Defined at module level rather than inside a factory so that joblib can
        pickle it when RandomizedSearchCV fans the search out across processes.
        """

        def fit(self, X, y, **kw):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder().fit(y)
            # _le stays None for the duration of the parent fit. XGBoost's own
            # fit reads self.classes_ back to validate the encoded y it was
            # just handed, so while it is running classes_ must still describe
            # the 0..k-1 space; only afterwards does it flip to 1..5.
            self._le = None
            super().fit(X, le.transform(y), **kw)
            self._le = le
            return self

        def predict(self, X, **kw):
            out = np.asarray(super().predict(X, **kw)).astype(int)
            return self._le.inverse_transform(out) if self._le is not None else out

        # sklearn's scorers and confusion-matrix helpers read classes_ to work
        # out the label space, so once fitted it has to report 1..5 rather than
        # the 0..4 the booster was actually trained on. It cannot simply be
        # assigned: in xgboost 3.x classes_ is a read-only property
        # (np.arange(n_classes_)), so assignment raises AttributeError.
        # Overriding it as a property covers both, and the setter absorbs the
        # assignment that xgboost 2.x still performs inside its own fit.
        @property
        def classes_(self):
            le = getattr(self, "_le", None)
            if le is not None:
                return np.asarray(le.classes_)
            stored = getattr(self, "_classes_", None)
            if stored is not None:
                return np.asarray(stored)
            return np.arange(getattr(self, "n_classes_", 0))

        @classes_.setter
        def classes_(self, value):
            self._classes_ = np.asarray(value)


def build_models(quick=False):
    seed = C.RANDOM_SEED
    models = {}

    models["CART"] = (
        DecisionTreeClassifier(random_state=seed),
        {
            "max_depth": [None, 5, 10, 15, 20, 30],
            "min_samples_leaf": [1, 2, 5, 10, 20],
            "min_samples_split": [2, 5, 10, 20],
            "criterion": ["gini", "entropy"],
            "ccp_alpha": [0.0, 1e-4, 1e-3, 1e-2],
        },
    )

    models["SVM (RBF)"] = (
        Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="rbf", random_state=seed))]),
        {
            "clf__C": np.logspace(-1, 3, 20),
            "clf__gamma": np.logspace(-4, 1, 20),
        },
    )

    models["Random Forest"] = (
        RandomForestClassifier(random_state=seed, n_jobs=-1),
        {
            "n_estimators": [10, 50, 100, 250, 500, 1000],
            "max_features": ["sqrt", "log2", None],
            "max_depth": [None, 10, 20, 30],
            "min_samples_leaf": [1, 2, 5, 10],
            "class_weight": [None, "balanced", "balanced_subsample"],
        },
    )

    if _XGBBase is not None:
        models["XGBoost"] = (
            XGBLabelSafe(random_state=seed, n_jobs=-1, tree_method="hist",
                         eval_metric="mlogloss", verbosity=0),
            {
                "n_estimators": [200, 400, 800],
                "max_depth": [3, 5, 7, 9],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "min_child_weight": [1, 3, 5],
                "reg_lambda": [0.1, 1.0, 5.0],
            },
        )
    else:
        print("  (xgboost not installed - skipping)")

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = (
            LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1),
            {
                "n_estimators": [200, 400, 800],
                "num_leaves": [15, 31, 63, 127],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "min_child_samples": [5, 10, 20, 40],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "reg_lambda": [0.0, 1.0, 5.0],
            },
        )
    except ImportError:
        print("  (lightgbm not installed - skipping)")

    models["MLP (deep)"] = (
        Pipeline([("sc", StandardScaler()),
                  ("clf", MLPClassifier(random_state=seed, max_iter=2000))]),
        {
            "clf__hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128, 64)],
            "clf__alpha": np.logspace(-5, -1, 10),
            "clf__learning_rate_init": [1e-3, 3e-3, 1e-2],
        },
    )

    if quick:
        # Thin each grid to 3 well-spread values so a dry run still exercises a
        # sensible part of the search space (taking the first 3 would leave SVM
        # with only tiny C/gamma values and produce a degenerate model).
        def thin(v, n=3):
            v = list(v)
            if len(v) <= n:
                return v
            idx = np.linspace(0, len(v) - 1, n).round().astype(int)
            return [v[i] for i in idx]
        for k, (est, grid) in models.items():
            models[k] = (est, {p: thin(v) for p, v in grid.items()})
    return models


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def bootstrap_ci(y_true, y_pred, metric, n=None, seed=None):
    n = n or C.N_BOOTSTRAP
    rng = np.random.default_rng(seed or C.RANDOM_SEED)
    idx = np.arange(len(y_true))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[s])) < 2:
            continue
        vals.append(metric(y_true[s], y_pred[s]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def per_class_table(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    # sklearn's confusion_matrix has rows = reference (y_true) and columns =
    # predicted. So the row total is the number of reference units of the
    # class and the column total is the number the map assigns to it:
    #   PA = TP / row total      (of the reference units, how many were found)
    #   UA = TP / column total   (of the mapped units, how many were right)
    # These two were the wrong way round until now, which reversed every
    # per-class UA and PA in the hold-out tables. The design-based estimates
    # in olofsson.py were unaffected - they build their own matrix with rows =
    # map, the Olofsson convention - and disagreeing with them is what
    # exposed the error.
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        n_ref = cm[i, :].sum()
        n_map = cm[:, i].sum()
        pa = tp / n_ref * 100 if n_ref else np.nan
        ua = tp / n_map * 100 if n_map else np.nan
        f1 = 2 * pa * ua / (pa + ua) if (pa + ua) else np.nan
        rows.append({
            "class_id": lab,
            "class": C.CLASS_NAMES.get(lab, str(lab)),
            "n_ref": int(n_ref),
            "n_map": int(n_map),
            "PA_%": round(pa, 2),
            "UA_%": round(ua, 2),
            "F1_%": round(f1, 2),
        })
    return pd.DataFrame(rows), cm, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(C.SAMPLE_CSV))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    X, y, groups, feats, df = load_data(args.input)
    Xtr, Xte, ytr, yte, gtr, gte, split_desc = split_data(X, y, groups, df)

    print(f"Features ({len(feats)}): {feats}")
    print(f"Split: {split_desc}  ->  train={len(ytr)}  test={len(yte)}")
    counts = pd.DataFrame({
        "class": [C.CLASS_NAMES.get(k, k) for k in sorted(set(y))],
        "train": [int((ytr == k).sum()) for k in sorted(set(y))],
        "test": [int((yte == k).sum()) for k in sorted(set(y))],
    })
    counts["total"] = counts["train"] + counts["test"]
    print("\nSample design (goes into the revised Table 2):\n", counts.to_string(index=False))
    counts.to_csv(C.TABLES / "sample_design.csv", index=False)

    # Group-aware CV so the hyperparameter search doesn't leak either.
    n_groups = len(np.unique(gtr))
    if n_groups >= C.CV_FOLDS:
        cv = StratifiedGroupKFold(n_splits=C.CV_FOLDS, shuffle=True,
                                  random_state=C.RANDOM_SEED)
        cv_groups = gtr
        print(f"CV: StratifiedGroupKFold over {n_groups} training polygons")
    else:
        cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True,
                             random_state=C.RANDOM_SEED)
        cv_groups = None
        print(f"CV: StratifiedKFold ({n_groups} groups is too few for "
              f"{C.CV_FOLDS}-fold group CV)")

    n_iter = 8 if args.quick else C.N_ITER_SEARCH

    # poly_id travels with the predictions so src/olofsson.py can correct the
    # variances for intra-polygon correlation. Without it the standard errors
    # are computed as if 1,999 pixels were 1,999 independent observations.
    summary, best_params = [], {}
    preds = {"y_true": yte.tolist(), "poly_id": np.asarray(gte).tolist()}

    for name, (est, grid) in build_models(quick=args.quick).items():
        print(f"\n--- {name} ---")
        t0 = time.perf_counter()
        search = RandomizedSearchCV(
            est, grid, n_iter=n_iter, cv=cv, scoring="accuracy",
            n_jobs=-1, random_state=C.RANDOM_SEED, refit=True, error_score="raise",
        )
        search.fit(Xtr, ytr, groups=cv_groups) if cv_groups is not None \
            else search.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        yp = search.best_estimator_.predict(Xte)
        pred_s = time.perf_counter() - t1

        oa = accuracy_score(yte, yp)
        kappa = cohen_kappa_score(yte, yp)
        oa_lo, oa_hi = bootstrap_ci(yte, yp, accuracy_score)
        k_lo, k_hi = bootstrap_ci(yte, yp, cohen_kappa_score)

        print(f"CV acc {search.best_score_:.4f} | OA {oa:.4f} "
              f"[{oa_lo:.3f}-{oa_hi:.3f}] | kappa {kappa:.4f} [{k_lo:.3f}-{k_hi:.3f}]")
        print(f"tuning {fit_s:.1f}s, inference {pred_s*1000:.1f}ms")
        print(classification_report(yte, yp, zero_division=0))

        tbl, cm, labels = per_class_table(yte, yp)
        tbl.to_csv(C.TABLES / f"perclass_{name.replace(' ', '_').replace('(', '').replace(')', '')}.csv", index=False)
        pd.DataFrame(cm, index=[C.CLASS_NAMES.get(l, l) for l in labels],
                     columns=[C.CLASS_NAMES.get(l, l) for l in labels]).to_csv(
            C.TABLES / f"confusion_{name.replace(' ', '_').replace('(', '').replace(')', '')}.csv")

        preds[name] = yp.tolist()
        best_params[name] = {k: (v.item() if hasattr(v, "item") else v)
                             for k, v in search.best_params_.items()}
        summary.append({
            "model": name,
            "cv_accuracy": round(search.best_score_, 4),
            "OA": round(oa, 4),
            "OA_CI_low": round(oa_lo, 4),
            "OA_CI_high": round(oa_hi, 4),
            "kappa": round(kappa, 4),
            "kappa_CI_low": round(k_lo, 4),
            "kappa_CI_high": round(k_hi, 4),
            "tuning_seconds": round(fit_s, 2),
            "inference_ms": round(pred_s * 1000, 2),
        })

    out = pd.DataFrame(summary).sort_values("OA", ascending=False)
    out.to_csv(C.TABLES / "model_comparison.csv", index=False)
    pd.DataFrame(preds).to_csv(C.INTERIM / "test_predictions.csv", index=False)
    (C.TABLES / "best_hyperparameters.json").write_text(json.dumps(best_params, indent=2))

    print("\n=== Summary ===")
    print(out.to_string(index=False))
    print(f"\nSaved -> {C.TABLES}")
    print("Next: python src/mcnemar_test.py")


if __name__ == "__main__":
    main()
