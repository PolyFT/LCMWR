#!/usr/bin/env python3
"""Nested 5×3 seed sweep for the current LOI best feature set.

The feature matrix is fixed.  Each seed controls a 5-fold outer KFold and a
3-fold inner KFold parameter search, matching LOI_motif_select.ipynb and the
standalone LOI model-comparison notebook exactly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results" / "loi_motif_select"
PLOTS_DIR = RESULT_DIR / "plots_data"
# The LOI notebook currently uses the blend-fixed v8 feature cache.
CACHE_FILE = RESULT_DIR / "loi_vocab_weighted_v8.pkl"
FINAL_FEATURES_FILE = RESULT_DIR / "best_improved_final_features_matrix.csv"
SEARCH_RESULTS_FILE = PLOTS_DIR / "improved_750_complete_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=100)
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="RandomizedSearchCV samples per outer fold (default: 20).",
    )
    parser.add_argument("--outer-cv", type=int, default=5, help="Outer KFold splits.")
    parser.add_argument("--inner-cv", type=int, default=3, help="Inner KFold splits.")
    parser.add_argument("--cv-n-jobs", type=int, default=2)
    parser.add_argument("--xgb-n-jobs", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=PLOTS_DIR / "seed_sweep_nested_5x3.csv",
        help="Checkpoint CSV for the requested seed range; completed seeds are reused on the next run.",
    )
    return parser.parse_args()


def model_parameter_distributions():
    """Same discrete XGBoost search space as the model-comparison notebooks."""
    return {
        "n_estimators": [100, 300, 600],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 5, 7],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
    }


def evaluate_seed(X: np.ndarray, y: np.ndarray, seed: int, args: argparse.Namespace) -> dict:
    if args.outer_cv < 2 or args.inner_cv < 2:
        raise ValueError("--outer-cv and --inner-cv must both be at least 2")
    if len(X) < args.outer_cv:
        raise ValueError(f"Outer CV requires {args.outer_cv} samples, got {len(X)}")

    outer_splitter = KFold(n_splits=args.outer_cv, shuffle=True, random_state=seed)
    fold_r2: list[float] = []
    best_params_by_fold: list[dict] = []
    started = time.monotonic()
    for fold, (train_idx, test_idx) in enumerate(outer_splitter.split(X), start=1):
        estimator = xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=args.xgb_n_jobs,
            random_state=seed,
            verbosity=0,
        )
        inner_splitter = KFold(n_splits=args.inner_cv, shuffle=True, random_state=seed)
        try:
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=model_parameter_distributions(),
                n_iter=args.n_iter,
                scoring="r2",
                cv=inner_splitter,
                n_jobs=args.cv_n_jobs,
                random_state=seed,
                error_score="raise",
                refit=True,
                verbose=0,
            )
            search.fit(X[train_idx], y[train_idx])
            model = search.best_estimator_
            selected_params = search.best_params_
            selection_method = "random_search"
            inner_score = float(search.best_score_)
        except Exception as search_error:
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                n_jobs=args.xgb_n_jobs,
                random_state=seed,
                verbosity=0,
            )
            model.fit(X[train_idx], y[train_idx])
            selected_params = {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1}
            selection_method = "feature_selection_defaults_fallback"
            inner_score = np.nan

        fold_r2.append(r2_score(y[test_idx], model.predict(X[test_idx])))
        best_params_by_fold.append({
            "fold": fold,
            "best_params": selected_params,
            "best_score_inner_cv": inner_score,
            "selection_method": selection_method,
        })

    return {
        "seed": seed,
        "status": "ok",
        "outer_cv_folds": args.outer_cv,
        "inner_cv_folds": args.inner_cv,
        "r2_mean": float(np.mean(fold_r2)),
        "r2_std": float(np.std(fold_r2, ddof=1)) if len(fold_r2) > 1 else 0.0,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "best_params_by_fold_json": json.dumps(best_params_by_fold, sort_keys=True),
        "error": "",
    }


def load_inputs() -> tuple[np.ndarray, np.ndarray, dict]:
    for path in (CACHE_FILE, FINAL_FEATURES_FILE, SEARCH_RESULTS_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    cache = joblib.load(CACHE_FILE)
    X = pd.read_csv(FINAL_FEATURES_FILE).to_numpy(dtype=np.float32, copy=False)
    y = np.asarray(cache["target_variable"], dtype=np.float32)
    if len(X) != len(y):
        raise ValueError(f"Feature rows ({len(X)}) and target rows ({len(y)}) do not match.")
    results = pd.read_csv(SEARCH_RESULTS_FILE)
    best = results.loc[results["performance"].idxmax()]
    return X, y, {
        "freq_threshold": float(best["freq_threshold"]),
        "var_threshold": float(best["var_threshold"]),
        "mi_threshold": float(best["mi_threshold"]),
        "corr_threshold": float(best["corr_threshold"]),
        "750_search_r2": float(best["performance"]),
        "feature_count": int(best["feature_count"]),
    }


def main() -> None:
    args = parse_args()
    if args.seed_start > args.seed_end:
        raise ValueError("--seed-start must not exceed --seed-end")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    X, y, best_config = load_inputs()
    print("Best 750 thresholds:", json.dumps(best_config, ensure_ascii=False))
    print(f"Fixed final feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    print(
        f"Seed range: {args.seed_start}–{args.seed_end}; n_iter={args.n_iter}; "
        f"nested CV={args.outer_cv}×{args.inner_cv}; "
        f"CV jobs={args.cv_n_jobs}; XGB jobs={args.xgb_n_jobs}"
    )

    existing = pd.DataFrame()
    if args.output.exists():
        existing = pd.read_csv(args.output)
    completed = set(existing.get("seed", pd.Series(dtype=int)).dropna().astype(int))
    rows = existing.to_dict("records")

    for seed in range(args.seed_start, args.seed_end + 1):
        if seed in completed:
            print(f"[{seed:3d}] already completed; skipping")
            continue
        print(f"[{seed:3d}] evaluating...", flush=True)
        try:
            row = evaluate_seed(X, y, seed, args)
            print(f"[{seed:3d}] R²={row['r2_mean']:.4f} ± {row['r2_std']:.4f} ({row['elapsed_seconds']:.1f}s)", flush=True)
        except Exception as exc:
            row = {"seed": seed, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            print(f"[{seed:3d}] failed: {row['error']}", flush=True)
        rows.append(row)
        pd.DataFrame(rows).sort_values("seed").to_csv(args.output, index=False)

    result_df = pd.DataFrame(rows).sort_values("seed")
    ok = result_df.loc[result_df["status"] == "ok"].copy()
    summary_path = args.output.with_suffix(".summary.json")
    summary = {
        "best_thresholds": best_config,
        "feature_matrix_shape": list(X.shape),
        "requested_seed_range": [args.seed_start, args.seed_end],
        "evaluation_protocol": "nested_kfold_5x3_r2_v1",
        "outer_cv": args.outer_cv,
        "inner_cv": args.inner_cv,
        "n_iter": args.n_iter,
        "completed": int(len(ok)),
        "failed": int((result_df["status"] == "error").sum()),
    }
    if not ok.empty:
        best_row = ok.loc[ok["r2_mean"].idxmax()]
        summary.update({
            "r2_mean_over_seeds": float(ok["r2_mean"].mean()),
            "r2_std_over_seeds": float(ok["r2_mean"].std(ddof=1)) if len(ok) > 1 else 0.0,
            "r2_min": float(ok["r2_mean"].min()),
            "r2_max": float(ok["r2_mean"].max()),
            "best_seed": int(best_row["seed"]),
            "best_seed_r2": float(best_row["r2_mean"]),
        })
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults: {args.output}")
    print(f"Summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
