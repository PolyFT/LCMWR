#!/usr/bin/env python3
"""Fit/explain each new best model and regenerate Figure 5/6 SHAP assets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import joblib
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from redraw_current_shap_plots import redraw_task


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from publication_fonts import register_publication_fonts

MODEL_DIR = ROOT / "results" / "model_compare" / "actual_best_oof"
TASKS = ("LOI", "T5", "UL94", "Tg")
TREE_NAMES = {"DecisionTree", "LightGBM", "GradientBoosting", "RandomForest", "ExtraTrees", "XGBoost"}
LINEAR_NAMES = {"Ridge", "Lasso"}


def transformed_matrix(estimator, X, names):
    matrix = X
    active_names = np.asarray(names, dtype=object)
    for _, transformer in estimator.steps[:-1]:
        matrix = transformer.transform(matrix)
        if hasattr(transformer, "get_support"):
            active_names = active_names[transformer.get_support()]
    return np.asarray(matrix, dtype=np.float32), active_names.tolist()


def normalize_values(values, classification):
    if isinstance(values, list):
        return np.asarray(values[1] if classification and len(values) > 1 else values[0])
    values = np.asarray(values)
    if classification and values.ndim == 3:
        return values[:, :, 1]
    return values


def explain(model_name, model, matrix, classification, seed):
    rng = np.random.default_rng(seed)
    background_size = min(100, len(matrix))
    background = matrix[rng.choice(len(matrix), size=background_size, replace=False)]
    if model_name in TREE_NAMES:
        try:
            if classification:
                explainer = shap.TreeExplainer(
                    model, data=background, model_output="probability", feature_perturbation="interventional"
                )
            else:
                explainer = shap.TreeExplainer(model)
            return normalize_values(explainer.shap_values(matrix), classification), "TreeExplainer"
        except Exception as exc:
            print(f"TreeExplainer probability path failed for {model_name}; using permutation: {exc}", flush=True)
    if model_name in LINEAR_NAMES and not classification:
        explainer = shap.LinearExplainer(model, background)
        return normalize_values(explainer.shap_values(matrix), False), "LinearExplainer"
    predict = (lambda values: model.predict_proba(values)[:, 1]) if classification else model.predict
    explainer = shap.Explainer(predict, background, algorithm="permutation", seed=seed)
    values = explainer(matrix, max_evals=max(2 * matrix.shape[1] + 1, 11)).values
    return normalize_values(values, False), "PermutationExplainer"


def run_task(task):
    payload = joblib.load(MODEL_DIR / f"{task}_actual_best_full_fit.joblib")
    estimator = payload["estimator"]
    model = estimator.steps[-1][1]
    matrix, feature_names = transformed_matrix(estimator, payload["X"], payload["feature_names"])
    classification = task == "UL94"
    values, explainer_name = explain(payload["best_model"], model, matrix, classification, payload["random_state"])
    if values.shape != matrix.shape:
        raise RuntimeError(f"{task}: SHAP shape {values.shape} != feature shape {matrix.shape}")
    if not np.isfinite(values).all():
        raise RuntimeError(f"{task}: nonfinite SHAP values")

    output_dir = ROOT / "results" / "interpretability" / task
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "UL94_V-0" if classification else task
    feature_frame = pd.DataFrame(matrix, columns=feature_names)
    shap_frame = pd.DataFrame(values, columns=feature_names)
    feature_frame.to_csv(output_dir / f"X_shap_{task}.csv", index=False)
    shap_frame.to_csv(output_dir / f"shap_values_{suffix}.csv", index=False)
    np.save(output_dir / f"shap_values_{suffix}.npy", values)
    ranking = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_SHAP": np.mean(np.abs(values), axis=0),
        "mean_SHAP": np.mean(values, axis=0),
    }).sort_values("mean_abs_SHAP", ascending=False, ignore_index=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(output_dir / f"shap_importance_ranking_{suffix}.csv", index=False)
    ranking.head(20).to_csv(output_dir / f"shap_importance_top_features_{task}.csv", index=False)

    # Keep the original paper layout used by the task notebooks.  The plotting
    # helper rereads and verifies the just-written arrays before exporting SVGs.
    redraw_task(task)

    prediction = estimator.predict(payload["X"])
    pd.DataFrame({"prediction": prediction}).to_csv(output_dir / f"y_pred_shap_{task}.csv", index=False)
    pd.DataFrame({"target": np.asarray(payload["y_display"])}).to_csv(output_dir / f"y_shap_{task}.csv", index=False)
    metadata = {
        "task": task,
        "best_model": payload["best_model"],
        "best_params": payload["params"],
        "explainer": explainer_name,
        "classification_output": "P(V-0)" if classification else None,
        "sample_count": int(matrix.shape[0]),
        "feature_count": int(matrix.shape[1]),
        "random_state": payload["random_state"],
        "input_path": payload["input_path"],
        "input_sha256": payload["input_sha256"],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=[task.lower() for task in TASKS])
    args = parser.parse_args()
    register_publication_fonts()
    plt.rcParams.update({"font.family": "Arial", "font.weight": "bold", "axes.labelweight": "bold", "svg.fonttype": "path"})
    run_task(next(task for task in TASKS if task.lower() == args.task))


if __name__ == "__main__":
    main()
