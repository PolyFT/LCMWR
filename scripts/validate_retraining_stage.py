#!/usr/bin/env python3
"""Acceptance gates between expensive LCMWR retraining stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASKS = {"loi": ("LOI.csv", "LOI"), "t5": ("T5.csv", "T5"), "tg": ("Tg.csv", "Tg"), "ul94": ("UL-94.csv", "UL-94")}
DISPLAY = {"loi": "LOI", "t5": "T5", "tg": "Tg", "ul94": "UL94"}
MODELS = {"MLP", "SVR/SVM", "Ridge", "Lasso", "KNN", "DecisionTree", "LightGBM", "GradientBoosting", "RandomForest", "ExtraTrees", "XGBoost"}
RULE = "hierarchical_mole_internal_mass_blend_v1"
CACHE = "weighted_feature_matrix_v4_hierarchical_composition"


def fraction_sum(value):
    values = json.loads(value)
    return float(sum(values))


def validate_features():
    expected = json.loads((ROOT / "reproducibility/composition_audit/composition_audit_summary.json").read_text(encoding="utf-8"))["task_summary"]
    for task, (filename, target) in TASKS.items():
        out = ROOT / "results" / f"{task}_motif_select"
        old = out / f"{task}_vocab_weighted_v8.pkl"
        if old.exists(): raise RuntimeError(f"old v8 cache exists: {old}")
        cache_path = out / f"{task}_vocab_weighted_v9.pkl"
        payload = joblib.load(cache_path)
        if payload.get("cache_version") != CACHE or payload.get("blend_rule_version") != RULE:
            raise RuntimeError(f"{task}: wrong cache metadata")
        processed = pd.read_csv(out / f"{task}_vocab_processed_data.csv")
        matrix = pd.read_csv(out / f"{task}_vocab_weighted_feature_matrix.csv")
        audit = pd.read_csv(ROOT / "results/data_quality" / f"{task}_composition_audit.csv")
        if len(processed) != expected[task]["valid_rows"] or len(audit) != expected[task]["rows_with_target"]:
            raise RuntimeError(f"{task}: audit/model row count mismatch")
        if len(processed) != len(matrix) or list(processed["source_record_id"]) != list(audit.loc[audit["composition_valid"].eq(True), "source_record_id"]):
            raise RuntimeError(f"{task}: diagnostic/feature/source ID alignment mismatch")
        if not np.isfinite(matrix.to_numpy(dtype=float)).all():
            raise RuntimeError(f"{task}: nonfinite feature matrix")
        for column in ("copolymer_mole_fractions", "blend_mole_fractions", "blend_weight_fractions"):
            sums = processed[column].map(fraction_sum)
            bound = np.finfo(float).eps * 32
            if ((sums - 1).abs() > bound).any():
                raise RuntimeError(f"{task}: {column} does not sum to one at machine precision")
        source = pd.read_csv(ROOT / "dataset" / filename)
        aligned_target = source.loc[processed["source_row_index"].astype(int), target].astype(str).reset_index(drop=True)
        if aligned_target.tolist() != processed[target].astype(str).tolist():
            raise RuntimeError(f"{task}: target alignment mismatch")
    ul94 = pd.read_csv(ROOT / "reproducibility/composition_audit/ul94_mix_mol_33_row_audit.csv")
    if len(ul94) != 33 or not ul94["valid"].eq(True).all():
        raise RuntimeError("UL94 33-row mix_mol gate failed")
    print("feature gate passed for all four tasks")


def validate_selection():
    for task in TASKS:
        out = ROOT / "results" / f"{task}_motif_select"
        search = pd.read_csv(out / "plots_data/improved_750_complete_results.csv")
        if len(search) != 750:
            raise RuntimeError(f"{task}: expected 750 search combinations, found {len(search)}")
        matrix = pd.read_csv(out / "best_improved_final_features_matrix.csv")
        with_target = pd.read_csv(out / "best_improved_final_features_with_target.csv")
        if matrix.empty or len(matrix) != len(with_target):
            raise RuntimeError(f"{task}: final selected matrix invalid")
    print("750-combination selection gate passed for all four tasks")


def validate_models():
    for task in TASKS:
        path = ROOT / "results/model_compare" / DISPLAY[task] / "model_performance_summary.csv"
        summary = pd.read_csv(path)
        if set(summary["model"]) != MODELS or len(summary) != 11:
            raise RuntimeError(f"{task}: candidate model set changed")
        failed = summary.loc[~summary["status"].eq("success"), ["model", "status", "error_message"]]
        if not failed.empty:
            raise RuntimeError(f"{task}: unsuccessful model evaluations: {failed.to_dict('records')}")
        config = json.loads((path.parent / "run_config.json").read_text(encoding="utf-8"))
        for key, value in (("random_state", 48), ("outer_cv_splits_requested", 5), ("inner_cv_splits_requested", 3), ("n_iter_search", 20)):
            if config.get(key) != value:
                raise RuntimeError(f"{task}: {key} changed: {config.get(key)}")
    print("nested model-comparison gate passed for all four tasks")


def validate_final():
    models = pd.read_csv(ROOT / "results/model_compare/actual_best_oof/actual_best_models.csv")
    if set(models["task"]) != set(DISPLAY.values()): raise RuntimeError("actual-best task set incomplete")
    figure_dir = ROOT / "results/model_compare/figure4_oof"
    expected_figures = {
        "figure4c_tg_oof.svg",
        "figure4d_t5_oof.svg",
        "figure4e_loi_oof.svg",
        "figure4f_ul94_oof_roc.svg",
    }
    actual_figures = {path.name for path in figure_dir.glob("figure4*.svg")}
    if actual_figures != expected_figures:
        raise RuntimeError(f"wrong Figure 4 SVG set: expected {sorted(expected_figures)}, found {sorted(actual_figures)}")
    for task in TASKS:
        directory = ROOT / "results/interpretability" / DISPLAY[task]
        suffix = "UL94_V-0" if task == "ul94" else DISPLAY[task]
        for path in (directory / f"shap_values_{suffix}.npy", directory / f"shap_importance_ranking_{suffix}.csv", directory / f"shap_summary_plot_{DISPLAY[task]}.svg"):
            if not path.is_file() or path.stat().st_size == 0: raise RuntimeError(f"missing final SHAP output: {path}")
        values = np.load(directory / f"shap_values_{suffix}.npy")
        if not np.isfinite(values).all(): raise RuntimeError(f"{task}: nonfinite SHAP values")
    print("final OOF/Figure/SHAP gate passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("features", "selection", "models", "final"))
    args = parser.parse_args()
    globals()[f"validate_{args.stage}"]()


if __name__ == "__main__":
    main()
