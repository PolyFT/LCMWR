#!/usr/bin/env python3
"""Rebuild pooled OOF outputs and full fits for each newly selected best model."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "results" / "model_compare"
OUTPUT_DIR = MODEL_DIR / "actual_best_oof"
NOTEBOOKS = {
    "LOI": ROOT / "scripts" / "model_comparison" / "loi_model_compare.ipynb",
    "Tg": ROOT / "scripts" / "model_comparison" / "tg_model_compare.ipynb",
    "T5": ROOT / "scripts" / "model_comparison" / "t5_model_compare.ipynb",
    "UL94": ROOT / "scripts" / "model_comparison" / "ul94_model_compare.ipynb",
}


def load_protocol(task):
    notebook = json.loads(NOTEBOOKS[task].read_text(encoding="utf-8"))
    module_name = f"_lcmwr_{task.lower()}_model_protocol"
    module = types.ModuleType(module_name)
    module.__file__ = str(NOTEBOOKS[task])
    module.display = lambda *_: None
    # dataclasses resolves annotations through sys.modules[cls.__module__].
    # Register the synthetic notebook module before executing its class cells.
    sys.modules[module_name] = module
    namespace = module.__dict__
    for index in range(1, 7):
        cell = notebook["cells"][index]
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            exec(compile(source, f"{NOTEBOOKS[task]}:cell-{index}", "exec"), namespace)
    return namespace


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for task in NOTEBOOKS:
        protocol = load_protocol(task)
        summary_path = MODEL_DIR / task / "model_performance_summary.csv"
        summary = pd.read_csv(summary_path)
        metric = "ROC-AUC_mean" if task == "UL94" else "R2_mean"
        usable = summary.loc[summary["status"].eq("success")].copy()
        usable[metric] = pd.to_numeric(usable[metric], errors="coerce")
        if usable[metric].notna().sum() == 0:
            raise RuntimeError(f"{task}: no successful model with finite {metric}")
        best_model = str(usable.loc[usable[metric].idxmax(), "model"])

        config = protocol["TASK_CONFIG"][task]
        X, y_raw, metadata, sample_metadata = protocol["load_task_data"](task, config)
        classification = metadata["task_type"] == "classification"
        if classification:
            y_display = y_raw.astype(str)
            y_model = pd.Series((y_display == "V-0").astype(int), index=y_raw.index)
            class_labels = ["non-V-0", "V-0"]
        else:
            y_model = pd.to_numeric(y_raw, errors="coerce")
            valid = y_model.notna()
            X, y_model = X.loc[valid], y_model.loc[valid]
            y_display = y_model.copy()
            class_labels = None

        specs = {item.name: item for item in protocol["get_model_specs"](metadata["task_type"], 2 if classification else None)}
        if best_model not in specs or not specs[best_model].available:
            raise RuntimeError(f"{task}: selected model unavailable: {best_model}")
        params_payload = json.loads((MODEL_DIR / task / "best_params.json").read_text(encoding="utf-8"))[best_model]
        fold_params = {int(item["fold"]): item["best_params"] for item in params_payload["best_params_by_fold"]}
        outer_cv = protocol["make_cv"](metadata["task_type"], y_model, protocol["OUTER_CV_SPLITS"], protocol["RANDOM_STATE"])
        predicted = np.empty(len(X), dtype=object if classification else float)
        probability = np.full(len(X), np.nan, dtype=float)
        fold_ids = np.full(len(X), -1, dtype=int)
        for fold, (train_index, valid_index) in enumerate(outer_cv.split(X, y_model), start=1):
            estimator = protocol["build_pipeline"](specs[best_model])
            estimator.set_params(**fold_params[fold])
            estimator.fit(X.iloc[train_index], y_model.iloc[train_index])
            fold_prediction = estimator.predict(X.iloc[valid_index])
            predicted[valid_index] = (
                np.asarray([class_labels[int(value)] for value in fold_prediction], dtype=object)
                if classification else np.asarray(fold_prediction, dtype=float)
            )
            if classification:
                probability[valid_index] = estimator.predict_proba(X.iloc[valid_index])[:, 1]
            fold_ids[valid_index] = fold
        if (fold_ids < 1).any() or (classification and np.isnan(probability).any()):
            raise RuntimeError(f"{task}: incomplete OOF reconstruction")

        output = sample_metadata.loc[X.index].reset_index(drop=True).copy()
        output.insert(0, "sample_index", X.index.to_numpy())
        processed_path = ROOT / "results" / f"{task.lower()}_motif_select" / f"{task.lower()}_vocab_processed_data.csv"
        processed = pd.read_csv(processed_path)
        if len(processed) != len(X):
            raise RuntimeError(f"{task}: selected matrix and processed-data rows are not aligned")
        for identifier in ("source_row_index", "source_record_id"):
            if identifier not in processed:
                raise RuntimeError(f"{task}: missing alignment field {identifier}")
            output[identifier] = processed[identifier].to_numpy()
        output["true_value"] = y_display.loc[X.index].to_numpy()
        output["outer_fold"] = fold_ids
        output["best_model"] = best_model
        output["predicted_value"] = predicted
        if classification:
            output["probability_V-0"] = probability
        output_path = OUTPUT_DIR / f"{task}_actual_best_oof.csv"
        output.to_csv(output_path, index=False, encoding="utf-8-sig")

        full_estimator = protocol["build_pipeline"](specs[best_model])
        full_estimator.set_params(**params_payload["best_params"])
        full_estimator.fit(X, y_model)
        model_path = OUTPUT_DIR / f"{task}_actual_best_full_fit.joblib"
        joblib.dump({
            "task": task,
            "best_model": best_model,
            "estimator": full_estimator,
            "X": X,
            "y_model": y_model,
            "y_display": y_display,
            "feature_names": list(X.columns),
            "params": params_payload["best_params"],
            "input_path": str(metadata["resolved_input"]),
            "input_sha256": sha256(Path(metadata["resolved_input"])),
            "random_state": protocol["RANDOM_STATE"],
        }, model_path)
        row = {
            "task": task,
            "best_model": best_model,
            "selection_metric": metric,
            "selection_metric_mean": float(usable.loc[usable["model"].eq(best_model), metric].iloc[0]),
            "n_samples": int(len(X)),
            "n_features": int(X.shape[1]),
            "oof_file": str(output_path.relative_to(ROOT)),
            "full_fit_file": str(model_path.relative_to(ROOT)),
        }
        summary_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "actual_best_models.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
