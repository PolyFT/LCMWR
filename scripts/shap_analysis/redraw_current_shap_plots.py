#!/usr/bin/env python3
"""Redraw current SHAP arrays with the repository's original paper style."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from publication_shap_style import export_publication_shap_plots


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from publication_fonts import register_publication_fonts

TASKS = ("LOI", "T5", "UL94", "Tg")

# These are the publication parameters retained in the original task notebooks.
PUBLICATION_PLOT = {
    "axis_width_in": 6.5,
    "axis_height_in": 5.2,
    "left_margin_in": 1.35,
    "bottom_margin_in": 1.05,
    "right_margin_in": 0.35,
    "top_margin_in": 0.25,
    "axis_label_size": 32,
    "y_axis_label_size": 32,
    "tick_label_size": 28,
    "spine_width": 2.8,
    "tick_width": 2.8,
    "tick_length": 6,
    "dot_size": 36,
    "alpha": 1.0,
    "transparent_background": True,
}

PUBLICATION_SUMMARY = {
    "max_display": 20,
    "width_in": 6.0,
    "base_height_in": 5.0,
    "height_per_feature_in": 0.35,
    "height_padding_in": 1.8,
    "axis_label_size": 20,
    "tick_label_size": 20,
    "feature_to_plot_gap_in": 0.25,
    "feature_label_alignment": "center",
    "tick_width": 2.5,
    "tick_length": 6,
    "spine_width": 1.2,
    "point_size": 36,
    "point_alpha": 1.0,
    "show_colorbar": True,
    "colorbar_label": "Feature Value",
    "colorbar_position": [0.875, 0.10, 0.028, 0.87],
    "feature_value_position": [0.955, 0.50],
    "high_position": [0.90, 0.955],
    "low_position": [0.90, 0.11],
    "colorbar_label_size": 20,
    "high_low_size": 20,
    "transparent_background": True,
}

TASK_STYLE = {
    "LOI": {
        "suffix": "LOI",
        "summary_x_label": "SHAP Value",
        "summary_tick_interval": None,
        "summary_x_min": -10.0,
        "summary_extra_x_ticks": (-10.0,),
    },
    "T5": {
        "suffix": "T5",
        "summary_x_label": "SHAP Value",
        "summary_tick_interval": 50.0,
    },
    "Tg": {
        "suffix": "Tg",
        "summary_x_label": "SHAP Value",
        "summary_tick_interval": 25.0,
        "summary_x_min": -50.0,
        "summary_x_max": 75.0,
        "summary_x_min_is_lower_bound": False,
    },
    "UL94": {
        "suffix": "UL94_V-0",
        "summary_x_label": "SHAP Value for V-0",
        "summary_tick_interval": 1.0,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def display_name(feature: str) -> str:
    molecule = Chem.MolFromSmiles(feature)
    return Chem.MolToSmiles(molecule) if molecule is not None else feature


def configure_matplotlib() -> None:
    register_publication_fonts()
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "pdf.fonttype": 42,
            "svg.fonttype": "path",
        }
    )


def redraw_task(task: str, *, clear_existing: bool = True) -> dict:
    style = TASK_STYLE[task]
    output_dir = ROOT / "results" / "interpretability" / task
    feature_path = output_dir / f"X_shap_{task}.csv"
    values_path = output_dir / f"shap_values_{style['suffix']}.npy"
    ranking_path = output_dir / f"shap_importance_ranking_{style['suffix']}.csv"
    protected_hashes = {path: sha256(path) for path in (feature_path, values_path, ranking_path)}

    features = pd.read_csv(feature_path)
    values = np.load(values_path)
    ranking = pd.read_csv(ranking_path)
    if values.shape != features.shape:
        raise RuntimeError(f"{task}: SHAP shape {values.shape} != feature shape {features.shape}")
    if set(ranking["feature"]) != set(features.columns) or len(ranking) != features.shape[1]:
        raise RuntimeError(f"{task}: saved SHAP ranking features do not match the current arrays")
    mean_abs_by_feature = pd.Series(np.abs(values).mean(axis=0), index=features.columns)
    saved_mean_abs = ranking["feature"].map(mean_abs_by_feature).to_numpy()
    if not np.allclose(saved_mean_abs, ranking["mean_abs_SHAP"].to_numpy(), rtol=1e-7, atol=1e-12):
        raise RuntimeError(f"{task}: saved SHAP importance values do not match the current arrays")
    # Zero-importance ties can have a different internal order depending on the
    # pandas sort implementation; only the scientific ordering is constrained.
    if np.any(np.diff(saved_mean_abs) > 1e-12):
        raise RuntimeError(f"{task}: saved SHAP ranking is not non-increasing")

    if clear_existing:
        for path in output_dir.glob(f"shap_dependence_*_{task}.svg"):
            path.unlink()

    RDLogger.DisableLog("rdApp.error")
    display_names = [display_name(feature) for feature in features.columns]
    kwargs = {
        key: value
        for key, value in style.items()
        if key != "suffix"
    }
    summary_path, plot_table = export_publication_shap_plots(
        task=task,
        output_dir=output_dir,
        features=features,
        shap_values=values,
        ranking=ranking,
        display_names=display_names,
        dependence_count=4,
        plot=PUBLICATION_PLOT,
        summary=PUBLICATION_SUMMARY,
        **kwargs,
    )
    plot_table_path = output_dir / f"shap_dependence_plot_files_{style['suffix']}.csv"
    plot_table.to_csv(plot_table_path, index=False)

    for path, expected in protected_hashes.items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"{task}: redraw unexpectedly changed {path.name}")
    return {
        "task": task,
        "summary": summary_path.name,
        "dependence_plots": plot_table["svg"].tolist(),
        "feature_sha256": protected_hashes[feature_path],
        "shap_sha256": protected_hashes[values_path],
        "ranking_sha256": protected_hashes[ranking_path],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["all", *(task.lower() for task in TASKS)])
    args = parser.parse_args()
    configure_matplotlib()
    tasks = TASKS if args.task == "all" else tuple(task for task in TASKS if task.lower() == args.task)
    for task in tasks:
        print(redraw_task(task), flush=True)


if __name__ == "__main__":
    main()
