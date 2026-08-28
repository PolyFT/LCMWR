#!/usr/bin/env python3
"""Generate Figure 4c-f from the actual post-retraining best models."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, mean_squared_error, r2_score, roc_curve


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from publication_fonts import register_publication_fonts

INPUT_DIR = ROOT / "results" / "model_compare" / "actual_best_oof"
OUTPUT_DIR = ROOT / "results" / "model_compare" / "figure4_oof"
PANEL = {"Tg": "figure4c_tg_oof.svg", "T5": "figure4d_t5_oof.svg", "LOI": "figure4e_loi_oof.svg"}
LABELS = {
    "LOI": ("Experimental LOI", "Predicted LOI"),
    "Tg": ("Experimental T$_g$ (°C)", "Predicted T$_g$ (°C)"),
    "T5": ("Experimental T$_{5\\%}$ (°C)", "Predicted T$_{5\\%}$ (°C)"),
}


def style(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(1.65)
        spine.set_color("#111111")
    ax.tick_params(direction="out", width=1.45, length=4.8, labelsize=10.5)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontweight("bold")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    register_publication_fonts()
    plt.rcParams.update({"font.family": "Arial", "font.weight": "bold", "axes.labelweight": "bold", "svg.fonttype": "path"})
    models = pd.read_csv(INPUT_DIR / "actual_best_models.csv").set_index("task")
    rows = []
    for task, filename in PANEL.items():
        data = pd.read_csv(INPUT_DIR / f"{task}_actual_best_oof.csv")
        y_true = pd.to_numeric(data["true_value"], errors="coerce").to_numpy()
        y_pred = pd.to_numeric(data["predicted_value"], errors="coerce").to_numpy()
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[valid], y_pred[valid]
        r2 = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        low, high = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        pad = max((high - low) * .04, 1e-8); low -= pad; high += pad
        fig, ax = plt.subplots(figsize=(3.35, 3.35))
        ax.scatter(y_true, y_pred, s=12, alpha=.60, color="#F28E16", edgecolors="none")
        ax.plot([low, high], [low, high], linestyle=(0, (6, 4)), linewidth=1.55, color="#9A9A9A")
        ax.set(xlim=(low, high), ylim=(low, high), xlabel=LABELS[task][0], ylabel=LABELS[task][1])
        ax.set_aspect("equal", adjustable="box")
        ax.text(.05, .95, f"R² = {r2:.3f}\nRMSE = {rmse:.3f}", transform=ax.transAxes, va="top", fontsize=10.5, fontweight="bold")
        style(ax); fig.tight_layout(pad=.35)
        fig.savefig(OUTPUT_DIR / filename, format="svg", bbox_inches="tight", transparent=True); plt.close(fig)
        rows.append({"task": task, "model": models.loc[task, "best_model"], "n_oof_samples": len(y_true), "OOF_R2": r2, "OOF_RMSE": rmse, "OOF_ROC_AUC": np.nan})

    data = pd.read_csv(INPUT_DIR / "UL94_actual_best_oof.csv")
    y_true = data["true_value"].astype(str).eq("V-0").astype(int)
    probability = pd.to_numeric(data["probability_V-0"], errors="raise")
    fpr, tpr, _ = roc_curve(y_true, probability); oof_auc = float(auc(fpr, tpr))
    fig, ax = plt.subplots(figsize=(3.35, 3.35))
    ax.plot([0, 1], [0, 1], linestyle=(0, (6, 4)), color="#9A9A9A", linewidth=1.55, label="Random baseline")
    ax.plot(fpr, tpr, color="#F28E16", linewidth=2.05, label=f"OOF (AUC = {oof_auc:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.set_aspect("equal", adjustable="box"); style(ax)
    legend = ax.legend(loc="lower right", frameon=False, fontsize=10)
    for text in legend.get_texts(): text.set_fontweight("bold")
    fig.tight_layout(pad=.35)
    fig.savefig(OUTPUT_DIR / "figure4f_ul94_oof_roc.svg", format="svg", bbox_inches="tight", transparent=True); plt.close(fig)
    rows.append({"task": "UL94", "model": models.loc["UL94", "best_model"], "n_oof_samples": len(data), "OOF_R2": np.nan, "OOF_RMSE": np.nan, "OOF_ROC_AUC": oof_auc})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "figure4_oof_metrics_summary.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
