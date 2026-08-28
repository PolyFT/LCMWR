#!/usr/bin/env python3
"""Create the Figure 3b-h and Figure 4 OOF assets for the Tg_GREA experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
RESULTS_DIR = EXPERIMENT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SCRIPTS_DIR = REPOSITORY_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from publication_fonts import register_publication_fonts


def figure3_paths(_: str) -> dict[str, Path]:
    """Map the shared Figure 3 plotting helpers onto this isolated experiment."""
    output_dir = FIGURES_DIR / "figure3_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_dir = RESULTS_DIR / "feature_selection"
    return {
        "root": REPOSITORY_ROOT,
        "result_dir": selection_dir,
        "output_dir": output_dir,
        "vocab": RESULTS_DIR / "local_vocab_parallel_threshold.csv",
        "selected": selection_dir / "best_improved_final_features_info.csv",
        "search": selection_dir / "plots_data" / "improved_750_complete_results.csv",
        "stepwise": selection_dir / "plots_data" / "best_improved_params_stepwise_results.csv",
        "matrix": RESULTS_DIR / "features" / "tg_grea_vocab_weighted_feature_matrix.csv",
    }


def make_figure3() -> list[Path]:
    shared_dir = REPOSITORY_ROOT / "scripts" / "feature_selection"
    if str(shared_dir) not in sys.path:
        sys.path.insert(0, str(shared_dir))
    import figure3_motif_panels as panels

    panels.TASK_LABELS["tg"] = "Tg_GREA"
    panels.task_paths = figure3_paths
    panels.STYLE.update(
        {
            "umap_axis_padding_fraction": 0.10,
            "umap_show_title": False,
            "stepwise_label_right_shift": 0.0,
            # Correlation (the fifth bar) is close to its R² marker; lift only
            # its feature-count label to keep the two annotations separate.
            "stepwise_bar_label_extra_offsets": {4: 14},
            "atom_number_tick_size": 9,
        }
    )
    panels.apply_style(panels.STYLE)
    data = panels.load_task_data("tg", panels.STYLE)
    panels.plot_distribution(data, "molecular_weight", selected=False, panel="b", style=panels.STYLE)
    panels.plot_distribution(data, "molecular_weight", selected=True, panel="c", style=panels.STYLE)
    panels.plot_atom_number_frequency(data, selected=False, panel="d", style=panels.STYLE)
    panels.plot_atom_number_frequency(data, selected=True, panel="e", style=panels.STYLE)
    panels.plot_umap(data, panels.STYLE)
    panels.plot_search(data, panels.STYLE)
    panels.plot_stepwise(data, panels.STYLE)

    generated = sorted(figure3_paths("tg")["output_dir"].glob("Figure3*.svg"))
    if len(generated) != 7:
        raise RuntimeError(f"Expected 7 Figure 3 assets, found {len(generated)}.")
    return generated


def make_figure4() -> Path:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_squared_error, r2_score

    source = RESULTS_DIR / "best_model" / "Tg_GREA_actual_best_oof.csv"
    metadata = RESULTS_DIR / "best_model" / "actual_best_model.json"
    if not source.is_file() or not metadata.is_file():
        raise FileNotFoundError("The validated best-model OOF outputs are required before plotting Figure 4.")
    data = pd.read_csv(source)
    observed = pd.to_numeric(data["Tg"], errors="coerce").to_numpy(dtype=float)
    predicted = pd.to_numeric(data["predicted_value"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[valid], predicted[valid]
    if len(observed) != 7174:
        raise RuntimeError(f"Expected 7,174 valid OOF pairs, found {len(observed)}.")

    r2 = float(r2_score(observed, predicted))
    rmse = float(np.sqrt(mean_squared_error(observed, predicted)))
    low, high = min(observed.min(), predicted.min()), max(observed.max(), predicted.max())
    pad = max((high - low) * 0.04, 1e-8)
    low, high = low - pad, high + pad

    register_publication_fonts()
    plt.rcParams.update(
        {"font.family": "Arial", "font.weight": "bold", "axes.labelweight": "bold", "svg.fonttype": "path"}
    )
    fig, ax = plt.subplots(figsize=(3.35, 3.35))
    fig.patch.set_alpha(0)
    ax.scatter(observed, predicted, s=12, c="#F28E16", alpha=0.60, linewidths=0, rasterized=False, zorder=3)
    ax.plot([low, high], [low, high], color="#9A9A9A", linewidth=1.55, linestyle=(0, (4, 3)), zorder=2)
    ax.set(xlim=(low, high), ylim=(low, high), xlabel="Experimental T$_g$ (°C)", ylabel="Predicted T$_g$ (°C)")
    ax.set_aspect("equal", adjustable="box")
    ax.text(0.05, 0.95, f"R² = {r2:.3f}\nRMSE = {rmse:.3f}", transform=ax.transAxes, va="top", fontsize=10.5, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.65)
        spine.set_color("#111111")
    ax.tick_params(direction="out", width=1.45, length=4.8, labelsize=10.5)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontweight("bold")
    fig.tight_layout(pad=0.35)

    output_dir = FIGURES_DIR / "figure4_oof"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "Figure4_tg_grea_oof.svg"
    fig.savefig(output, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    pd.DataFrame(
        [{"task": "Tg_GREA", "n_oof_samples": len(observed), "OOF_R2": r2, "OOF_RMSE": rmse}]
    ).to_csv(output_dir / "figure4_tg_grea_oof_metrics.csv", index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("figure3", "figure4", "all"), nargs="?", default="all")
    args = parser.parse_args()
    generated: list[Path] = []
    if args.stage in {"figure3", "all"}:
        generated.extend(make_figure3())
    if args.stage in {"figure4", "all"}:
        generated.append(make_figure4())
    print("Generated:")
    for path in generated:
        print(path.relative_to(EXPERIMENT_ROOT))


if __name__ == "__main__":
    main()
