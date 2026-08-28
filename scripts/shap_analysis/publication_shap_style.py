"""Shared publication styling for the Figure 5/6 SHAP SVG assets."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import shap


def _safe_stem(rank: int, feature: str, task: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", feature).strip("_")
    return f"shap_dependence_{rank:02d}_{token}_{task}"


def export_publication_shap_plots(
    *,
    task: str,
    output_dir: Path,
    features: pd.DataFrame,
    shap_values: np.ndarray,
    ranking: pd.DataFrame,
    display_names: list[str],
    summary_x_label: str,
    summary_tick_interval: float | None,
    summary_x_min: float | None = None,
    summary_x_max: float | None = None,
    summary_x_min_is_lower_bound: bool = True,
    summary_extra_x_ticks: tuple[float, ...] = (),
    dependence_count: int,
    plot: dict,
    summary: dict,
) -> tuple[Path, pd.DataFrame]:
    """Overwrite Figure 5 SVGs using the LOI/Tg publication style."""
    if shap_values.shape != features.shape:
        raise ValueError("SHAP values must have the same shape as the feature matrix.")
    if summary_tick_interval is not None and summary_tick_interval <= 0:
        raise ValueError("summary_tick_interval must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    n_show = min(summary["max_display"], features.shape[1])
    figure_height = max(
        summary["base_height_in"],
        summary["height_per_feature_in"] * n_show + summary["height_padding_in"],
    )
    fig = plt.figure(figsize=(summary["width_in"], figure_height))
    shap.summary_plot(
        shap_values,
        features,
        feature_names=display_names,
        max_display=n_show,
        plot_size=None,
        show=False,
        alpha=summary["point_alpha"],
        color_bar=summary["show_colorbar"],
        color_bar_label=summary["colorbar_label"],
    )
    ax = plt.gca()
    if summary["transparent_background"]:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
    for collection in ax.collections:
        collection.set_sizes([summary["point_size"]])
    ax.set_title("")
    ax.set_xlabel(summary_x_label, fontsize=summary["axis_label_size"], fontweight="bold")
    x_left, x_right = ax.get_xlim()
    if summary_x_min is None:
        new_left = x_left
    elif summary_x_min_is_lower_bound:
        new_left = min(summary_x_min, x_left)
    else:
        new_left = summary_x_min
    new_right = x_right if summary_x_max is None else summary_x_max
    ax.set_xlim(new_left, new_right)
    if summary_tick_interval is not None:
        ax.xaxis.set_major_locator(MultipleLocator(summary_tick_interval))
    elif summary_extra_x_ticks:
        ax.set_xticks(np.sort(np.unique(np.append(ax.get_xticks(), summary_extra_x_ticks))))
    ax.tick_params(
        axis="both",
        labelsize=summary["tick_label_size"],
        width=summary["tick_width"],
        length=summary["tick_length"],
    )
    ax.tick_params(axis="y", pad=0)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_horizontalalignment(summary["feature_label_alignment"])
    for spine in ax.spines.values():
        spine.set_linewidth(summary["spine_width"])
    ax.spines["bottom"].set_linewidth(summary["tick_width"])
    fig.tight_layout()
    fig.canvas.draw()

    feature_labels = ax.get_yticklabels()
    if feature_labels:
        renderer = fig.canvas.get_renderer()
        max_label_width_px = max(label.get_window_extent(renderer).width for label in feature_labels)
        axis_width_px = ax.get_window_extent(renderer).width
        gap_px = summary["feature_to_plot_gap_in"] * fig.dpi
        label_center_x = -(gap_px + max_label_width_px / 2) / axis_width_px
        for label in feature_labels:
            label.set_x(label_center_x)
            label.set_horizontalalignment(summary["feature_label_alignment"])

    colorbar_axes = [other_ax for other_ax in fig.axes if other_ax is not ax]
    if summary["show_colorbar"] and colorbar_axes:
        colorbar_ax = colorbar_axes[0]
        colorbar_ax.set_position(summary["colorbar_position"])
        colorbar_ax.set_ylabel("")
        colorbar_ax.set_yticks([])
        fig.text(*summary["feature_value_position"], summary["colorbar_label"], rotation=90,
                 va="center", ha="center", fontsize=summary["colorbar_label_size"], fontweight="bold")
        fig.text(*summary["high_position"], "High", va="center", ha="left",
                 fontsize=summary["high_low_size"], fontweight="bold")
        fig.text(*summary["low_position"], "Low", va="center", ha="left",
                 fontsize=summary["high_low_size"], fontweight="bold")

    summary_path = output_dir / f"shap_summary_plot_{task}.svg"
    fig.savefig(summary_path, bbox_inches="tight", transparent=summary["transparent_background"])
    plt.close(fig)

    figure_width = plot["left_margin_in"] + plot["axis_width_in"] + plot["right_margin_in"]
    figure_height = plot["bottom_margin_in"] + plot["axis_height_in"] + plot["top_margin_in"]
    axis_position = [
        plot["left_margin_in"] / figure_width,
        plot["bottom_margin_in"] / figure_height,
        plot["axis_width_in"] / figure_width,
        plot["axis_height_in"] / figure_height,
    ]
    rows = []
    for rank, feature in enumerate(ranking.head(dependence_count)["feature"], start=1):
        fig = plt.figure(figsize=(figure_width, figure_height))
        ax = fig.add_axes(axis_position)
        if plot["transparent_background"]:
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
        shap.dependence_plot(
            features.columns.get_loc(feature),
            shap_values,
            features,
            feature_names=display_names,
            interaction_index="auto",
            ax=ax,
            show=False,
            dot_size=plot["dot_size"],
            alpha=plot["alpha"],
        )
        for extra_axis in list(fig.axes):
            if extra_axis is not ax:
                extra_axis.remove()
        # SHAP 会重设传入 Axes 的位置；恢复预设位置以锁定内框为 6.5 × 5.2 in。
        ax.set_position(axis_position)
        ax.set_title("")
        ax.set_ylabel(
            "SHAP Value",
            fontsize=plot.get("y_axis_label_size", plot["axis_label_size"]),
            fontweight="bold",
        )
        ax.xaxis.label.set_fontsize(plot["axis_label_size"])
        ax.xaxis.label.set_fontweight("bold")
        ax.tick_params(
            axis="both", which="major", labelsize=plot["tick_label_size"],
            width=plot["tick_width"], length=plot["tick_length"],
        )
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(plot["spine_width"])
        output_path = output_dir / f"{_safe_stem(rank, feature, task)}.svg"
        fig.savefig(output_path, transparent=plot["transparent_background"])
        plt.close(fig)
        rows.append(
            {
                "rank": rank,
                "feature": feature,
                "display_feature": display_names[features.columns.get_loc(feature)],
                "svg": output_path.name,
            }
        )
    return summary_path, pd.DataFrame(rows)
