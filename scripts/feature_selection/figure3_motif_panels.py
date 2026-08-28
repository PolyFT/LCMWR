"""Shared plotting helpers for the four Figure 3b-h motif notebooks."""

from __future__ import annotations

import os
import re
from pathlib import Path

CACHE_ROOT = Path("/tmp/lcmwr_figure3b_h_cache")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_ROOT / "numba"))
(CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "numba").mkdir(parents=True, exist_ok=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd
from IPython.display import SVG, display
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler

try:
    import umap.umap_ as umap
except ImportError as exc:
    raise ImportError("Install umap-learn to generate Figure 3f.") from exc


TASK_LABELS = {"loi": "LOI", "tg": "Tg", "t5": "T5", "ul94": "UL94"}
METRIC_LABELS = {"loi": "R²", "tg": "R²", "t5": "R²", "ul94": "ROC-AUC"}

# ----- Figure 3 style: adjust these values in the notebooks if desired -----
STYLE = {
    "font_family": "Arial",
    "font_size": 9,
    "axis_label_size": 10,
    "panel_label_size": 13,
    "line_width": 1.2,
    "tick_width": 1.2,
    "figure_dpi": 180,
    "transparent_background": True,
    "blue": "#73A9CC",
    "light_blue": "#B7D1E5",
    "dark_blue": "#4A7899",
    "orange": "#F29B6D",
    "yellow": "#F4D03F",
    "grey": "#B7B7B7",
    "black": "#222222",
    "histogram_bins": 20,
    # b/c：每一个 vocab 对应一根分子量柱；坐标框尺寸独立于外侧标签留白。
    "vocab_figure_width": 6.5,
    "vocab_figure_height": 2.7,
    "vocab_frame_width_inches": 7.7,
    "vocab_frame_height_inches": 2.842,
    "vocab_frame_width_to_height": 7.7 / 2.842,
    "vocab_bar_width": 0.96,
    # d/e：Atom Number 柱条渐变与柱顶数量标签。
    "atom_number_gradient_top": "#EAF3F8",
    "atom_number_count_label_size": 8,
    "atom_number_count_label_offset": 3,
    "umap_random_state": 48,
    "umap_n_neighbors": 5,
    "umap_min_dist": 0.8,
    "umap_metric": "euclidean",
    "umap_init": "random",
    "recompute_umap": False,
    # f：参考论文点云样式；坐标轴隐藏，仅保留透明背景、图例和 motif 分布。
    "umap_figure_width": 4.2,
    "umap_figure_height": 3.4,
    # 当前 UMAP 经 tight_layout 后的实际坐标框大小（in）；d–h 与其严格对齐。
    "umap_frame_width_inches": 3.4909722222,
    "umap_frame_height_inches": 2.8416666667,
    "umap_other_size": 12,
    "umap_selected_size": 16,
    "umap_other_face": "#D9EEF9",
    "umap_other_edge": "#73A9CC",
    "umap_selected_face": "#FFD09A",
    "umap_selected_edge": "#F29B6D",
    "umap_other_alpha": 0.92,
    "umap_selected_alpha": 1.0,
    "umap_marker_edge_width": 0.45,
    "umap_axis_padding_fraction": 0.05,
    "umap_show_title": True,
    "stepwise_metric_low_fraction": 0.20,
    "stepwise_metric_high_fraction": 0.80,
    # Additional text offsets (in points) for individual feature-count labels.
    # Keys are zero-based screening-stage positions.
    "stepwise_bar_label_extra_offsets": {},
    "stepwise_label_rotation": 30,
    "stepwise_label_right_shift": 0.10,
    "stepwise_label_y_offset": -0.045,
    "stepwise_xlabel_labelpad": 30,
    "stepwise_figure_height": 3.55,
    "search_point_size": 18,
    "search_point_edge_width": 0.35,
    "search_colorbar_fraction": 0.050,
    "search_y_tick_decimals": 2,
    "atom_number_width_per_bar": 0.22,
    "atom_number_min_width": 5.4,
    "atom_number_max_width": 9.0,
    "atom_number_tick_size": 8,
}


def find_project_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    required = (root / "results", root / "scripts", root / "dataset")
    if not all(path.is_dir() for path in required):
        raise FileNotFoundError(f"Could not locate the repository from {__file__}")
    return root


def apply_style(style: dict = STYLE) -> None:
    font_paths = (
        Path(os.environ.get("LCMWR_ARIAL_REGULAR", "/mnt/c/Windows/Fonts/arial.ttf")),
        Path(os.environ.get("LCMWR_ARIAL_BOLD", "/mnt/c/Windows/Fonts/arialbd.ttf")),
    )
    missing = [path for path in font_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Arial publication font file(s) not found: "
            + ", ".join(str(path) for path in missing)
            + ". Set LCMWR_ARIAL_REGULAR and LCMWR_ARIAL_BOLD."
        )
    for path in font_paths:
        font_manager.fontManager.addfont(str(path))
    mpl.rcParams.update(
        {
            "font.family": style["font_family"],
            "font.size": style["font_size"],
            "axes.linewidth": style["line_width"],
            "axes.labelweight": "bold",
            # PPT 对 SVG 的文字重排会改变字符位置；转为路径以保持版式稳定。
            "svg.fonttype": "path",
            "figure.facecolor": "none",
            "axes.facecolor": "none",
        }
    )


def task_paths(task: str) -> dict[str, Path]:
    root = find_project_root()
    result_dir = root / "results" / f"{task}_motif_select"
    output_dir = result_dir / "figure3_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "result_dir": result_dir,
        "output_dir": output_dir,
        "vocab": root / "results" / "local_vocab_parallel_threshold.csv",
        "selected": result_dir / "best_improved_final_features_info.csv",
        "search": result_dir / "plots_data" / "improved_750_complete_results.csv",
        "stepwise": result_dir / "plots_data" / "best_improved_params_stepwise_results.csv",
        "matrix": result_dir / f"{task}_vocab_weighted_feature_matrix.csv",
    }


def motif_properties(smiles: str) -> pd.Series:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid motif SMILES: {smiles!r}")
    return pd.Series({"molecular_weight": Descriptors.MolWt(mol), "atom_count": mol.GetNumAtoms()})


def load_task_data(task: str, style: dict = STYLE) -> dict:
    if task not in TASK_LABELS:
        raise ValueError(f"Unknown task: {task}")
    paths = task_paths(task)
    vocabulary = pd.read_csv(paths["vocab"])
    selected = pd.read_csv(paths["selected"])
    search = pd.read_csv(paths["search"])
    stepwise = pd.read_csv(paths["stepwise"])
    motifs = pd.concat([vocabulary[["smiles"]], vocabulary["smiles"].apply(motif_properties)], axis=1)
    # Original 特征集定义为特征矩阵中已去除全 0 motif 后保留下来的列，且维持全局词表顺序。
    original_feature_names = pd.read_csv(paths["matrix"], nrows=1).columns
    original_motifs = motifs.loc[motifs["smiles"].isin(original_feature_names)].copy()
    if len(original_motifs) != len(original_feature_names):
        raise ValueError("The Original feature matrix contains motif names absent from the global vocabulary.")
    selected_smiles = set(selected["feature_name"])
    motifs["selected"] = motifs["smiles"].isin(selected_smiles)
    selected_motifs = motifs.loc[motifs["selected"]].copy()
    if len(selected_motifs) != len(selected_smiles):
        raise ValueError("The selected motif list is not a subset of the global vocabulary.")
    step_names = stepwise["step"].astype(str).str.strip().str.casefold()
    original_rows = stepwise.loc[step_names.eq("original"), "feature_count"]
    correlation_rows = stepwise.loc[step_names.eq("correlation"), "feature_count"]
    if original_rows.empty or correlation_rows.empty:
        raise ValueError("Stepwise results must contain both Original and Correlation feature counts.")
    original_feature_count = int(original_rows.iloc[0])
    correlation_feature_count = int(correlation_rows.iloc[-1])
    best = search.loc[search["performance"].idxmax()].copy()
    active_count = len(original_feature_names)
    if active_count != original_feature_count:
        raise ValueError("Original feature count in stepwise results does not match the non-zero feature matrix.")
    summary = pd.DataFrame(
        [
            {
                "task": TASK_LABELS[task],
                "metric": METRIC_LABELS[task],
                "global_vocabulary_size": len(motifs),
                "active_feature_count": active_count,
                "selected_motif_count": len(selected_motifs),
                "original_feature_count": original_feature_count,
                "correlation_feature_count": correlation_feature_count,
                "best_750_performance": best["performance"],
                "best_750_feature_count": best["feature_count"],
                "umap_random_state": style["umap_random_state"],
                "umap_n_neighbors": style["umap_n_neighbors"],
                "umap_min_dist": style["umap_min_dist"],
            }
        ]
    )
    summary.to_csv(paths["output_dir"] / f"figure3_{task}_analysis_summary.csv", index=False)
    return {
        "task": task, "label": TASK_LABELS[task], "metric": METRIC_LABELS[task],
        "paths": paths, "motifs": motifs, "original_motifs": original_motifs, "selected_motifs": selected_motifs,
        "selected_smiles": selected_smiles, "search": search, "stepwise": stepwise,
        "best": best, "summary": summary,
        "original_feature_count": original_feature_count,
        "correlation_feature_count": correlation_feature_count,
    }


def style_axes(ax, style: dict = STYLE, top_right: bool = True) -> None:
    for side in ("bottom", "left", "top", "right"):
        ax.spines[side].set_linewidth(style["line_width"])
        ax.spines[side].set_color(style["black"])
    if not top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=style["tick_width"], length=4, labelsize=style["font_size"], color=style["black"])
    for tick_label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick_label.set_fontweight("bold")
    ax.grid(False)


def panel_label(ax, label: str, style: dict = STYLE) -> None:
    ax.text(-0.16, 1.08, f"({label})", transform=ax.transAxes, fontsize=style["panel_label_size"],
            fontweight="bold", va="top", ha="left", color=style["black"])


def save_and_display(fig, path: Path, style: dict = STYLE) -> None:
    fig.savefig(path, format="svg", transparent=style["transparent_background"], bbox_inches="tight", pad_inches=0.03)
    # SVG 文字已按真实 Arial 轮廓写入路径，避免 PowerPoint 字体替换造成重叠。
    display(SVG(filename=str(path)))
    plt.close(fig)
    print(f"Saved: {path}")


def fit_frame_to_umap(fig, ax, style: dict = STYLE) -> None:
    """Resize the outer canvas so the plotted coordinate frame matches Figure 3f's UMAP frame."""
    fit_frame_to_dimensions(
        fig, ax,
        width_inches=style["umap_frame_width_inches"],
        height_inches=style["umap_frame_height_inches"],
    )


def fit_frame_to_dimensions(fig, ax, *, width_inches: float, height_inches: float) -> None:
    """Resize the outer canvas so an axes frame has the requested physical size."""
    position = ax.get_position()
    if position.width <= 0 or position.height <= 0:
        raise ValueError("Cannot align a zero-size axes frame to the UMAP frame.")
    fig.set_size_inches(
        width_inches / position.width,
        height_inches / position.height,
        forward=True,
    )


def plot_distribution(data: dict, property_name: str, selected: bool, panel: str, style: dict = STYLE) -> None:
    """Plot every motif vocabulary entry as one molecular-weight bar, in source-file order."""
    if property_name != "molecular_weight":
        raise ValueError("Figure 3b/c are defined as vocabulary-by-molecular-weight bar plots.")
    # Do not sort: the x position is the existing vocabulary/selected-feature order.
    values = (data["selected_motifs"] if selected else data["original_motifs"]).reset_index(drop=True)
    label = "selected motifs" if selected else "Original non-zero motif features"
    suffix = f"{property_name}_{'selected' if selected else 'all'}"
    positions = np.arange(len(values))
    # b：原始 vocab 的每根细柱黑—蓝交替，无描边，便于在高密度区域辨认单个 vocab；
    # c：筛选后 vocab 保持单一蓝色。
    if selected:
        colours = style["blue"]
    else:
        colours = [style["black"] if index % 2 == 0 else style["blue"] for index in positions]
    fig, ax = plt.subplots(
        figsize=(style["vocab_figure_width"], style["vocab_figure_height"]), dpi=style["figure_dpi"]
    )
    ax.bar(positions, values["molecular_weight"].to_numpy(), width=style["vocab_bar_width"],
           color=colours, edgecolor="none", linewidth=0)
    ax.set_xlim(-0.5, len(values) - 0.5)
    ax.set_xticks([])  # 每根柱即一个 vocab；不为数千个 vocab 逐一显示不可读的文字。
    ax.set_xlabel("Vocab", fontsize=style["axis_label_size"], fontweight="bold")
    ax.set_ylabel("Molecular weight (Da)", fontsize=style["axis_label_size"], fontweight="bold")
    # ax.set_title(f"{data['label']}: {label}", fontsize=style["axis_label_size"], fontweight="bold", pad=4)
    style_axes(ax, style)
    # 将真实坐标框锁定为用户指定的 7.7 × 2.842 in，不受外侧标签边距影响。
    ax.set_box_aspect(1 / style["vocab_frame_width_to_height"])
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.32, top=0.84)
    fit_frame_to_dimensions(
        fig, ax,
        width_inches=style["vocab_frame_width_inches"],
        height_inches=style["vocab_frame_height_inches"],
    )
    save_and_display(fig, data["paths"]["output_dir"] / f"Figure3{panel}_{data['task']}_{suffix}.svg", style)


def plot_atom_number_frequency(data: dict, selected: bool, panel: str, style: dict = STYLE) -> None:
    """Draw one bar and one x-axis tick for every observed atom number."""
    values = data["selected_motifs"] if selected else data["original_motifs"]
    # RDKit 原子数本身为整数；显式转换避免 pandas 以 1.0、2.0 显示横坐标。
    counts = values["atom_count"].round().astype(int).value_counts().sort_index()
    width = min(
        style["atom_number_max_width"],
        max(style["atom_number_min_width"], len(counts) * style["atom_number_width_per_bar"]),
    )
    fig, ax = plt.subplots(figsize=(width, 2.8), dpi=style["figure_dpi"])
    positions = np.arange(len(counts))
    # d/e 使用同一套浅底—深顶蓝色渐变，仅数据范围不同。
    colour = style["blue"]
    label = "selected motifs" if selected else "Original non-zero motif features"
    bars = ax.bar(positions, counts.to_numpy(), color="none", edgecolor=style["black"],
                  linewidth=0.45, width=0.88, zorder=3)
    # 由浅蓝（柱底）向深蓝（柱顶）的渐变，裁切到每根独立柱条中，SVG 仍为矢量边框。
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    cmap = LinearSegmentedColormap.from_list("atom_number_gradient", [style["atom_number_gradient_top"], colour])
    for bar, value in zip(bars, counts.to_numpy()):
        image = ax.imshow(
            gradient, extent=(bar.get_x(), bar.get_x() + bar.get_width(), 0, value),
            origin="lower", aspect="auto", cmap=cmap, interpolation="bicubic", zorder=2,
        )
        image.set_clip_path(bar)
    # imshow 会重新计算数据范围；重设范围避免第一个 Atom Number=1 的柱被左边框裁掉一半。
    ax.set_xlim(-0.5, len(counts) - 0.5)
    y_limit = float(counts.max()) * 1.13
    ax.set_ylim(0, y_limit)
    for bar, value in zip(bars, counts.to_numpy()):
        ax.annotate(
            f"{int(value)}", (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, style["atom_number_count_label_offset"]), textcoords="offset points",
            ha="center", va="bottom", fontsize=style["atom_number_count_label_size"],
            fontweight="bold", color=style["black"], clip_on=False, zorder=4,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(counts.index.astype(str), rotation=0, ha="center", fontsize=style["atom_number_tick_size"], fontweight="bold")
    ax.set_xlabel("Atom Number", fontsize=style["axis_label_size"], fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=style["axis_label_size"], fontweight="bold")
    # ax.set_title(f"{data['label']}: {label}", fontsize=style["axis_label_size"], fontweight="bold", pad=7)
    style_axes(ax, style)
    for tick_label in ax.get_xticklabels():
        tick_label.set_fontsize(style["atom_number_tick_size"])
        tick_label.set_fontweight("bold")
    ax.tick_params(axis="x", length=2)
    fig.tight_layout(pad=0.55)
    fit_frame_to_umap(fig, ax, style)
    suffix = f"atom_number_frequency_{'selected' if selected else 'all'}"
    save_and_display(fig, data["paths"]["output_dir"] / f"Figure3{panel}_{data['task']}_{suffix}.svg", style)


def get_umap_coordinates(data: dict, style: dict = STYLE) -> pd.DataFrame:
    path = data["paths"]["output_dir"] / f"figure3f_{data['task']}_motif_umap_coordinates.csv"
    required = {"smiles", "umap_1", "umap_2", "selected"}
    if path.exists() and not style["recompute_umap"]:
        coordinates = pd.read_csv(path)
        if required.issubset(coordinates.columns):
            return coordinates
    matrix = pd.read_csv(data["paths"]["matrix"]).round(5)
    motif_by_sample = matrix.T
    motif_by_sample = motif_by_sample.loc[motif_by_sample.var(axis=1) > 1e-12].copy()
    scaled = StandardScaler().fit_transform(motif_by_sample.to_numpy(dtype=np.float32))
    reducer = umap.UMAP(
        n_components=2, n_neighbors=style["umap_n_neighbors"], min_dist=style["umap_min_dist"],
        metric=style["umap_metric"], init=style["umap_init"], random_state=style["umap_random_state"],
    )
    embedding = reducer.fit_transform(scaled)
    coordinates = pd.DataFrame(
        {
            "smiles": motif_by_sample.index,
            "umap_1": embedding[:, 0],
            "umap_2": embedding[:, 1],
            "selected": motif_by_sample.index.isin(data["selected_smiles"]),
            "random_state": style["umap_random_state"],
            "n_neighbors": style["umap_n_neighbors"],
            "min_dist": style["umap_min_dist"],
            "metric": style["umap_metric"],
            "initialization": style["umap_init"],
            "scaling": "StandardScaler_per_sample_dimension",
        }
    )
    coordinates.to_csv(path, index=False)
    return coordinates


def plot_umap(data: dict, style: dict = STYLE) -> None:
    coordinates = get_umap_coordinates(data, style)
    fig, ax = plt.subplots(
        figsize=(style["umap_figure_width"], style["umap_figure_height"]), dpi=style["figure_dpi"]
    )
    other = coordinates.loc[~coordinates["selected"]]
    kept = coordinates.loc[coordinates["selected"]]
    # 参考图：全体 motif 为浅蓝色空心点，最终筛选 motif 以浅橙色叠加突出。
    ax.scatter(
        other["umap_1"], other["umap_2"], s=style["umap_other_size"],
        facecolors=style["umap_other_face"], edgecolors=style["umap_other_edge"],
        alpha=style["umap_other_alpha"], linewidths=style["umap_marker_edge_width"],
        label="Other motifs", zorder=2,
    )
    ax.scatter(
        kept["umap_1"], kept["umap_2"], s=style["umap_selected_size"],
        facecolors=style["umap_selected_face"], edgecolors=style["umap_selected_edge"],
        alpha=style["umap_selected_alpha"], linewidths=style["umap_marker_edge_width"],
        label="Selected motifs", zorder=3,
    )
    # 在 UMAP 点云四周预留可调比例的空间，避免边缘点贴近坐标框。
    padding = style["umap_axis_padding_fraction"]
    x_min, x_max = coordinates["umap_1"].min(), coordinates["umap_1"].max()
    y_min, y_max = coordinates["umap_2"].min(), coordinates["umap_2"].max()
    x_span = max(float(x_max - x_min), np.finfo(float).eps)
    y_span = max(float(y_max - y_min), np.finfo(float).eps)
    ax.set_xlim(x_min - x_span * padding, x_max + x_span * padding)
    ax.set_ylim(y_min - y_span * padding, y_max + y_span * padding)
    # 保留 Figure 3 原有的坐标轴、框线与标题；仅调整散点和图例位置。
    ax.set_xlabel("UMAP 1", fontsize=style["axis_label_size"], fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=style["axis_label_size"], fontweight="bold")
    if style["umap_show_title"]:
        ax.set_title(f"{data['label']}: motif feature space", fontsize=style["axis_label_size"], fontweight="bold", pad=7)
    # 图例首行记录相关性过滤后的特征数/筛选流程起始特征数。
    handles, _ = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles, ["Other motifs", "Selected motifs"], frameon=False,
        prop={"size": style["font_size"] - 1, "weight": "bold"},
        title=f"Selected: {data['correlation_feature_count']}/{data['original_feature_count']}",
        title_fontproperties={"size": style["font_size"] - 1, "weight": "bold"},
        loc="upper left", handletextpad=0.3,
    )
    # Legend title is the first row; align it with the scatter-symbol rows.
    legend._legend_box.align = "left"
    legend.get_title().set_ha("left")
    style_axes(ax, style)
    fig.tight_layout(pad=0.55)
    # 将 Figure 3f 的实际坐标框锁定为 3.491 × 2.842 in，与 d/e/g/h 一致。
    fit_frame_to_umap(fig, ax, style)
    save_and_display(fig, data["paths"]["output_dir"] / f"Figure3f_{data['task']}_motif_umap.svg", style)


def plot_search(data: dict, style: dict = STYLE) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.2), dpi=style["figure_dpi"])
    performance = data["search"]["performance"].to_numpy(dtype=float)
    # 论文图3g：性能低处蓝色、性能高处橙色，颜色沿纵向自然渐变。
    cmap = LinearSegmentedColormap.from_list("search_performance_gradient", [style["blue"], style["orange"]])
    norm = mpl.colors.Normalize(vmin=float(np.nanmin(performance)), vmax=float(np.nanmax(performance)))
    face_colours = cmap(norm(performance))
    # 每个点的描边与填充色保持同色系，但降低亮度以形成清晰层次。
    edge_colours = face_colours.copy()
    edge_colours[:, :3] = np.clip(edge_colours[:, :3] * 0.55, 0, 1)
    points = ax.scatter(
        data["search"]["feature_count"], performance, c=performance, cmap=cmap, norm=norm,
        s=style["search_point_size"], alpha=0.88, edgecolors=edge_colours,
        linewidths=style["search_point_edge_width"], zorder=3,
    )
    colorbar = fig.colorbar(points, ax=ax, pad=0.025, fraction=style["search_colorbar_fraction"])
    colorbar.set_ticks([])
    colorbar.outline.set_visible(False)
    ax.scatter([data["best"]["feature_count"]], [data["best"]["performance"]], s=90, marker="*",
               color=style["yellow"], edgecolor=style["black"], linewidth=0.55, zorder=4, label="Best combination")
    ax.set_xlabel("Selected feature count", fontsize=style["axis_label_size"], fontweight="bold")
    ax.set_ylabel(data["metric"], fontsize=style["axis_label_size"], fontweight="bold")
    ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{int(style['search_y_tick_decimals'])}f"))
    # ax.set_title(f"{data['label']}: 750 threshold combinations", fontsize=style["axis_label_size"], fontweight="bold", pad=7)
    ax.legend(frameon=False, fontsize=style["font_size"] - 1, loc="best")
    style_axes(ax, style)
    fig.tight_layout(pad=0.55)
    fit_frame_to_umap(fig, ax, style)
    save_and_display(fig, data["paths"]["output_dir"] / f"Figure3g_{data['task']}_750_search.svg", style)


def plot_stepwise(data: dict, style: dict = STYLE) -> None:
    # The final row repeats the correlation result; Figure 3 shows the five
    # distinct screening stages only.
    stepwise = data["stepwise"].loc[data["stepwise"]["step"].ne("Final")].head(5).copy()
    fig, left = plt.subplots(figsize=(4.3, style["stepwise_figure_height"]), dpi=style["figure_dpi"])
    x = np.arange(len(stepwise))
    bars = left.bar(x, stepwise["feature_count"], color=style["light_blue"], edgecolor=style["black"], linewidth=0.55, label="Feature count")
    bar_labels = left.bar_label(
        bars,
        labels=[f"{int(value)}" for value in stepwise["feature_count"]],
        padding=2,
        fontsize=style["font_size"],
        fontweight="bold",
    )
    for index, extra_offset in style["stepwise_bar_label_extra_offsets"].items():
        if 0 <= index < len(bar_labels):
            x_offset, y_offset = bar_labels[index].get_position()
            bar_labels[index].set_position((x_offset, y_offset + extra_offset))
    left.set_ylim(0, stepwise["feature_count"].max() * 1.13)
    left.set_ylabel("Feature count", fontsize=style["axis_label_size"], fontweight="bold")
    left.set_xticks(x)
    # Matplotlib 会在绘制时重置自动 tick-label 的 x 位置，因而不能用 set_x() 微调。
    # 改为独立文字对象：STEPWISE_LABEL_RIGHT_SHIFT 直接参与 data-x 坐标，数值变化会真实移动标签。
    left.set_xticklabels([])
    for xpos, step_name in zip(x, stepwise["step"].astype(str)):
        left.text(
            xpos + style["stepwise_label_right_shift"], style["stepwise_label_y_offset"], step_name,
            transform=left.get_xaxis_transform(), ha="right", va="top",
            rotation=style["stepwise_label_rotation"], rotation_mode="anchor",
            fontsize=style["font_size"], fontweight="bold", color=style["black"],
            clip_on=False,
        )
    left.set_xlabel("Selection Step", fontsize=style["axis_label_size"], fontweight="bold",
                    labelpad=style["stepwise_xlabel_labelpad"])
    style_axes(left, style)
    right = left.twinx()
    line, = right.plot(x, stepwise["performance"], color=style["black"], marker="o", markersize=4, linewidth=1.8, label=data["metric"])
    right.set_ylabel(data["metric"], fontsize=style["axis_label_size"], fontweight="bold")
    right.tick_params(axis="y", width=style["tick_width"], length=4, labelsize=style["font_size"], color=style["black"])
    for tick_label in right.get_yticklabels():
        tick_label.set_fontweight("bold")
    right.spines["right"].set_linewidth(style["line_width"])
    right.spines["top"].set_visible(False)
    metric_min, metric_max = stepwise["performance"].min(), stepwise["performance"].max()
    metric_span = max(metric_max - metric_min, abs(metric_max) * 0.01, 0.01)
    line_fraction_span = style["stepwise_metric_high_fraction"] - style["stepwise_metric_low_fraction"]
    if not 0 < line_fraction_span < 1:
        raise ValueError("Stepwise metric fractions must satisfy 0 <= low < high <= 1.")
    axis_span = metric_span / line_fraction_span
    axis_min = metric_min - style["stepwise_metric_low_fraction"] * axis_span
    right.set_ylim(axis_min, axis_min + axis_span)
    right.yaxis.set_major_locator(MaxNLocator(nbins=5))
    # left.set_title(f"{data['label']}: stepwise feature screening", fontsize=style["axis_label_size"], fontweight="bold", pad=7)
    # Matplotlib 根据柱条与折线位置自动选择遮挡最少的角落。
    left.legend([bars, line], ["Feature count", data["metric"]], frameon=False,
                fontsize=style["font_size"] - 1, loc="best")
    fig.tight_layout(pad=0.55)
    fit_frame_to_umap(fig, left, style)
    save_and_display(fig, data["paths"]["output_dir"] / f"Figure3h_{data['task']}_stepwise_screening.svg", style)
