#!/usr/bin/env python3
"""Execute a MOTIF selection notebook as a reproducible batch job.

The notebook is deliberately kept as the source of truth.  This runner only
loads its code cells in order, which makes a long 750-combination search easy
to launch from a terminal or tmux while preserving the same Cell 0/Cell 1
workflow used interactively.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = {
    "motif_vocab": ROOT / "scripts" / "motif_generating.ipynb",
    "loi": ROOT / "scripts" / "feature_selection" / "LOI_motif_select.ipynb",
    "tg": ROOT / "scripts" / "feature_selection" / "Tg_motif_select.ipynb",
    "t5": ROOT / "scripts" / "feature_selection" / "T5_motif_select.ipynb",
    "ul94": ROOT / "scripts" / "feature_selection" / "UL94_motif_select.ipynb",
    "loi_model_compare": ROOT / "scripts" / "model_comparison" / "loi_model_compare.ipynb",
    "tg_model_compare": ROOT / "scripts" / "model_comparison" / "tg_model_compare.ipynb",
    "t5_model_compare": ROOT / "scripts" / "model_comparison" / "t5_model_compare.ipynb",
    "ul94_model_compare": ROOT / "scripts" / "model_comparison" / "ul94_model_compare.ipynb",
    "figure3_loi": ROOT / "scripts" / "feature_selection" / "figure3_loi_motif_analysis.ipynb",
    "figure3_tg": ROOT / "scripts" / "feature_selection" / "figure3_tg_motif_analysis.ipynb",
    "figure3_t5": ROOT / "scripts" / "feature_selection" / "figure3_t5_motif_analysis.ipynb",
    "figure3_ul94": ROOT / "scripts" / "feature_selection" / "figure3_ul94_motif_analysis.ipynb",
    "figure4": ROOT / "scripts" / "model_comparison" / "figure4_oof_analysis.ipynb",
    "shap_loi": ROOT / "scripts" / "shap_analysis" / "loi_extratrees_shap_analysis.ipynb",
    "shap_tg": ROOT / "scripts" / "shap_analysis" / "tg_lightgbm_shap_analysis.ipynb",
    "shap_t5": ROOT / "scripts" / "shap_analysis" / "t5_xgboost_shap_analysis.ipynb",
    "shap_ul94": ROOT / "scripts" / "shap_analysis" / "ul94_xgboost_shap_analysis.ipynb",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=NOTEBOOKS, type=str.lower)
    parser.add_argument("--start-cell", type=int, default=0)
    parser.add_argument("--stop-after-cell", type=int, default=None)
    args = parser.parse_args()

    notebook = NOTEBOOKS[args.task]
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    try:
        from IPython.display import display
    except Exception:
        display = print
    namespace = {"__name__": "__main__", "__file__": str(notebook), "display": display}

    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") != "code":
            continue
        if index < args.start_cell:
            continue
        if args.stop_after_cell is not None and index > args.stop_after_cell:
            break
        source = "".join(cell.get("source", []))
        print(f"\n{'=' * 70}\nExecuting {notebook.name}, cell {index}\n{'=' * 70}", flush=True)
        exec(compile(source, f"{notebook}:cell-{index}", "exec"), namespace)


if __name__ == "__main__":
    main()
