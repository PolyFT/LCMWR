#!/usr/bin/env python3
"""Execute a Figure 3b-h notebook and retain SVG previews in its cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = {
    task: ROOT / "scripts" / "feature_selection" / f"figure3_{task}_motif_analysis.ipynb"
    for task in ("loi", "tg", "t5", "ul94")
}


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=NOTEBOOKS, type=str.lower)
    parser.add_argument(
        "--output-notebook-dir",
        type=Path,
        default=ROOT / "results" / "executed_notebooks" / "figure3",
        help="Directory for the executed notebook copy.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Legacy behavior: replace outputs in the source notebook.",
    )
    args = parser.parse_args()

    notebook_path = NOTEBOOKS[args.task]
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__", "__file__": str(notebook_path)}
    execution_count = 0

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        execution_count += 1
        cell["outputs"] = []
        print(f"Executing {notebook_path.name}, cell {index}", flush=True)
        exec(compile(source_text(cell), f"{notebook_path}:cell-{index}", "exec"), namespace)
        cell["execution_count"] = execution_count

        panel = cell.get("metadata", {}).get("figure3_panel")
        if panel:
            output_dir = namespace["data"]["paths"]["output_dir"]
            assets = sorted(output_dir.glob(f"Figure3{panel}_*.svg"))
            if len(assets) != 1:
                raise RuntimeError(f"Expected exactly one SVG for panel {panel}; found {assets}")
            cell["outputs"].append(
                {
                    "output_type": "display_data",
                    "data": {"image/svg+xml": assets[0].read_text(encoding="utf-8")},
                    "metadata": {},
                }
            )

    if args.in_place:
        output_path = notebook_path
    else:
        args.output_notebook_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_notebook_dir / notebook_path.name
    output_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Completed {notebook_path}; executed copy: {output_path}")


if __name__ == "__main__":
    main()
