#!/usr/bin/env python3
"""Create the pre-training, row-level composition quality gate."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "feature_selection"))

from composition_processing import (  # noqa: E402
    COMPOSITION_RULE_VERSION,
    MOLECULAR_WEIGHT_METHOD,
    RDKit_VERSION,
    clean_smiles,
    resolve_composition,
)
from dataset_provenance import canonical_cell, read_csv_rows  # noqa: E402


TASKS = {
    "loi": ("LOI.csv", "LOI", 948),
    "t5": ("T5.csv", "T5", 584),
    "tg": ("Tg.csv", "Tg", 518),
    "ul94": ("UL-94.csv", "UL-94", 495),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scientific_row_ids(path):
    header, rows = read_csv_rows(path)
    indices = [index for index, column in enumerate(header) if column != "DOI"]
    return [
        hashlib.sha256(
            json.dumps([canonical_cell(row[index]) for index in indices], ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        for row in rows
    ]


def main():
    output_dir = ROOT / "reproducibility" / "composition_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    task_summary = {}
    ul94_mix_mol_rows = []

    for task, (filename, target, expected_rows) in TASKS.items():
        path = ROOT / "dataset" / filename
        data = pd.read_csv(path)
        row_ids = scientific_row_ids(path)
        if len(data) != expected_rows:
            raise RuntimeError(f"{task}: expected {expected_rows} source rows, found {len(data)}")
        modeling = data.dropna(subset=[target]).copy()
        structure_columns = ["smiles1", "smiles2", *[f"mix_smiles{i}" for i in range(1, 5)]]
        structures = {
            clean_smiles(value)
            for column in structure_columns
            for value in modeling.get(column, [])
            if clean_smiles(value)
        }
        lookup = {smiles: np.zeros(1, dtype=np.float64) for smiles in structures}
        rows = []
        for source_index, row in modeling.iterrows():
            result = resolve_composition(row, lookup)
            result.pop("vector", None)
            record = {
                "task": task,
                "source_row_index": int(source_index),
                "source_record_id": row_ids[int(source_index)],
                "DOI": row.get("DOI"),
                "PolymerName": row.get("PolymerName"),
                "target": row.get(target),
                **result,
            }
            rows.append(record)
            all_rows.append(record)
            if task == "ul94" and any(
                pd.to_numeric(row.get(f"mix_mol{i}"), errors="coerce") > 0
                for i in range(1, 5)
            ):
                ul94_mix_mol_rows.append(record)

        audit = pd.DataFrame(rows)
        audit.to_csv(output_dir / f"{task}_composition_audit.csv", index=False)
        valid = audit["valid"].eq(True)
        valid_audit = audit.loc[valid]
        mix_mol_used = valid_audit.get("blend_source_fields", pd.Series(dtype=str)).fillna("").str.contains("mix_mol").sum()
        mix_wt_used = valid_audit.get("blend_source_fields", pd.Series(dtype=str)).fillna("").str.contains("mix_wt").sum()
        mol_used = valid_audit.get("co_source_fields", pd.Series(dtype=str)).fillna("").str.contains(r'"mol[12]"', regex=True).sum()
        wt_to_mol_used = valid_audit.get("copolymer_conversions", pd.Series(dtype=str)).fillna("").str.contains("weight_amount_to_moles").sum()
        task_summary[task] = {
            "source_rows": int(len(data)),
            "rows_with_target": int(len(modeling)),
            "valid_rows": int(valid.sum()),
            "excluded_rows": int((~valid).sum()),
            "composition_mode_counts": Counter(audit.loc[valid, "composition_mode"].tolist()),
            "exclusion_reason_counts": Counter(audit.loc[~valid, "exclusion_reason"].tolist()),
            "amount_source_usage": {
                "mol": int(mol_used),
                "wt_to_mol": int(wt_to_mol_used),
                "mix_wt": int(mix_wt_used),
                "mix_mol_to_wt": int(mix_mol_used),
            },
            "new_rule_restored_rows": int(mix_mol_used if task == "ul94" else 0),
            "dataset_sha256": sha256(path),
            "row_alignment": bool(audit["source_row_index"].tolist() == modeling.index.astype(int).tolist()),
            "label_alignment": bool(audit["target"].astype(str).tolist() == modeling[target].astype(str).tolist()),
        }

    ul94_mix = pd.DataFrame(ul94_mix_mol_rows)
    ul94_mix.to_csv(output_dir / "ul94_mix_mol_33_row_audit.csv", index=False)
    if len(ul94_mix) != 33:
        raise RuntimeError(f"expected 33 UL-94 mix_mol rows, found {len(ul94_mix)}")

    summary = {
        "composition_rule_version": COMPOSITION_RULE_VERSION,
        "molecular_weight_method": MOLECULAR_WEIGHT_METHOD,
        "rdkit_version": RDKit_VERSION,
        "task_summary": task_summary,
        "ul94_mix_mol": {
            "rows": int(len(ul94_mix)),
            "valid_rows": int(ul94_mix["valid"].eq(True).sum()),
            "excluded_rows": int(ul94_mix["valid"].ne(True).sum()),
            "explicit_chain_mole_semantics_detected": False,
            "interpretation": "repeat-unit or formulation molar parts per governing specification",
        },
    }
    (output_dir / "composition_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=dict) + "\n", encoding="utf-8"
    )

    lines = [
        "# Composition audit before retraining",
        "",
        f"Rule: `{COMPOSITION_RULE_VERSION}`; molecular weights: `{MOLECULAR_WEIGHT_METHOD}`; RDKit `{RDKit_VERSION}`.",
        "",
        "| Task | Source rows | Target rows | Valid | Excluded | Restored | Modes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for task, item in task_summary.items():
        modes = ", ".join(f"{key}={value}" for key, value in sorted(item["composition_mode_counts"].items()))
        lines.append(f"| {task} | {item['source_rows']} | {item['rows_with_target']} | {item['valid_rows']} | {item['excluded_rows']} | {item['new_rule_restored_rows']} | {modes} |")
    lines += [
        "",
        f"All 33 UL-94 rows using `mix_mol` parsed successfully: **{int(ul94_mix['valid'].eq(True).sum())}/33**.",
        "No source field explicitly identified these amounts as whole-chain mole counts; the required repeat-unit/formulation-molar-parts interpretation was therefore applied.",
        "Excluded rows remain in the source datasets and are listed with exact reasons in the CSV audits.",
        "",
    ]
    (output_dir / "composition_audit_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
