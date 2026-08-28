#!/usr/bin/env python3
"""Verify that a dataset revision changes only declared source metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FEATURE_SELECTION_DIR = SCRIPT_DIR / "feature_selection"
sys.path.insert(0, str(FEATURE_SELECTION_DIR))

from dataset_provenance import compare_datasets, dataset_fingerprints  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Frozen or previously released CSV.")
    parser.add_argument("candidate", type=Path, help="CSV containing updated metadata.")
    parser.add_argument(
        "--metadata-column",
        action="append",
        default=None,
        help="Column allowed to change; repeat as needed (default: DOI).",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for the machine-readable validation report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_columns = tuple(args.metadata_column or ["DOI"])
    report = compare_datasets(args.reference, args.candidate, metadata_columns)
    report["reference_fingerprints"] = dataset_fingerprints(
        args.reference, metadata_columns
    )
    report["candidate_fingerprints"] = dataset_fingerprints(
        args.candidate, metadata_columns
    )

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["valid_metadata_only_change"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
