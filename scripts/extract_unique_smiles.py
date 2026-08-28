#!/usr/bin/env python3
"""Extract unique SMILES from the four polymer property tables.

The script reads LOI, T5, Tg, and UL-94 tables, collects values from
smiles1, smiles2, and mix_smiles* columns, deduplicates them, and writes a
SMILES list for local chemical fragment generation.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DEFAULT_TABLES = ("LOI.csv", "T5.csv", "Tg.csv", "UL-94.csv")
DEFAULT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1")
TABLE_SLUGS = {
    "LOI.csv": "loi",
    "T5.csv": "t5",
    "Tg.csv": "tg",
    "UL-94.csv": "ul94",
}


def repo_motif_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def read_csv_with_fallback(path: Path) -> tuple[list[dict[str, str]], list[str], str]:
    last_error: Exception | None = None
    for encoding in DEFAULT_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
                return rows, list(reader.fieldnames or []), encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"failed to decode {path} with {DEFAULT_ENCODINGS}: {last_error}",
    )


def smiles_columns(fieldnames: list[str]) -> list[str]:
    return [
        col
        for col in fieldnames
        if col in {"smiles1", "smiles2"} or col.startswith("mix_smiles")
    ]


def collect_unique_smiles(dataset_dir: Path, tables: tuple[str, ...]) -> tuple[dict[str, set[str]], list[str]]:
    smiles_sources: dict[str, set[str]] = defaultdict(set)
    report: list[str] = []

    for table in tables:
        path = dataset_dir / table
        if not path.exists():
            raise FileNotFoundError(f"table not found: {path}")

        rows, fieldnames, encoding = read_csv_with_fallback(path)
        columns = smiles_columns(fieldnames)
        report.append(
            f"{table}: rows={len(rows)}, encoding={encoding}, smiles_columns={','.join(columns)}"
        )

        for row in rows:
            for col in columns:
                smiles = (row.get(col) or "").strip()
                if not smiles:
                    continue
                smiles_sources[smiles].add(f"{table}:{col}")

    return smiles_sources, report


def write_outputs(smiles_sources: dict[str, set[str]], output_csv: Path, output_txt: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    sorted_smiles = sorted(smiles_sources)

    with output_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["smiles", "source_columns"])
        writer.writeheader()
        for smiles in sorted_smiles:
            writer.writerow(
                {
                    "smiles": smiles,
                    "source_columns": ";".join(sorted(smiles_sources[smiles])),
                }
            )

    with output_txt.open("w", encoding="utf-8", newline="\n") as fh:
        for smiles in sorted_smiles:
            fh.write(f"{smiles}\n")


def write_per_table_outputs(
    smiles_sources: dict[str, set[str]],
    output_dir: Path,
    tables: tuple[str, ...],
) -> list[tuple[str, int, Path, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_table: dict[str, dict[str, set[str]]] = {
        table: defaultdict(set) for table in tables
    }

    for smiles, sources in smiles_sources.items():
        for source in sources:
            table, _, column = source.partition(":")
            if table in per_table:
                per_table[table][smiles].add(source if column else table)

    written = []
    for table in tables:
        slug = TABLE_SLUGS.get(table, Path(table).stem.lower().replace("-", ""))
        table_sources = per_table[table]
        output_csv = output_dir / f"{slug}_unique_smiles_for_fragments.csv"
        output_txt = output_dir / f"{slug}_unique_smiles_for_fragments.txt"
        write_outputs(table_sources, output_csv, output_txt)
        written.append((table, len(table_sources), output_csv, output_txt))
    return written


def parse_args() -> argparse.Namespace:
    motif_dir = repo_motif_dir()
    parser = argparse.ArgumentParser(
        description="Extract deduplicated SMILES from LOI/T5/Tg/UL-94 tables."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=motif_dir / "dataset",
        help="Directory containing LOI.csv, T5.csv, Tg.csv, and UL-94.csv.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=motif_dir / "data" / "unique_smiles_for_fragments.csv",
        help="CSV output with SMILES and source columns.",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=motif_dir / "data" / "unique_smiles_for_fragments.txt",
        help="Plain text output with one SMILES per line.",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=DEFAULT_TABLES,
        help="Property table filenames to read.",
    )
    parser.add_argument(
        "--per-table-output-dir",
        type=Path,
        default=None,
        help="Optional directory for per-property unique SMILES files used to build task-specific vocab.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = tuple(args.tables)
    smiles_sources, report = collect_unique_smiles(args.dataset_dir, tables)
    write_outputs(smiles_sources, args.output_csv, args.output_txt)
    per_table_written = []
    if args.per_table_output_dir is not None:
        per_table_written = write_per_table_outputs(
            smiles_sources,
            args.per_table_output_dir,
            tables,
        )

    print("Processed tables:")
    for line in report:
        print(f"  {line}")
    print(f"Unique SMILES: {len(smiles_sources)}")
    print(f"CSV output: {args.output_csv}")
    print(f"TXT output: {args.output_txt}")
    for table, count, output_csv, output_txt in per_table_written:
        print(f"{table} unique SMILES: {count}")
        print(f"  CSV output: {output_csv}")
        print(f"  TXT output: {output_txt}")


if __name__ == "__main__":
    main()
