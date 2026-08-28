"""Dataset fingerprints that distinguish scientific content from DOI metadata."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1")


def _read_text(path: Path) -> str:
    payload = path.read_bytes()
    last_error: UnicodeDecodeError | None = None
    for encoding in DEFAULT_ENCODINGS:
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def read_csv_rows(path: str | Path) -> tuple[list[str], list[list[str]]]:
    """Read a CSV with the same encoding fallbacks used by the workflow."""
    path = Path(path)
    with_rows = [
        row
        for row in csv.reader(io.StringIO(_read_text(path)))
        if any(cell.strip() for cell in row)
    ]
    if not with_rows:
        raise ValueError(f"Dataset is empty: {path}")
    header = with_rows[0]
    rows = with_rows[1:]
    width = len(header)
    for line_number, row in enumerate(rows, start=2):
        if len(row) != width:
            raise ValueError(
                f"CSV row width mismatch in {path} at line {line_number}: "
                f"expected {width}, got {len(row)}"
            )
    return header, rows


def canonical_cell(value: str) -> str:
    """Normalize numeric display only; preserve nonnumeric scientific text."""
    value = value.strip()
    if not value:
        return ""
    try:
        number = Decimal(value)
    except InvalidOperation:
        return value
    if not number.is_finite():
        return value
    if number == 0:
        return "0"
    normalized = format(number.normalize(), "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def _digest_records(records: Iterable[object]) -> str:
    digest = hashlib.sha1()
    for record in records:
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()[:12]


def dataset_fingerprints(
    path: str | Path,
    metadata_columns: Sequence[str] = ("DOI",),
) -> dict[str, object]:
    """Return separate ordered fingerprints for scientific and source fields."""
    path = Path(path)
    header, rows = read_csv_rows(path)
    missing = [column for column in metadata_columns if column not in header]
    if missing:
        raise ValueError(f"Missing metadata columns in {path}: {missing}")

    metadata_indices = {header.index(column) for column in metadata_columns}
    scientific_indices = [i for i in range(len(header)) if i not in metadata_indices]
    ordered_metadata_indices = [header.index(column) for column in metadata_columns]
    scientific_header = [header[i] for i in scientific_indices]

    scientific_records = [scientific_header]
    scientific_records.extend(
        [canonical_cell(row[i]) for i in scientific_indices] for row in rows
    )
    metadata_records = [list(metadata_columns)]
    metadata_records.extend(
        [row[i].strip() for i in ordered_metadata_indices] for row in rows
    )

    return {
        "row_count": len(rows),
        "column_count": len(header),
        "scientific_columns": scientific_header,
        "scientific_data_hash": _digest_records(scientific_records),
        "source_metadata_hash": _digest_records(metadata_records),
    }


def compare_datasets(
    reference: str | Path,
    candidate: str | Path,
    metadata_columns: Sequence[str] = ("DOI",),
) -> dict[str, object]:
    """Compare ordered CSV rows while allowing only declared metadata changes."""
    reference_header, reference_rows = read_csv_rows(reference)
    candidate_header, candidate_rows = read_csv_rows(candidate)
    report: dict[str, object] = {
        "reference": str(Path(reference)),
        "candidate": str(Path(candidate)),
        "metadata_columns": list(metadata_columns),
        "schema_equal": reference_header == candidate_header,
        "reference_rows": len(reference_rows),
        "candidate_rows": len(candidate_rows),
        "row_count_equal": len(reference_rows) == len(candidate_rows),
        "scientific_difference_rows": [],
        "metadata_difference_rows": [],
    }
    if reference_header != candidate_header:
        report["valid_metadata_only_change"] = False
        return report

    metadata_indices = {reference_header.index(column) for column in metadata_columns}
    scientific_indices = [
        i for i in range(len(reference_header)) if i not in metadata_indices
    ]
    metadata_indices_ordered = sorted(metadata_indices)

    scientific_differences: list[int] = []
    metadata_differences: list[int] = []
    for line_number, (old, new) in enumerate(
        zip(reference_rows, candidate_rows), start=2
    ):
        if [canonical_cell(old[i]) for i in scientific_indices] != [
            canonical_cell(new[i]) for i in scientific_indices
        ]:
            scientific_differences.append(line_number)
        if [old[i].strip() for i in metadata_indices_ordered] != [
            new[i].strip() for i in metadata_indices_ordered
        ]:
            metadata_differences.append(line_number)

    report["scientific_difference_rows"] = scientific_differences
    report["metadata_difference_rows"] = metadata_differences
    report["valid_metadata_only_change"] = bool(
        report["schema_equal"]
        and report["row_count_equal"]
        and not scientific_differences
    )
    return report
