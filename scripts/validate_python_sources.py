#!/usr/bin/env python3
"""Read-only syntax validation for tracked Python source files."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    files = sorted(args.root.rglob("*.py"))
    failures = []
    for path in files:
        if any(part in {"__pycache__", ".venv", "venv"} for part in path.parts):
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            failures.append((path, exc))
    for path, exc in failures:
        print(f"FAIL {path}: {exc}")
    print(f"Validated {len(files)} Python files; failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

