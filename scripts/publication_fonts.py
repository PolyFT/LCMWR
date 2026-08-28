"""Resolve and register the Arial files used by publication figures."""

from __future__ import annotations

import os
from pathlib import Path

from matplotlib import font_manager


ENV_REGULAR = "LCMWR_ARIAL_REGULAR"
ENV_BOLD = "LCMWR_ARIAL_BOLD"
DEFAULT_REGULAR = Path("/mnt/c/Windows/Fonts/arial.ttf")
DEFAULT_BOLD = Path("/mnt/c/Windows/Fonts/arialbd.ttf")


def publication_font_paths() -> tuple[Path, Path]:
    regular = Path(os.environ.get(ENV_REGULAR, DEFAULT_REGULAR)).expanduser()
    bold = Path(os.environ.get(ENV_BOLD, DEFAULT_BOLD)).expanduser()
    missing = [path for path in (regular, bold) if not path.is_file()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Arial publication font file(s) not found: {names}. "
            f"Set {ENV_REGULAR} and {ENV_BOLD} to the regular and bold Arial files."
        )
    return regular.resolve(), bold.resolve()


def register_publication_fonts() -> tuple[Path, Path]:
    paths = publication_font_paths()
    for path in paths:
        font_manager.fontManager.addfont(str(path))
    return paths
