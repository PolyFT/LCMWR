#!/usr/bin/env bash
set -euo pipefail

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LCMWR_ROOT="$(cd -- "$WORKFLOW_DIR/.." && pwd)"
PYTHON="${LCMWR_PYTHON:-python}"

SMOKE_TMP="$(mktemp -d "${TMPDIR:-/tmp}/lcmwr-smoke.XXXXXX")"
cleanup() { rm -rf -- "$SMOKE_TMP"; }
trap cleanup EXIT HUP INT TERM

# Keep all interpreter and plotting state outside the checkout.
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$SMOKE_TMP/matplotlib"
export NUMBA_CACHE_DIR="$SMOKE_TMP/numba"
export XDG_CACHE_HOME="$SMOKE_TMP/xdg-cache"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$XDG_CACHE_HOME"

cd "$LCMWR_ROOT"
"$PYTHON" scripts/validate_python_sources.py
"$PYTHON" -m unittest discover -s tests -p 'test_*.py' -v
"$PYTHON" scripts/validate_paper_alignment.py
"$PYTHON" workflows/run_retraining_pipeline.py --preflight >/dev/null
"$PYTHON" scripts/extract_unique_smiles.py --help >/dev/null
"$PYTHON" scripts/validate_dataset_metadata.py --help >/dev/null
echo "LCMWR smoke validation passed."
