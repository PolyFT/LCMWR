#!/usr/bin/env bash
# Production workflow. This is intentionally not used by CI.
set -euo pipefail

if [[ "${1:-}" != "--confirm-production-run" ]]; then
    echo "This command runs the expensive production workflow and writes results/." >&2
    echo "Usage: $0 --confirm-production-run" >&2
    exit 2
fi

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LCMWR_ROOT="$(cd -- "$WORKFLOW_DIR/.." && pwd)"
PYTHON="${LCMWR_PYTHON:-python}"
export LCMWR_PYTHON="$PYTHON"

cd "$LCMWR_ROOT"
exec "$PYTHON" workflows/run_retraining_pipeline.py
