#!/usr/bin/env bash
set -euo pipefail

WORKFLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LCMWR_ROOT="$(cd -- "$WORKFLOW_DIR/.." && pwd)"

"$WORKFLOW_DIR/run_smoke_test.sh"

if find "$LCMWR_ROOT" -type f -size +50M \
    -not -path "$LCMWR_ROOT/.git/*" -print -quit | grep -q .; then
    echo "Release warning: files larger than 50 MiB require an explicit artifact policy." >&2
fi

echo "Lightweight release validation completed; production reproduction was not run."

