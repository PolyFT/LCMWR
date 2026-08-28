#!/usr/bin/env bash
# Run T5 and UL94 only after the already-running Tg selection has completed.
# This avoids oversubscribing the 8 logical CPUs reserved by CV=2 / XGB=4.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LCMWR_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${LCMWR_PYTHON:-python}"

while tmux has-session -t motif_tg_rerun 2>/dev/null; do
    sleep 60
done

test -s "$LCMWR_ROOT/results/tg_motif_select/plots_data/improved_750_complete_results.csv"

for task in t5 ul94; do
    lower="$task"
    result_dir="$LCMWR_ROOT/results/${lower}_motif_select"
    mkdir -p "$result_dir/logs"
    log="$result_dir/logs/rerun_$(date +%Y%m%d_%H%M%S).log"
    echo "Starting $task at $(date -Is); log: $log"
    cd "$LCMWR_ROOT"
    PYTHONPATH="$LCMWR_ROOT/scripts/feature_selection${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON" "$SCRIPT_DIR/run_motif_selection_notebook.py" "$task" 2>&1 | tee "$log"
    test -s "$result_dir/plots_data/improved_750_complete_results.csv"
done
