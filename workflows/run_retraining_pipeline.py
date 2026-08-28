#!/usr/bin/env python3
"""Resumable, checksum-recorded execution of the LCMWR retraining stages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("LCMWR_PYTHON", sys.executable))
LOG_DIR = ROOT / "reproducibility" / "retraining_logs"
STATE_FILE = ROOT / "reproducibility" / "retraining_state.json"
TASKS = ("loi", "tg", "t5", "ul94")
DISPLAY = {"loi": "LOI", "tg": "Tg", "t5": "T5", "ul94": "UL94"}


def commands():
    runner = "scripts/feature_selection/run_motif_selection_notebook.py"
    figure_runner = "scripts/feature_selection/run_figure3_analysis_notebook.py"
    return {
        "composition_audit": [[str(PYTHON), "scripts/audit_composition_data.py"]],
        "unique_structures": [[str(PYTHON), "scripts/extract_unique_smiles.py", "--per-table-output-dir", "data/task_unique_smiles"]],
        "motif_vocabulary": [[str(PYTHON), runner, "motif_vocab"]],
        "feature_matrices": [[str(PYTHON), runner, task, "--stop-after-cell", "0"] for task in TASKS] + [[str(PYTHON), "scripts/validate_retraining_stage.py", "features"]],
        "motif_selection_750": [[str(PYTHON), runner, task] for task in TASKS] + [[str(PYTHON), "scripts/validate_retraining_stage.py", "selection"]],
        "figure3": [[str(PYTHON), figure_runner, task] for task in TASKS],
        "nested_model_comparison": [[str(PYTHON), runner, f"{task}_model_compare"] for task in TASKS] + [[str(PYTHON), "scripts/validate_retraining_stage.py", "models"]],
        "actual_best_oof": [[str(PYTHON), "scripts/model_comparison/rebuild_actual_best_oof.py"]],
        "figure4": [[str(PYTHON), "scripts/model_comparison/figure4_actual_best.py"]],
        "shap_figure5_6": [[str(PYTHON), "scripts/shap_analysis/run_actual_best_shap.py", task] for task in TASKS] + [[str(PYTHON), "scripts/validate_retraining_stage.py", "final"]],
        "release_validation": [["bash", "workflows/validate_release.sh"]],
    }


def expected_outputs(stage):
    mapping = {
        "composition_audit": ["reproducibility/composition_audit/composition_audit_summary.json"],
        "unique_structures": ["data/unique_smiles_for_fragments.csv"],
        "motif_vocabulary": ["results/local_vocab_parallel_threshold.csv", "results/local_vocab_parallel_threshold_stats.json"],
        "feature_matrices": [f"results/{task}_motif_select/{task}_vocab_weighted_v9.pkl" for task in TASKS],
        "motif_selection_750": [f"results/{task}_motif_select/plots_data/improved_750_complete_results.csv" for task in TASKS],
        "figure3": [f"results/{task}_motif_select/figure3_analysis/figure3_{task}_analysis_summary.csv" for task in TASKS],
        "nested_model_comparison": [f"results/model_compare/{DISPLAY[task]}/model_performance_summary.csv" for task in TASKS],
        "actual_best_oof": ["results/model_compare/actual_best_oof/actual_best_models.csv"],
        "figure4": ["results/model_compare/figure4_oof/figure4_oof_metrics_summary.csv"],
        "shap_figure5_6": [f"results/interpretability/{DISPLAY[task]}/run_metadata.json" for task in TASKS],
        "release_validation": [],
    }
    return [ROOT / item for item in mapping[stage]]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def input_signature():
    paths = [ROOT / "configs/scientific_protocol.json", ROOT / "environment.yml"]
    paths += sorted((ROOT / "dataset").glob("*.csv"))
    paths += sorted((ROOT / "scripts").rglob("*.py"))
    paths += sorted((ROOT / "scripts").rglob("*.ipynb"))
    paths += sorted((ROOT / "workflows").glob("*.py"))
    paths += sorted((ROOT / "workflows").glob("*.sh"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def preflight(stage_commands):
    required = [
        ROOT / "environment.yml",
        ROOT / "configs/scientific_protocol.json",
        ROOT / "dataset/LOI.csv",
        ROOT / "dataset/T5.csv",
        ROOT / "dataset/Tg.csv",
        ROOT / "dataset/UL-94.csv",
        ROOT / "dataset/Tg_GREA.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required workflow inputs: {missing}")
    python_executable = str(PYTHON) if PYTHON.is_absolute() else shutil.which(str(PYTHON))
    if not python_executable or not Path(python_executable).is_file():
        raise FileNotFoundError(f"Python executable not found: {PYTHON}")
    dataset_rows = {}
    for path in sorted((ROOT / "dataset").glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            dataset_rows[path.name] = max(sum(1 for _ in csv.reader(handle)) - 1, 0)
    return {
        "repository_root": str(ROOT),
        "python_executable": str(Path(python_executable).resolve()),
        "input_signature": input_signature(),
        "dataset_rows": dataset_rows,
        "stages": list(stage_commands),
        "results_writable": os.access(ROOT / "results", os.W_OK),
        "status": "ready",
    }


def load_state():
    return json.loads(STATE_FILE.read_text(encoding="utf-8")) if STATE_FILE.exists() else {"stages": {}}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate(stage):
    outputs = expected_outputs(stage)
    missing = [str(path) for path in outputs if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"{stage}: missing/empty outputs: {missing}")
    return [{"path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in outputs]


def main():
    stage_commands = commands()
    names = list(stage_commands)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-stage", choices=names, default=names[0])
    parser.add_argument("--through-stage", choices=names, default=names[-1])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Validate workflow inputs and exit without writing outputs.")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(stage_commands), ensure_ascii=False, indent=2))
        return
    start_index, end_index = names.index(args.from_stage), names.index(args.through_stage)
    if start_index > end_index:
        raise ValueError("--from-stage must not follow --through-stage")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    state.setdefault("environment", {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "matplotlib_cache": "/tmp/lcmwr_mpl",
        "numba_cache": "/tmp/lcmwr_numba",
    })
    signature = input_signature()
    environment = os.environ.copy()
    environment.update({
        "LCMWR_PYTHON": str(PYTHON),
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": "/tmp/lcmwr_mpl",
        "MPLBACKEND": "Agg",
        "QT_QPA_PLATFORM": "offscreen",
        "NUMBA_CACHE_DIR": "/tmp/lcmwr_numba",
        "PYTHONPATH": str(ROOT / "scripts" / "feature_selection") + (":" + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""),
    })
    for stage in names[start_index:end_index + 1]:
        previous = state["stages"].get(stage, {})
        if not args.force and previous.get("status") == "completed" and previous.get("input_signature") == signature:
            validate(stage)
            print(f"[{stage}] already completed with matching inputs; skipping", flush=True)
            continue
        started = time.time()
        state["stages"][stage] = {
            "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
            "input_signature": signature, "commands": stage_commands[stage],
        }
        save_state(state)
        log_path = LOG_DIR / f"{names.index(stage) + 1:02d}_{stage}.log"
        try:
            with log_path.open("w", encoding="utf-8") as log:
                for command in stage_commands[stage]:
                    print(f"[{stage}] {' '.join(command)}", flush=True)
                    log.write(f"COMMAND: {' '.join(command)}\n"); log.flush()
                    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                    assert process.stdout is not None
                    for line in process.stdout:
                        print(line, end="", flush=True); log.write(line); log.flush()
                    if process.wait() != 0:
                        raise subprocess.CalledProcessError(process.returncode, command)
            artifacts = validate(stage)
            state["stages"][stage].update({
                "status": "completed", "finished_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.time() - started, "artifacts": artifacts,
            })
            save_state(state)
        except Exception as exc:
            state["stages"][stage].update({
                "status": "failed", "finished_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.time() - started, "error": repr(exc),
            })
            save_state(state)
            raise


if __name__ == "__main__":
    main()
