#!/usr/bin/env python3
"""Run the isolated Tg_GREA motif-to-OOF experiment.

The implementation intentionally reuses the current motif-generation,
feature-selection and model-comparison notebook algorithms, but routes every
input-derived artifact to ``LCMWR/experiments/tg_grea``.  Existing four-task
results and caches are never used as inputs or outputs.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import logging
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
LCMWR_ROOT = EXPERIMENT_ROOT.parents[1]
SOURCE_DATASET = LCMWR_ROOT / "dataset" / "Tg_GREA.csv"
INPUT_DIR = EXPERIMENT_ROOT / "input"
DATA_DIR = EXPERIMENT_ROOT / "data"
RESULTS_DIR = EXPERIMENT_ROOT / "results"
LOG_DIR = EXPERIMENT_ROOT / "logs"
FEATURE_DIR = RESULTS_DIR / "features"
SELECT_DIR = RESULTS_DIR / "feature_selection"
MODEL_DIR = RESULTS_DIR / "model_compare" / "Tg_GREA"
BEST_DIR = RESULTS_DIR / "best_model"

PREPARED_DATA = INPUT_DIR / "tg_grea_homopolymers.csv"
UNIQUE_SMILES = DATA_DIR / "unique_smiles_for_fragments.csv"
VOCAB_FILE = RESULTS_DIR / "local_vocab_parallel_threshold.csv"
VOCAB_STATS = RESULTS_DIR / "local_vocab_parallel_threshold_stats.json"
FREQUENCY_CACHE = RESULTS_DIR / "local_vocab_full_frequency_cache.csv"
FREQUENCY_CACHE_STATS = RESULTS_DIR / "local_vocab_full_frequency_cache_stats.json"
CANDIDATE_CACHE = DATA_DIR / "candidate_fragments_v1.joblib"
FEATURE_CACHE = FEATURE_DIR / "tg_grea_vocab_weighted_v1.pkl"
PROCESSED_DATA = FEATURE_DIR / "tg_grea_vocab_processed_data.csv"
FEATURE_MATRIX = FEATURE_DIR / "tg_grea_vocab_weighted_feature_matrix.csv"
RAW_FEATURES = FEATURE_DIR / "tg_grea_vocab_weighted_raw_features.csv"
AUDIT_FILE = FEATURE_DIR / "tg_grea_composition_audit.csv"
FEATURE_STATS = FEATURE_DIR / "tg_grea_vocab_feature_matrix_stats.json"
MATCHING_DIAGNOSTICS = FEATURE_DIR / "tg_grea_vocab_matching_diagnostics.json"

RANDOM_STATE = 48
OUTER_CV_SPLITS = 5
INNER_CV_SPLITS = 3
N_ITER_SEARCH = 20
N_JOBS = 2
XGB_N_JOBS = 4


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_dirs() -> None:
    for directory in (INPUT_DIR, DATA_DIR, RESULTS_DIR, LOG_DIR, FEATURE_DIR, SELECT_DIR, MODEL_DIR, BEST_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _normalise_smiles(value: str) -> str:
    return value.strip().replace("[Fr]", "[H]").replace("[Rb]", "[H]")


def _read_grea_rows(source_path: Path) -> list[tuple[str, float]]:
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Empty Tg_GREA file: {source_path}")
    if rows[0] == ["smiles1", "Tg"]:
        rows = rows[1:]
    parsed: list[tuple[str, float]] = []
    for number, row in enumerate(rows, start=1):
        if len(row) != 2:
            raise ValueError(f"Tg_GREA row {number} must have two columns, got {len(row)}")
        smiles = row[0].strip()
        if not smiles:
            raise ValueError(f"Tg_GREA row {number} has an empty SMILES")
        try:
            target = float(row[1])
        except ValueError as exc:
            raise ValueError(f"Tg_GREA row {number} has non-numeric Tg={row[1]!r}") from exc
        if not np.isfinite(target):
            raise ValueError(f"Tg_GREA row {number} has non-finite Tg")
        if Chem.MolFromSmiles(_normalise_smiles(smiles)) is None:
            raise ValueError(f"Tg_GREA row {number} contains an invalid SMILES: {smiles!r}")
        parsed.append((smiles, target))
    if len({smiles for smiles, _ in parsed}) != len(parsed):
        raise ValueError("Tg_GREA contains duplicate SMILES; independent vocabulary input must be unique")
    return parsed


def prepare_input(source_path: Path = SOURCE_DATASET, prepared_path: Path = PREPARED_DATA, unique_path: Path = UNIQUE_SMILES) -> dict[str, Any]:
    """Add the requested header and materialize a homopolymer-only working table."""
    rows = _read_grea_rows(source_path)
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        first_row = next(csv.reader(handle))
    if first_row != ["smiles1", "Tg"]:
        original = source_path.read_text(encoding="utf-8")
        source_path.write_text("smiles1,Tg\n" + original, encoding="utf-8", newline="")

    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    unique_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_rows = []
    for index, (smiles, target) in enumerate(rows):
        payload = json.dumps([smiles, format(target, ".12g")], ensure_ascii=False, separators=(",", ":"))
        prepared_rows.append({
            "source_row_index": index,
            "source_record_id": sha256_text(payload),
            "sample_id": f"GREA_{index + 1:05d}",
            "DOI": "",
            "type": "homopolymer",
            "polymer_family": "GREA",
            "PolymerName": f"GREA_{index + 1:05d}",
            "smiles1": smiles,
            "wt1": 100.0,
            "Tg": target,
        })
    prepared = pd.DataFrame(prepared_rows)
    prepared.to_csv(prepared_path, index=False)
    pd.DataFrame({"smiles": [smiles for smiles, _ in rows]}).to_csv(unique_path, index=False)
    manifest = {
        "source_dataset": str(source_path),
        "source_sha1": sha1_file(source_path),
        "prepared_dataset": str(prepared_path),
        "unique_smiles": str(unique_path),
        "n_samples": len(prepared),
        "n_unique_smiles": int(prepared["smiles1"].nunique()),
        "all_homopolymers": True,
    }
    (prepared_path.parent / "prepare_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_experiment_manifest() -> None:
    """Record the fixed scientific protocol and experiment-local layout."""
    manifest = {
        "experiment": "Tg_GREA",
        "source_dataset": str(SOURCE_DATASET),
        "task_type": "regression",
        "target_column": "Tg",
        "composition": {"type": "homopolymer", "component_column": "smiles1", "weight_amount": 100.0},
        "motif_generation": {
            "maximum_atoms": 17,
            "chon_minimum_support_ratio": 0.01,
            "other_elements_minimum_support_ratio": 0.005,
            "replace_ports_with_h": True,
        },
        "feature_selection_grid": {
            "frequency_thresholds": [0.01, 0.03, 0.05, 0.08, 0.10, 0.15],
            "variance_thresholds": [0.001, 0.005, 0.01, 0.02, 0.05],
            "mutual_information_percentiles": [30, 40, 50, 60, 70],
            "correlation_thresholds": [0.80, 0.85, 0.90, 0.95, 0.98],
        },
        "nested_cross_validation": {"random_state": RANDOM_STATE, "outer_splits": OUTER_CV_SPLITS, "inner_splits": INNER_CV_SPLITS, "random_search_iterations": N_ITER_SEARCH},
        "outputs_root": str(RESULTS_DIR),
    }
    (RESULTS_DIR / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _notebook_source(path: Path, cell_index: int) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cell = notebook["cells"][cell_index]
    if cell.get("cell_type") != "code":
        raise ValueError(f"Expected code cell {cell_index} in {path}")
    return "".join(cell["source"])


def _candidate_cache_signature(config: Any) -> dict[str, Any]:
    keys = (
        "min_size", "max_size", "max_candidates", "max_candidates_per_mol",
        "max_atom_sets_per_mol", "include_rings", "include_aromatic_rings",
        "complete_partial_rings", "complete_aromatic_rings",
        "complete_aliphatic_rings", "replace_ports_with_h",
    )
    return {
        "input_sha1": sha1_file(UNIQUE_SMILES),
        "candidate_config": {key: getattr(config, key) for key in keys},
    }


def _candidate_molecule_mask(query: Chem.Mol, molecule_bitsets: list[int], all_molecules_mask: int) -> int:
    probe = Chem.PatternFingerprint(query, fpSize=len(molecule_bitsets))
    mask = all_molecules_mask
    bits = probe.GetOnBits()
    for bit in sorted(bits, key=lambda item: molecule_bitsets[item].bit_count()):
        mask &= molecule_bitsets[bit]
        if not mask:
            break
    return mask


def _count_fragment_frequencies(records: list[dict[str, Any]], candidate_sources: dict[str, set[str]], config: Any, namespace: dict[str, Any]) -> pd.DataFrame:
    mols = [Chem.MolFromSmiles(record["processed_smiles"]) for record in records]
    n_mols = len(mols)
    fp_size = 2048
    molecule_bitsets = [0] * fp_size
    for index, mol in enumerate(mols):
        if mol is None:
            continue
        for bit in Chem.PatternFingerprint(mol, fpSize=fp_size).GetOnBits():
            molecule_bitsets[bit] |= 1 << index
    all_molecules_mask = (1 << n_mols) - 1
    candidate_smiles = sorted(candidate_sources, key=lambda value: (len(value), value))
    safe_matches = namespace["safe_substruct_matches"]
    fragment_mol_from_smiles = namespace["fragment_mol_from_smiles"]
    fragment_ring_flags = namespace["fragment_ring_flags"]
    fragment_elements = namespace["fragment_elements"]

    def count_one(smiles: str) -> dict[str, Any] | None:
        query = fragment_mol_from_smiles(smiles)
        if query is None:
            return None
        support_mol_count = 0
        occurrence_count = 0
        candidates = _candidate_molecule_mask(query, molecule_bitsets, all_molecules_mask)
        while candidates:
            low_bit = candidates & -candidates
            index = low_bit.bit_length() - 1
            candidates ^= low_bit
            mol = mols[index]
            if mol is None:
                continue
            matches = safe_matches(mol, query)
            if matches:
                support_mol_count += 1
                occurrence_count += len(matches)
        has_ring, has_aromatic_ring, has_aliphatic_ring = fragment_ring_flags(query, smiles)
        return {
            "smiles": smiles,
            "support_mol_count": support_mol_count,
            "occurrence_count": occurrence_count,
            "support_ratio": support_mol_count / n_mols if n_mols else 0,
            "elements": ";".join(fragment_elements(query)),
            "num_atoms": query.GetNumAtoms(),
            "num_bonds": query.GetNumBonds(),
            "is_ring": has_ring,
            "is_aromatic": has_aromatic_ring,
            "has_aromatic_ring": has_aromatic_ring,
            "has_aliphatic_ring": has_aliphatic_ring,
            "source_types": ";".join(sorted(candidate_sources[smiles])),
        }

    namespace["logger"].info("开始全局频率统计：%d 个候选 × %d 个分子", len(candidate_smiles), n_mols)
    rows = Parallel(n_jobs=-1, prefer="threads")(delayed(count_one)(smiles) for smiles in candidate_smiles)
    rows = [row for row in rows if row is not None]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(
        by=["support_mol_count", "occurrence_count", "num_atoms", "smiles"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    frame.insert(0, "vocab_id", range(1, len(frame) + 1))
    return frame


def build_vocab() -> None:
    if not UNIQUE_SMILES.exists():
        raise FileNotFoundError("Run prepare before vocab")
    notebook_path = LCMWR_ROOT / "scripts" / "motif_generating.ipynb"
    module_name = "__tg_grea_vocab__"
    module = types.ModuleType(module_name)
    module.__file__ = str(notebook_path)
    module.display = lambda value: None
    # dataclasses resolves postponed annotations via sys.modules, so notebook
    # cells must run in a registered module rather than an anonymous dict.
    sys.modules[module_name] = module
    namespace = module.__dict__
    for index in range(1, 9):
        exec(compile(_notebook_source(notebook_path, index), f"{notebook_path}:cell-{index}", "exec"), namespace)
    config = namespace["FragmentConfig"](
        motif_root=EXPERIMENT_ROOT,
        input_file=UNIQUE_SMILES,
        smiles_column="smiles",
        output_file=VOCAB_FILE,
        stats_file=VOCAB_STATS,
        frequency_cache_file=FREQUENCY_CACHE,
        frequency_cache_stats_file=FREQUENCY_CACHE_STATS,
        result_cache_files=(VOCAB_FILE,),
        min_size=1,
        max_size=17,
        chon_min_support_ratio=0.01,
        other_min_support_ratio=0.005,
        min_support_floor=1,
        n_jobs=-1,
        parallel_prefer="threads",
        visualize=False,
    )
    original_generate = namespace["generate_candidate_fragments"]

    def generate_with_cache(records: list[dict[str, Any]], run_config: Any):
        signature = _candidate_cache_signature(run_config)
        if CANDIDATE_CACHE.exists():
            try:
                cached = joblib.load(CANDIDATE_CACHE)
                if cached.get("signature") == signature:
                    return cached["candidate_sources"], cached["generation_stats"]
            except Exception:
                pass
        candidate_sources, generation_stats = original_generate(records, run_config)
        joblib.dump({
            "signature": signature,
            "candidate_sources": candidate_sources,
            "generation_stats": generation_stats,
        }, CANDIDATE_CACHE)
        return candidate_sources, generation_stats

    namespace["generate_candidate_fragments"] = generate_with_cache
    namespace["count_fragment_frequencies"] = lambda records, candidate_sources, run_config: _count_fragment_frequencies(
        records, candidate_sources, run_config, namespace
    )
    namespace["run_pipeline"](config)


def _load_vocab_queries() -> tuple[list[str], list[Chem.Mol]]:
    vocabulary = pd.read_csv(VOCAB_FILE)
    if "passed_support_filter" in vocabulary.columns:
        vocabulary = vocabulary.loc[vocabulary["passed_support_filter"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    column = next((name for name in ("fragment_smiles", "canonical_smiles", "smiles", "fragment", "Vocab") if name in vocabulary.columns), None)
    if column is None:
        raise ValueError("GREA vocabulary has no recognized SMILES column")
    names: list[str] = []
    queries: list[Chem.Mol] = []
    for value in dict.fromkeys(vocabulary[column].dropna().astype(str).map(str.strip)):
        if not value:
            continue
        query = Chem.MolFromSmiles(_normalise_smiles(value), sanitize=True)
        if query is not None:
            names.append(value)
            queries.append(query)
    if not names:
        raise ValueError("GREA vocabulary has no valid motif queries")
    return names, queries


def _count_motifs(smiles: str, queries: list[Chem.Mol]) -> np.ndarray:
    molecule = Chem.MolFromSmiles(_normalise_smiles(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES after preparation: {smiles!r}")
    return np.asarray([len(molecule.GetSubstructMatches(query, uniquify=True)) for query in queries], dtype=np.float32)


def build_features(force: bool = False) -> dict[str, Any]:
    if not PREPARED_DATA.exists() or not VOCAB_FILE.exists():
        raise FileNotFoundError("Run prepare and vocab before features")
    data_hash = sha1_file(PREPARED_DATA)
    vocab_hash = sha1_file(VOCAB_FILE)
    if FEATURE_CACHE.exists() and not force:
        cached = joblib.load(FEATURE_CACHE)
        if cached.get("data_hash") == data_hash and cached.get("vocab_hash") == vocab_hash:
            return cached
    processed = pd.read_csv(PREPARED_DATA)
    names, queries = _load_vocab_queries()
    vectors = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_count_motifs)(smiles, queries) for smiles in processed["smiles1"]
    )
    matrix = pd.DataFrame(np.vstack(vectors), columns=names)
    active = matrix.columns[(matrix.abs() > 1e-6).any(axis=0)].tolist()
    matrix = matrix.loc[:, active].copy()
    if matrix.empty:
        raise ValueError("No non-zero GREA motif features were produced")
    processed["composition_valid"] = True
    processed["composition_mode"] = "homopolymer"
    processed["co_n_components"] = 1
    processed["co_mole_fractions"] = "[1.0]"
    processed["blend_weight_fractions"] = "[1.0]"
    processed["composition_rule_version"] = "homopolymer_direct_count_v1"
    processed.to_csv(PROCESSED_DATA, index=False)
    processed.to_csv(AUDIT_FILE, index=False)
    matrix.to_csv(FEATURE_MATRIX, index=False)
    pd.concat([processed.reset_index(drop=True), matrix], axis=1).to_csv(RAW_FEATURES, index=False)
    diagnostics = {
        "data_file": str(PREPARED_DATA),
        "data_hash": data_hash,
        "vocab_file": str(VOCAB_FILE),
        "vocab_hash": vocab_hash,
        "n_samples": int(len(processed)),
        "unique_component_smiles_count": int(processed["smiles1"].nunique()),
        "valid_vocab_count": len(names),
        "active_vocab_count": len(active),
        "all_zero_vocab_count": len(names) - len(active),
        "composition_mode_counts": {"homopolymer": int(len(processed))},
    }
    MATCHING_DIAGNOSTICS.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    FEATURE_STATS.write_text(json.dumps({**diagnostics, "feature_matrix_shape": list(matrix.shape)}, indent=2), encoding="utf-8")
    output = {
        "processed_data": processed,
        "feature_matrix": matrix,
        "target_variable": processed["Tg"].astype(float),
        "feature_names": active,
        "data_hash": data_hash,
        "vocab_hash": vocab_hash,
    }
    joblib.dump(output, FEATURE_CACHE)
    return output


def run_feature_selection() -> None:
    feature_data = build_features()
    notebook_path = LCMWR_ROOT / "scripts" / "feature_selection" / "Tg_motif_select.ipynb"
    namespace: dict[str, Any] = {
        "__name__": "__tg_grea_selection__",
        "__file__": str(notebook_path),
        "display": lambda value: None,
        "MOTIF_ROOT": EXPERIMENT_ROOT,
        "MODEL_RESULTS_DIR": SELECT_DIR,
        "feature_data": feature_data,
    }
    exec(compile(_notebook_source(notebook_path, 1), f"{notebook_path}:cell-1", "exec"), namespace)


def _model_namespace() -> dict[str, Any]:
    notebook_path = LCMWR_ROOT / "scripts" / "model_comparison" / "tg_model_compare.ipynb"
    module_name = "__tg_grea_model_compare__"
    module = types.ModuleType(module_name)
    module.__file__ = str(notebook_path)
    module.display = lambda value: None
    sys.modules[module_name] = module
    namespace = module.__dict__
    namespace.update({
        "Path": Path,
        "RANDOM_STATE": RANDOM_STATE,
        "OUTER_CV_SPLITS": OUTER_CV_SPLITS,
        "INNER_CV_SPLITS": INNER_CV_SPLITS,
        "N_ITER_SEARCH": N_ITER_SEARCH,
        "N_JOBS": N_JOBS,
        "XGB_N_JOBS": XGB_N_JOBS,
        "USE_RESULT_CACHE": True,
        "FORCE_RECOMPUTE": False,
        "CACHE_VERSION": 5,
        "PROJECT_ROOT": EXPERIMENT_ROOT,
        "RESULTS_DIR": RESULTS_DIR,
        "OUTPUT_ROOT": MODEL_DIR.parent,
        "TASK_CONFIG": {
            "Tg_GREA": {
                "input_path": SELECT_DIR / "best_improved_final_features_with_target.csv",
                "target_column": "Tg",
                "task_type": "regression",
                "processed_data_path": PROCESSED_DATA,
                "feature_matrix_path": SELECT_DIR / "best_improved_final_features_matrix.csv",
            }
        },
        "RUN_TASKS": ["Tg_GREA"],
    })
    for index in range(2, 8):
        exec(compile(_notebook_source(notebook_path, index), f"{notebook_path}:cell-{index}", "exec"), namespace)
    return namespace


def run_model_comparison() -> None:
    final_input = SELECT_DIR / "best_improved_final_features_with_target.csv"
    if not final_input.exists():
        raise FileNotFoundError("Run select before model_compare")
    namespace = _model_namespace()
    notebook_path = LCMWR_ROOT / "scripts" / "model_comparison" / "tg_model_compare.ipynb"
    for index in (8, 9):
        exec(compile(_notebook_source(notebook_path, index), f"{notebook_path}:cell-{index}", "exec"), namespace)
    augment_model_oof()


def augment_model_oof() -> dict[str, Any]:
    """Attach stable GREA source identifiers to the all-model OOF table.

    The shared comparison notebook intentionally writes generic ``sample_index``
    values.  This experiment-local companion makes those OOF rows auditable
    against the prepared homopolymer table without modifying the notebook's
    artifacts or any other task.
    """
    predictions_path = MODEL_DIR / "cv_predictions.csv"
    summary_path = MODEL_DIR / "model_performance_summary.csv"
    if not predictions_path.exists() or not summary_path.exists() or not PROCESSED_DATA.exists():
        raise FileNotFoundError("Model-comparison OOF inputs are incomplete")

    predictions = pd.read_csv(predictions_path)
    processed = pd.read_csv(PROCESSED_DATA).reset_index(drop=True)
    if "sample_index" not in predictions.columns:
        raise ValueError("Model-comparison OOF file has no sample_index column")
    sample_index = pd.to_numeric(predictions["sample_index"], errors="raise").astype(int)
    if len(predictions) != len(processed) or sample_index.nunique() != len(processed):
        raise ValueError("Model-comparison OOF rows are not one-to-one with GREA processed data")
    if set(sample_index) != set(range(len(processed))):
        raise ValueError("Model-comparison OOF sample_index does not cover every GREA row")

    source_columns = ["source_row_index", "source_record_id", "sample_id", "smiles1", "Tg"]
    source = processed.loc[sample_index.to_numpy(), source_columns].reset_index(drop=True)
    augmented = predictions.reset_index(drop=True).copy()
    for column in reversed(source_columns):
        augmented.insert(0, column, source[column].to_numpy())

    summary = pd.read_csv(summary_path)
    successful_models = summary.loc[summary["status"].eq("success"), "model"].astype(str).tolist()
    if len(successful_models) != 11:
        raise RuntimeError(f"Expected 11 successful GREA models, got {len(successful_models)}")
    for model in successful_models:
        fold_column = f"{model}_fold"
        if model not in augmented.columns or fold_column not in augmented.columns:
            raise ValueError(f"OOF table is missing prediction or fold column for {model}")
        if augmented[model].isna().any() or augmented[fold_column].isna().any():
            raise ValueError(f"OOF prediction or fold assignment is missing for {model}")
        folds = sorted(pd.to_numeric(augmented[fold_column], errors="raise").astype(int).unique().tolist())
        if folds != list(range(1, OUTER_CV_SPLITS + 1)):
            raise ValueError(f"OOF folds for {model} must be 1..{OUTER_CV_SPLITS}, got {folds}")

    output_path = MODEL_DIR / "cv_predictions_with_source_ids.csv"
    augmented.to_csv(output_path, index=False)
    audit = {
        "n_samples": int(len(augmented)),
        "n_successful_models": len(successful_models),
        "models": successful_models,
        "all_models_complete": True,
        "output_file": str(output_path.relative_to(EXPERIMENT_ROOT)),
    }
    (MODEL_DIR / "oof_integrity.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def build_best_model() -> dict[str, Any]:
    summary_path = MODEL_DIR / "model_performance_summary.csv"
    params_path = MODEL_DIR / "best_params.json"
    final_input = SELECT_DIR / "best_improved_final_features_with_target.csv"
    if not all(path.exists() for path in (summary_path, params_path, final_input)):
        raise FileNotFoundError("Run model_compare before best_model")
    namespace = _model_namespace()
    summary = pd.read_csv(summary_path)
    usable = summary.loc[summary["status"].eq("success")].copy()
    usable["R2_mean"] = pd.to_numeric(usable["R2_mean"], errors="coerce")
    if usable["R2_mean"].notna().sum() == 0:
        raise RuntimeError("No successful GREA model with finite R2")
    best_model = str(usable.loc[usable["R2_mean"].idxmax(), "model"])
    params = json.loads(params_path.read_text(encoding="utf-8"))[best_model]
    fold_params = {int(item["fold"]): item["best_params"] for item in params["best_params_by_fold"]}
    config = namespace["TASK_CONFIG"]["Tg_GREA"]
    X, y_raw, metadata, sample_metadata = namespace["load_task_data"]("Tg_GREA", config)
    y_model = pd.to_numeric(y_raw, errors="coerce")
    valid = y_model.notna()
    X, y_model = X.loc[valid], y_model.loc[valid]
    y_display = y_model.copy()
    specs = {item.name: item for item in namespace["get_model_specs"]("regression", None)}
    outer_cv = namespace["make_cv"]("regression", y_model, OUTER_CV_SPLITS, RANDOM_STATE)
    prediction = np.full(len(X), np.nan, dtype=float)
    fold_id = np.full(len(X), -1, dtype=int)
    for fold, (train_index, valid_index) in enumerate(outer_cv.split(X, y_model), start=1):
        estimator = namespace["build_pipeline"](specs[best_model])
        estimator.set_params(**fold_params[fold])
        estimator.fit(X.iloc[train_index], y_model.iloc[train_index])
        prediction[valid_index] = estimator.predict(X.iloc[valid_index])
        fold_id[valid_index] = fold
    if (fold_id < 1).any() or np.isnan(prediction).any():
        raise RuntimeError("Incomplete GREA best-model OOF reconstruction")
    processed = pd.read_csv(PROCESSED_DATA)
    if len(processed) != len(X):
        raise RuntimeError("GREA processed-data and model-input rows are not aligned")
    oof = processed[["source_row_index", "source_record_id", "sample_id", "smiles1", "Tg"]].copy()
    oof["outer_fold"] = fold_id
    oof["best_model"] = best_model
    oof["predicted_value"] = prediction
    oof.to_csv(BEST_DIR / "Tg_GREA_actual_best_oof.csv", index=False)
    full_estimator = namespace["build_pipeline"](specs[best_model])
    full_estimator.set_params(**params["best_params"])
    full_estimator.fit(X, y_model)
    model_payload = {
        "task": "Tg_GREA",
        "best_model": best_model,
        "estimator": full_estimator,
        "feature_names": list(X.columns),
        "params": params["best_params"],
        "input_path": str(final_input),
        "input_sha256": hashlib.sha256(final_input.read_bytes()).hexdigest(),
        "random_state": RANDOM_STATE,
    }
    joblib.dump(model_payload, BEST_DIR / "Tg_GREA_actual_best_full_fit.joblib")
    result = {
        "task": "Tg_GREA",
        "best_model": best_model,
        "selection_metric": "R2_mean",
        "selection_metric_mean": float(usable.loc[usable["model"].eq(best_model), "R2_mean"].iloc[0]),
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "oof_file": str((BEST_DIR / "Tg_GREA_actual_best_oof.csv").relative_to(EXPERIMENT_ROOT)),
        "full_fit_file": str((BEST_DIR / "Tg_GREA_actual_best_full_fit.joblib").relative_to(EXPERIMENT_ROOT)),
    }
    (BEST_DIR / "actual_best_model.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def validate_oof_frame(oof: pd.DataFrame, expected_samples: int) -> dict[str, Any]:
    """Validate the complete one-prediction-per-sample OOF contract."""
    required = {"source_row_index", "outer_fold", "predicted_value"}
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(f"OOF output is missing required columns: {sorted(missing)}")
    if len(oof) != expected_samples:
        raise ValueError(f"OOF row count {len(oof)} does not match expected samples {expected_samples}")
    if oof["source_row_index"].nunique() != expected_samples:
        raise ValueError("OOF output does not contain exactly one row per source sample")
    folds = sorted(pd.to_numeric(oof["outer_fold"], errors="coerce").dropna().astype(int).unique().tolist())
    if folds != list(range(1, OUTER_CV_SPLITS + 1)):
        raise ValueError(f"OOF folds must be 1..{OUTER_CV_SPLITS}, got {folds}")
    if oof["predicted_value"].isna().any():
        raise ValueError("OOF output contains missing predictions")
    return {"oof_rows": len(oof), "oof_unique_samples": int(oof["source_row_index"].nunique()), "oof_folds": folds, "oof_complete": True}


def validate_artifacts() -> dict[str, Any]:
    rows = _read_grea_rows(SOURCE_DATASET)
    result: dict[str, Any] = {"n_source_rows": len(rows), "n_unique_smiles": len({smiles for smiles, _ in rows})}
    if PROCESSED_DATA.exists():
        processed = pd.read_csv(PROCESSED_DATA)
        matrix = pd.read_csv(FEATURE_MATRIX)
        result.update({
            "n_processed_rows": len(processed),
            "all_homopolymers": bool(processed["type"].eq("homopolymer").all()),
            "all_composition_valid": bool(processed["composition_valid"].eq(True).all()),
            "feature_shape": list(matrix.shape),
            "features_finite": bool(np.isfinite(matrix.to_numpy(dtype=float)).all()),
            "rows_aligned": len(processed) == len(matrix),
        })
    best_oof = BEST_DIR / "Tg_GREA_actual_best_oof.csv"
    if best_oof.exists():
        result.update(validate_oof_frame(pd.read_csv(best_oof), len(rows)))
    all_models_oof = MODEL_DIR / "cv_predictions_with_source_ids.csv"
    if all_models_oof.exists():
        all_models_audit = augment_model_oof()
        result["all_models_oof"] = all_models_audit
    (RESULTS_DIR / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


@contextmanager
def experiment_lock():
    ensure_dirs()
    lock_path = LOG_DIR / ".pipeline.lock"
    with lock_path.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another Tg_GREA pipeline stage is already running") from exc
        yield


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "vocab", "features", "select", "model_compare", "best_model", "validate", "all"])
    parser.add_argument("--force-features", action="store_true", help="Recompute the experiment-local feature cache.")
    args = parser.parse_args()
    ensure_dirs()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.FileHandler(LOG_DIR / f"{args.stage}.log", encoding="utf-8"), logging.StreamHandler()])
    stages = [args.stage] if args.stage != "all" else ["prepare", "vocab", "features", "select", "model_compare", "best_model", "validate"]
    with experiment_lock():
        for stage in stages:
            logging.info("Starting Tg_GREA stage: %s", stage)
            if stage == "prepare":
                prepare_input()
                write_experiment_manifest()
            elif stage == "vocab":
                build_vocab()
            elif stage == "features":
                build_features(force=args.force_features)
            elif stage == "select":
                run_feature_selection()
            elif stage == "model_compare":
                run_model_comparison()
            elif stage == "best_model":
                build_best_model()
            elif stage == "validate":
                logging.info("Validation: %s", validate_artifacts())
            logging.info("Completed Tg_GREA stage: %s", stage)


if __name__ == "__main__":
    main()
