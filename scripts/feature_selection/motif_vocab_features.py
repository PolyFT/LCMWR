"""Canonical SMILES-vocabulary feature generation for the four MOTIF tasks.

This module deliberately owns Cell 0 only. The feature-selection notebooks
retain their Cell 1 algorithms unchanged.
"""

import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem

from composition_processing import (
    COMPOSITION_RULE_VERSION,
    MOLECULAR_WEIGHT_METHOD,
    RDKit_VERSION,
    clean_smiles,
    combine_blend_features,
    resolve_composition,
)
from dataset_provenance import canonical_cell, dataset_fingerprints, read_csv_rows

MATCHING_VERSION = "fragment_smiles_query_v2"
BLEND_RULE_VERSION = COMPOSITION_RULE_VERSION
FEATURE_FILTER_VERSION = "drop_all_zero_v5_smiles_query"
CACHE_VERSION = "weighted_feature_matrix_v4_hierarchical_composition"
VOCAB_FORMAT = "smiles"
VOCAB_COLUMN_CANDIDATES = [
    "fragment_smiles",
    "canonical_smiles",
    "smiles",
    "fragment",
    "Vocab",
]
NUMERIC_ZERO_ATOL = 1e-6


def find_motif_root():
    candidates = [Path(__file__).resolve().parents[2], Path.cwd(), Path.cwd().parent]
    candidates += list(Path.cwd().parents)
    for candidate in candidates:
        if all((candidate / part).exists() for part in ("data", "results", "scripts")):
            return candidate.resolve()
    raise FileNotFoundError("Could not locate the motif project root.")


def file_sha1(path):
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


def _scientific_row_ids(path):
    header, rows = read_csv_rows(path)
    indices = [index for index, column in enumerate(header) if column != "DOI"]
    output = []
    for row in rows:
        payload = [canonical_cell(row[index]) for index in indices]
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        output.append(hashlib.sha256(encoded.encode()).hexdigest())
    return output


def _feature_output_paths(out_dir, task_key):
    prefix = f"{task_key}_vocab"
    return {
        "processed": out_dir / f"{prefix}_processed_data.csv",
        "matrix": out_dir / f"{prefix}_weighted_feature_matrix.csv",
        "raw": out_dir / f"{prefix}_weighted_raw_features.csv",
    }


def _write_feature_tables(paths, processed_data, feature_matrix, write_matrix=True):
    processed_data.to_csv(paths["processed"], index=False)
    if write_matrix:
        feature_matrix.to_csv(paths["matrix"], index=False)
    pd.concat(
        [processed_data.reset_index(drop=True), feature_matrix.reset_index(drop=True)],
        axis=1,
    ).to_csv(paths["raw"], index=False)


def _refresh_cached_source_metadata(cached, data_file, target, hashes, logger):
    """Refresh DOI-bearing cached tables without recomputing motif features."""
    processed_data = cached.get("processed_data")
    source_row_indices = cached.get("source_row_indices")
    if not isinstance(processed_data, pd.DataFrame) or source_row_indices is None:
        return False

    current = _read_csv(data_file).dropna(subset=[target]).copy()
    if "DOI" not in current or "DOI" not in processed_data:
        return False
    try:
        current_doi = current.loc[list(source_row_indices), "DOI"].tolist()
    except KeyError:
        return False
    if len(current_doi) != len(processed_data):
        return False

    refreshed = processed_data.copy()
    refreshed.loc[:, "DOI"] = current_doi
    cached["processed_data"] = refreshed
    cached.update(hashes)
    logger.info("Refreshed DOI/source metadata in the validated feature cache.")
    return True


def preprocess_smiles(smiles):
    return clean_smiles(smiles)


def compile_single_vocab(fragment_smiles):
    """Compile canonical fragment SMILES as an RDKit molecule query."""
    if not isinstance(fragment_smiles, str):
        return None
    fragment_smiles = fragment_smiles.strip()
    if not fragment_smiles:
        return None
    fragment_smiles = preprocess_smiles(fragment_smiles)
    try:
        return Chem.MolFromSmiles(fragment_smiles, sanitize=True)
    except Exception:
        return None


def run_fragment_query_self_test():
    benzene_target = Chem.MolFromSmiles("c1ccccc1")
    benzene_query = compile_single_vocab("C1=CC=CC=C1")
    assert benzene_target is not None and benzene_query is not None
    assert benzene_target.HasSubstructMatch(benzene_query), (
        "Kekule benzene fragment failed to match aromatic benzene."
    )

    cyclohexane_target = Chem.MolFromSmiles("C1CCCCC1")
    cyclohexane_query = compile_single_vocab("C1CCCCC1")
    assert cyclohexane_target is not None and cyclohexane_query is not None
    assert cyclohexane_target.HasSubstructMatch(cyclohexane_query)


def _read_csv(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="iso-8859-1")


def _prepare_vocab(vocab_file, logger):
    vocab_df = _read_csv(vocab_file)
    raw_rows = len(vocab_df)

    if "passed_support_filter" in vocab_df.columns:
        mask = (
            vocab_df["passed_support_filter"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        vocab_df = vocab_df.loc[mask].copy()

    filtered_rows = len(vocab_df)
    vocab_col = next(
        (c for c in VOCAB_COLUMN_CANDIDATES if c in vocab_df.columns),
        None,
    )
    if vocab_col is None:
        raise ValueError(
            "Canonical vocab has no structure column; expected one of "
            f"{VOCAB_COLUMN_CANDIDATES}"
        )

    logger.info("Using vocab structure column: %s", vocab_col)
    cleaned = [preprocess_smiles(x) for x in vocab_df[vocab_col].tolist()]
    nonempty = [x for x in cleaned if isinstance(x, str) and x]
    unique = list(dict.fromkeys(nonempty))

    compiled, valid, failed = [], [], []
    for fragment in unique:
        query = compile_single_vocab(fragment)
        if query is None:
            failed.append(fragment)
        else:
            valid.append(fragment)
            compiled.append(query)

    logger.info(
        "Vocab rows: raw=%d, after passed_support_filter=%d, "
        "after dropping empty=%d, unique=%d, compiled=%d, failed=%d",
        raw_rows,
        filtered_rows,
        len(nonempty),
        len(unique),
        len(valid),
        len(failed),
    )
    if unique and len(failed) / len(unique) > 0.01:
        logger.warning(
            "More than 1%% of canonical vocab failed SMILES compilation (%d/%d).",
            len(failed),
            len(unique),
        )

    stats = {
        "raw_vocab_count": raw_rows,
        "rows_after_passed_support_filter": filtered_rows,
        "rows_after_dropping_empty": len(nonempty),
        "unique_vocab_count": len(unique),
        "valid_vocab_count": len(valid),
        "compiled_vocab_count": len(valid),
        "failed_vocab_count": len(failed),
    }
    return valid, compiled, stats


def _count(smiles, queries):
    out = np.zeros(len(queries), dtype=np.float32)
    cleaned = preprocess_smiles(smiles)
    mol = Chem.MolFromSmiles(cleaned) if isinstance(cleaned, str) else None
    if mol is not None:
        for i, query in enumerate(queries):
            out[i] = len(mol.GetSubstructMatches(query, uniquify=True))
    return out


def _combine_strict(row, lookup, n):
    result = resolve_composition(row, lookup)
    vector = result.pop("vector", None)
    if vector is None:
        vector = np.zeros(n, dtype=np.float32)
    result["composition_valid"] = bool(result.pop("valid", False))
    return np.asarray(vector, dtype=np.float32), result


def run_composition_rule_self_test():
    lookup = {
        "CC": np.array([2.0, 0.0], dtype=np.float32),
        "O": np.array([0.0, 1.0], dtype=np.float32),
    }
    row = pd.Series({"smiles1": "CC", "mol1": 100.0, "mix_smiles1": "O", "mix_mol1": 10.0})
    vector, metadata = _combine_strict(row, lookup, 2)
    assert metadata["composition_valid"]
    base_mass = 100.0 * 30.046950192
    mix_mass = 10.0 * 18.010564684
    weights = np.array([base_mass, mix_mass]) / (base_mass + mix_mass)
    expected = combine_blend_features([(weights[0], lookup["CC"]), (weights[1], lookup["O"])])
    assert np.allclose(vector, expected)


# Compatibility names used by older notebooks now resolve to the strict rule;
# the inert legacy notebook text cannot reactivate the former inference path.
_combine = _combine_strict
run_blend_rule_self_test = run_composition_rule_self_test


def run_feature_cell(task, use_cache=True, force_recompute=False, n_jobs=-1):
    """Build task features, reusing a hash-validated cache by default."""
    task_key = task.lower()
    root = find_motif_root()
    logger = logging.getLogger(f"{task_key}_motif_feature_matrix")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    vocab_file = root / "results" / "local_vocab_parallel_threshold.csv"
    if not vocab_file.exists():
        raise FileNotFoundError(f"Canonical vocab file not found: {vocab_file}")

    data_names = {
        "loi": "LOI.csv",
        "t5": "T5.csv",
        "tg": "Tg.csv",
        "ul94": "UL-94.csv",
    }
    data_file = root / "dataset" / data_names[task_key]
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    out_dir = root / "results" / f"{task_key}_motif_select"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = _feature_output_paths(out_dir, task_key)
    target = {
        "loi": "LOI",
        "t5": "T5",
        "tg": "Tg",
        # The dataset header keeps the hyphen, while the task key does not.
        "ul94": "UL-94",
    }[task_key]

    # v9 invalidates every cache created before strict hierarchical composition.
    cache_file = out_dir / f"{task_key}_vocab_weighted_v9.pkl"
    stats_file = out_dir / f"{task_key}_vocab_feature_matrix_stats.json"
    data_fingerprints = dataset_fingerprints(data_file)
    hashes = {
        "vocab_hash": file_sha1(vocab_file),
        "data_hash": file_sha1(data_file),
        "scientific_data_hash": data_fingerprints["scientific_data_hash"],
        "source_metadata_hash": data_fingerprints["source_metadata_hash"],
    }
    scientific_cache_metadata = {
        "cache_version": CACHE_VERSION,
        "task": task_key,
        "target": target,
        "matching_version": MATCHING_VERSION,
        "blend_rule_version": BLEND_RULE_VERSION,
        "molecular_weight_method": MOLECULAR_WEIGHT_METHOD,
        "rdkit_version": RDKit_VERSION,
        "feature_filter_version": FEATURE_FILTER_VERSION,
        "vocab_hash": hashes["vocab_hash"],
        "scientific_data_hash": hashes["scientific_data_hash"],
    }
    cache_metadata = {**scientific_cache_metadata, **hashes}

    if use_cache and not force_recompute and cache_file.exists():
        try:
            cached = joblib.load(cache_file)
            if all(
                cached.get(key) == value
                for key, value in scientific_cache_metadata.items()
            ):
                provenance_changed = any(
                    cached.get(key) != hashes[key]
                    for key in ("data_hash", "source_metadata_hash")
                )
                if provenance_changed:
                    refreshed = _refresh_cached_source_metadata(
                        cached, data_file, target, hashes, logger
                    )
                    if not refreshed:
                        logger.info(
                            "Validated features lack refresh metadata; rebuilding once."
                        )
                    else:
                        _write_feature_tables(
                            output_paths,
                            cached["processed_data"],
                            cached["feature_matrix"],
                            write_matrix=False,
                        )
                        joblib.dump(cached, cache_file)
                else:
                    refreshed = True

                if not refreshed:
                    raise ValueError("cache requires a one-time provenance upgrade")
                logger.info("Using validated feature-matrix cache: %s", cache_file)
                cached_stats = {
                    **cache_metadata,
                    "cache_file": str(cache_file),
                    "feature_matrix_shape": cached.get("feature_matrix_shape"),
                    "feature_count": len(cached.get("feature_names", [])),
                    "sample_count": len(cached.get("target_variable", [])),
                    "loaded_from_validated_v9_cache": True,
                }
                stats_file.write_text(
                    json.dumps(cached_stats, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return cached
            logger.info("Ignoring stale feature-matrix cache: %s", cache_file)
        except Exception as exc:
            logger.warning("Feature-matrix cache could not be read; recomputing: %s", exc)
    elif force_recompute:
        logger.info("force_recompute=True; rebuilding feature matrix.")

    run_fragment_query_self_test()
    run_composition_rule_self_test()
    logger.info("Fragment-query and strict-composition self-tests passed.")

    valid_vocab, queries, vocab_stats = _prepare_vocab(vocab_file, logger)
    data = _read_csv(data_file)
    data = data.dropna(subset=[target]).copy()
    scientific_ids = _scientific_row_ids(data_file)
    data["source_record_id"] = [scientific_ids[int(index)] for index in data.index]
    data["source_row_index"] = data.index.astype(int)

    component_columns = ["smiles1", "smiles2"] + [
        f"mix_smiles{i}" for i in range(1, 5)
    ]
    for col in component_columns:
        if col in data:
            data[col] = data[col].map(preprocess_smiles)

    unique = sorted(
        {
            s
            for col in component_columns
            if col in data
            for s in data[col]
            if isinstance(s, str)
        }
    )
    lookup = dict(
        zip(
            unique,
            Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_count)(s, queries) for s in unique
            ),
        )
    )

    component_nonzero = np.array([np.count_nonzero(v) for v in lookup.values()])
    first_twenty_counts = {
        smi: int(np.count_nonzero(lookup[smi])) for smi in unique[:20]
    }
    logger.info(
        "First 20 component-SMILES vocab-match counts: %s",
        first_twenty_counts,
    )

    combined = [_combine_strict(row, lookup, len(valid_vocab)) for _, row in data.iterrows()]
    vectors = [item[0] for item in combined]
    metadata = [item[1] for item in combined]

    X = (
        pd.DataFrame(np.vstack(vectors), columns=valid_vocab, index=data.index)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    assert list(X.columns) == valid_vocab

    all_zero_mask = ~(X.abs() > NUMERIC_ZERO_ATOL).any(axis=0)
    logger.info(
        "Dropping vocab features that are zero across all samples: %d -> %d",
        X.shape[1],
        int((~all_zero_mask).sum()),
    )
    active_vocab = X.columns[~all_zero_mask].tolist()
    X = X.loc[:, ~all_zero_mask].copy()
    assert X.shape[1] == len(active_vocab)

    metadata_df = pd.DataFrame(metadata, index=data.index)
    for column in metadata_df.columns:
        data[column] = metadata_df[column]

    audit_dir = root / "results" / "data_quality"
    audit_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(
        audit_dir / f"{task_key}_composition_audit.csv", index=False
    )

    valid_rows = data["composition_valid"].eq(True)
    invalid_mode_counts = data.loc[~valid_rows, "composition_mode"].value_counts().to_dict()
    if invalid_mode_counts:
        logger.warning("Rows excluded by composition handling: %s", invalid_mode_counts)

    data = data.loc[valid_rows].copy()
    X = X.loc[valid_rows].copy()
    source_row_indices = [int(index) for index in data.index]

    prefix = f"{task_key}_vocab"
    mode_counts = data["composition_mode"].value_counts().to_dict()
    diagnostics = {
        "matching_version": MATCHING_VERSION,
        "blend_rule_version": BLEND_RULE_VERSION,
        "vocab_file": str(vocab_file),
        **hashes,
        "data_file": str(data_file),
        **vocab_stats,
        "unique_component_smiles_count": len(unique),
        "active_vocab_count": len(active_vocab),
        "all_zero_vocab_count": int(all_zero_mask.sum()),
        "active_vocab_ratio": len(active_vocab) / max(len(valid_vocab), 1),
        "feature_matrix_shape_before_zero_drop": [len(metadata), len(valid_vocab)],
        "feature_matrix_shape_after_zero_drop": list(X.shape),
        "component_nonzero_mean": float(component_nonzero.mean()) if len(component_nonzero) else 0.0,
        "component_nonzero_median": float(np.median(component_nonzero)) if len(component_nonzero) else 0.0,
        "component_nonzero_min": int(component_nonzero.min()) if len(component_nonzero) else 0,
        "component_nonzero_max": int(component_nonzero.max()) if len(component_nonzero) else 0,
        "first_twenty_component_match_counts": first_twenty_counts,
        "composition_mode_counts": mode_counts,
        "excluded_composition_mode_counts": invalid_mode_counts,
    }

    logger.info(
        "Active vocab: %d / %d (%.2f%%); per molecule nonzero "
        "mean/median/min/max: %.1f/%.1f/%d/%d",
        len(active_vocab),
        len(valid_vocab),
        100 * diagnostics["active_vocab_ratio"],
        diagnostics["component_nonzero_mean"],
        diagnostics["component_nonzero_median"],
        diagnostics["component_nonzero_min"],
        diagnostics["component_nonzero_max"],
    )

    test_queries = valid_vocab[:100]
    test_mols = [Chem.MolFromSmiles(s) for s in unique[:20]]
    test_mols = [mol for mol in test_mols if mol is not None]
    smiles_n = sum(
        bool(Chem.MolFromSmiles(fragment))
        and any(
            mol.HasSubstructMatch(Chem.MolFromSmiles(fragment))
            for mol in test_mols
        )
        for fragment in test_queries
    )
    smarts_n = sum(
        bool(Chem.MolFromSmarts(fragment))
        and any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(fragment))
            for mol in test_mols
        )
        for fragment in test_queries
    )
    logger.warning(
        "Diagnostic only: SMILES queries matched %d fragments, SMARTS queries "
        "matched %d fragments on the test molecules.",
        smiles_n,
        smarts_n,
    )

    _write_feature_tables(output_paths, data, X)
    (out_dir / f"{task_key}_vocab_matching_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output = {
        "processed_data": data,
        "feature_matrix": X,
        # Regression targets are numeric; UL-94 labels are stored as strings.
        "target_variable": (
            data[target].astype(str)
            if task_key == "ul94"
            else data[target].astype(float)
        ),
        "vocab_smiles": active_vocab,
        "vocab_fragments": active_vocab,
        "feature_names": active_vocab,
        "vocab_file": str(vocab_file),
        "data_file": str(data_file),
        "source_row_indices": source_row_indices,
        **cache_metadata,
        "valid_vocab": valid_vocab,
        "active_vocab": active_vocab,
        "feature_matrix_shape": list(X.shape),
    }
    stats_file.write_text(
        json.dumps(
            {
                **cache_metadata,
                "cache_file": str(cache_file),
                "feature_matrix_shape": list(X.shape),
                "feature_count": int(X.shape[1]),
                "sample_count": int(X.shape[0]),
                "active_vocab_count": len(active_vocab),
                "all_zero_vocab_count": int(all_zero_mask.sum()),
                "composition_mode_counts": mode_counts,
                "excluded_composition_mode_counts": invalid_mode_counts,
                "loaded_from_validated_v9_cache": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if use_cache:
        joblib.dump(output, cache_file)
        logger.info("Feature-matrix cache saved: %s", cache_file)
    return output
