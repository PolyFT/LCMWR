#!/usr/bin/env python3
"""Validate repository artifacts against the declared paper and SI claims."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "reproducibility" / "paper_claims.json"
MAPPING_PATH = ROOT / "reproducibility" / "paper_reproduction_matrix.md"
DISPLAY_DIR = {"LOI": "LOI", "T5": "T5", "Tg": "Tg", "UL94": "UL94"}


class AlignmentError(RuntimeError):
    """Raised when a repository artifact does not match a declared claim."""


def load_claims(path: Path = CLAIMS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise AlignmentError(f"Missing or empty paper artifact: {path.relative_to(ROOT)}")
    return path


def assert_close(actual: float, expected: float, label: str, tolerance: float = 1e-9) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise AlignmentError(f"{label}: expected {expected}, found {actual}")


def validate_dataset_scope(claims: dict) -> None:
    public_total = 0
    complete_total = 0
    for task, expected in claims["dataset_scope"]["tasks"].items():
        public = pd.read_csv(require_file(ROOT / expected["public_file"]))
        if len(public) != expected["public_rows"]:
            raise AlignmentError(f"{task}: expected {expected['public_rows']} public rows, found {len(public)}")
        slug = claims["tasks"][task]["slug"]
        matrix = pd.read_csv(require_file(ROOT / "results" / f"{slug}_motif_select" / "best_improved_final_features_matrix.csv"))
        if len(matrix) != expected["complete_rows"]:
            raise AlignmentError(f"{task}: expected {expected['complete_rows']} complete feature rows, found {len(matrix)}")
        if matrix.shape[1] != claims["tasks"][task]["selected_features"]:
            raise AlignmentError(f"{task}: complete matrix feature count changed")
        public_total += len(public)
        complete_total += len(matrix)
    if public_total != claims["dataset_scope"]["public_total"]:
        raise AlignmentError(f"Public total: expected {claims['dataset_scope']['public_total']}, found {public_total}")
    if complete_total != claims["dataset_scope"]["complete_total"]:
        raise AlignmentError(f"Complete total: expected {claims['dataset_scope']['complete_total']}, found {complete_total}")
    if complete_total - public_total != claims["dataset_scope"]["restricted_total"]:
        raise AlignmentError("Public/complete dataset difference does not match the declared restricted total")


def validate_motif_scope(claims: dict) -> None:
    components = pd.read_csv(require_file(ROOT / "data/unique_smiles_for_fragments.csv"))
    cleaned = components["smiles"].astype(str).str.replace("[Fr]", "[H]", regex=False).str.replace("[Rb]", "[H]", regex=False)
    if cleaned.nunique() != claims["motifs"]["cleaned_unique_components"]:
        raise AlignmentError("Cleaned unique-component count changed")
    vocabulary = pd.read_csv(require_file(ROOT / "results/local_vocab_parallel_threshold.csv"))
    if len(vocabulary) != claims["motifs"]["global_vocabulary_size"]:
        raise AlignmentError("Global motif-vocabulary size changed")
    stats = json.loads(require_file(ROOT / "results/local_vocab_parallel_threshold_stats.json").read_text(encoding="utf-8"))
    if stats["candidate_count_before_support_filter"] != claims["motifs"]["candidate_motifs"]:
        raise AlignmentError("Candidate motif count changed")


def validate_protocol(claims: dict) -> None:
    declared = claims["protocol"]
    protocol = json.loads(require_file(ROOT / "configs/scientific_protocol.json").read_text(encoding="utf-8"))
    nested = protocol["nested_cross_validation"]
    if protocol["random_state"] != declared["random_state"]:
        raise AlignmentError("Configured random seed differs from the paper claim")
    if nested["outer_splits"] != declared["nested_cross_validation"]["outer_splits"]:
        raise AlignmentError("Configured outer-fold count differs from the paper claim")
    if nested["inner_splits"] != declared["nested_cross_validation"]["inner_splits"]:
        raise AlignmentError("Configured inner-fold count differs from the paper claim")
    if nested["parameter_search_iterations"] != declared["parameter_search_iterations"]:
        raise AlignmentError("Configured parameter-search iterations differ from the paper claim")
    grid = protocol["feature_selection_grid"]
    combinations = math.prod(len(values) for values in grid.values())
    if combinations != declared["feature_selection_combinations"]:
        raise AlignmentError("Configured feature-selection grid does not contain 750 combinations")
    if protocol["ul94"]["positive_class"] != declared["ul94_positive_class"]:
        raise AlignmentError("Configured UL-94 positive class differs from the paper claim")


def validate_task_results(claims: dict) -> None:
    oof = pd.read_csv(require_file(ROOT / "results/model_compare/figure4_oof/figure4_oof_metrics_summary.csv")).set_index("task")
    for task, expected in claims["tasks"].items():
        slug = expected["slug"]
        selection_dir = ROOT / "results" / f"{slug}_motif_select"
        info = pd.read_csv(require_file(selection_dir / "best_improved_final_features_info.csv"))
        matrix_columns = pd.read_csv(require_file(selection_dir / "best_improved_final_features_matrix.csv"), nrows=1).columns.tolist()
        search = pd.read_csv(require_file(selection_dir / "plots_data/improved_750_complete_results.csv"))
        summary = pd.read_csv(require_file(selection_dir / f"figure3_analysis/figure3_{slug}_analysis_summary.csv")).iloc[0]
        if len(info) != expected["selected_features"]:
            raise AlignmentError(f"{task}: selected-feature count changed")
        if matrix_columns != info["feature_name"].tolist():
            raise AlignmentError(f"{task}: selected-feature order differs between the matrix and feature table")
        if len(search) != claims["protocol"]["feature_selection_combinations"]:
            raise AlignmentError(f"{task}: expected 750 feature-selection combinations")
        if int(summary["global_vocabulary_size"]) != claims["motifs"]["global_vocabulary_size"]:
            raise AlignmentError(f"{task}: global vocabulary size changed")
        if int(summary["active_feature_count"]) != expected["active_features"]:
            raise AlignmentError(f"{task}: active-feature count changed")
        if int(summary["selected_motif_count"]) != expected["selected_features"]:
            raise AlignmentError(f"{task}: figure summary selected-feature count changed")
        best = search.loc[search["performance"].idxmax()]
        threshold_columns = {
            "frequency": "freq_threshold",
            "variance": "var_threshold",
            "mi_percentile": "mi_threshold",
            "correlation": "corr_threshold",
        }
        for claim_key, column in threshold_columns.items():
            assert_close(best[column], expected["thresholds"][claim_key], f"{task} {claim_key}")
        if int(best["feature_count"]) != expected["selected_features"]:
            raise AlignmentError(f"{task}: best threshold row has the wrong feature count")

        model_summary = pd.read_csv(require_file(ROOT / "results/model_compare" / DISPLAY_DIR[task] / "model_performance_summary.csv"))
        if len(model_summary) != 11 or not model_summary["status"].eq("success").all():
            raise AlignmentError(f"{task}: model comparison must contain 11 successful models")
        primary_column = "ROC-AUC_mean" if task == "UL94" else "R2_mean"
        best_model = model_summary.sort_values(primary_column, ascending=False, kind="stable").iloc[0]
        if best_model["model"] != expected["best_model"]:
            raise AlignmentError(f"{task}: expected best model {expected['best_model']}, found {best_model['model']}")
        assert_close(best_model[primary_column], expected["nested_cv"]["mean"], f"{task} nested-CV mean")
        std_column = "ROC-AUC_std" if task == "UL94" else "R2_std"
        assert_close(best_model[std_column], expected["nested_cv"]["std"], f"{task} nested-CV standard deviation")
        if task == "UL94":
            assert_close(best_model["Accuracy_mean"], expected["nested_cv"]["accuracy_mean"], "UL94 accuracy")
            assert_close(oof.loc[task, "OOF_ROC_AUC"], expected["pooled_oof"]["roc_auc"], "UL94 pooled OOF ROC-AUC")
        else:
            assert_close(best_model["RMSE_mean"], expected["nested_cv"]["rmse_mean"], f"{task} nested-CV RMSE")
            assert_close(best_model["MAE_mean"], expected["nested_cv"]["mae_mean"], f"{task} nested-CV MAE")
            assert_close(oof.loc[task, "OOF_R2"], expected["pooled_oof"]["r2"], f"{task} pooled OOF R2")
            assert_close(oof.loc[task, "OOF_RMSE"], expected["pooled_oof"]["rmse"], f"{task} pooled OOF RMSE")

        ranking = pd.read_csv(require_file(ROOT / expected["shap_ranking_file"]))
        display_column = "display_feature" if "display_feature" in ranking else "feature"
        actual_features = ranking[display_column].head(4).tolist()
        if actual_features != expected["shap_top_display_features"]:
            raise AlignmentError(f"{task}: SHAP top features changed: {actual_features}")


def validate_si_example(claims: dict) -> None:
    expected = claims["si_example_ia"]
    data = pd.read_csv(require_file(ROOT / "dataset/LOI.csv"))
    rows = data.loc[data["PolymerName"].eq(expected["polymer_name"])]
    if len(rows) != 1:
        raise AlignmentError(f"Expected one IA row in the public LOI table, found {len(rows)}")
    row = rows.iloc[0]
    if row["smiles1"] != expected["component_1"] or row["smiles2"] != expected["component_2"]:
        raise AlignmentError("IA component structures do not match Table S1")
    mole_fractions = [float(row["mol1"]) / 100.0, float(row["mol2"]) / 100.0]
    if mole_fractions != expected["mole_fractions"]:
        raise AlignmentError(f"IA mole fractions changed: {mole_fractions}")
    assert_close(row["LOI"], expected["target"], "IA LOI")

    components = [Chem.MolFromSmiles(row["smiles1"]), Chem.MolFromSmiles(row["smiles2"])]
    if any(component is None for component in components):
        raise AlignmentError("IA contains an invalid component SMILES")
    expected_features = {item["motif"]: item for item in expected["nonzero_selected_features"]}
    selected = pd.read_csv(require_file(ROOT / "results/loi_motif_select/best_improved_final_features_info.csv"))["feature_name"].tolist()
    observed = {}
    for motif in selected:
        query = Chem.MolFromSmiles(motif)
        if query is None:
            raise AlignmentError(f"Invalid selected LOI motif: {motif}")
        counts = [len(component.GetSubstructMatches(query, uniquify=True)) for component in components]
        weighted = sum(fraction * count for fraction, count in zip(mole_fractions, counts))
        if weighted:
            observed[motif] = {"component_counts": counts, "weighted_value": weighted}
    if set(observed) != set(expected_features):
        raise AlignmentError(f"IA nonzero selected motifs changed: expected {len(expected_features)}, found {len(observed)}")
    for motif, actual in observed.items():
        item = expected_features[motif]
        if actual["component_counts"] != item["component_counts"]:
            raise AlignmentError(f"IA {motif}: component counts changed")
        assert_close(actual["weighted_value"], item["weighted_value"], f"IA {motif} weighted value")


def validate_figures_and_mapping(claims: dict) -> None:
    figure4_dir = ROOT / "results/model_compare/figure4_oof"
    actual_figure4 = sorted(path.name for path in figure4_dir.glob("figure4*.svg"))
    if actual_figure4 != sorted(claims["figure4_svg_files"]):
        raise AlignmentError(f"Figure 4 SVG set changed: {actual_figure4}")
    for name in actual_figure4:
        require_file(figure4_dir / name)

    for slug in ("loi", "t5", "tg", "ul94"):
        directory = ROOT / "results" / f"{slug}_motif_select" / "figure3_analysis"
        for panel in "bcdefgh":
            matches = list(directory.glob(f"Figure3{panel}_{slug}_*.svg"))
            if len(matches) != 1:
                raise AlignmentError(f"{slug}: expected one Figure 3{panel} SVG, found {len(matches)}")
            require_file(matches[0])

    for task in ("LOI", "T5", "Tg", "UL94"):
        directory = ROOT / "results/interpretability" / task
        require_file(directory / f"shap_summary_plot_{task}.svg")
        dependence = list(directory.glob(f"shap_dependence_*_{task}.svg"))
        if len(dependence) != 4:
            raise AlignmentError(f"{task}: expected four SHAP dependence SVGs, found {len(dependence)}")

    grea_figure3 = ROOT / "experiments/tg_grea/results/figures/figure3_analysis"
    for panel in "bcdefgh":
        matches = list(grea_figure3.glob(f"Figure3{panel}_tg_*.svg"))
        if len(matches) != 1:
            raise AlignmentError(f"GREA: expected one screening panel {panel}, found {len(matches)}")
        require_file(matches[0])
    require_file(ROOT / "experiments/tg_grea/results/figures/figure4_oof/Figure4_tg_grea_oof.svg")

    mapping = require_file(MAPPING_PATH).read_text(encoding="utf-8")
    for item in claims["paper_items"]:
        if f"| {item} |" not in mapping:
            raise AlignmentError(f"Paper reproduction matrix is missing {item}")


def validate_grea(claims: dict) -> None:
    expected = claims["grea"]
    source = pd.read_csv(require_file(ROOT / expected["public_file"]))
    if len(source) != expected["samples"] or list(source.columns) != ["smiles1"]:
        raise AlignmentError("The released GREA input must contain 7174 structure-only rows")
    validation = json.loads(require_file(ROOT / "experiments/tg_grea/results/validation.json").read_text(encoding="utf-8"))
    if validation["feature_shape"] != [expected["samples"], expected["active_features"]]:
        raise AlignmentError("GREA feature shape changed")
    if validation["oof_rows"] != expected["samples"] or not validation["oof_complete"]:
        raise AlignmentError("GREA OOF coverage is incomplete")
    selection = pd.read_csv(require_file(ROOT / "experiments/tg_grea/results/feature_selection/plots_data/improved_750_complete_results.csv"))
    if len(selection) != expected["feature_selection_combinations"]:
        raise AlignmentError("GREA feature-selection combination count changed")
    selected = pd.read_csv(require_file(ROOT / "experiments/tg_grea/results/feature_selection/best_improved_final_features_info.csv"))
    if len(selected) != expected["selected_features"]:
        raise AlignmentError("GREA selected-feature count changed")
    models = pd.read_csv(require_file(ROOT / "experiments/tg_grea/results/model_compare/Tg_GREA/model_performance_summary.csv"))
    best = models.sort_values("R2_mean", ascending=False, kind="stable").iloc[0]
    if best["model"] != expected["best_model"]:
        raise AlignmentError("GREA best model changed")
    for column, claim_key in (("R2_mean", "r2_mean"), ("R2_std", "r2_std"), ("RMSE_mean", "rmse_mean"), ("MAE_mean", "mae_mean")):
        assert_close(best[column], expected["nested_cv"][claim_key], f"GREA {column}")
    oof = pd.read_csv(require_file(ROOT / "experiments/tg_grea/results/figures/figure4_oof/figure4_tg_grea_oof_metrics.csv")).iloc[0]
    assert_close(oof["OOF_R2"], expected["pooled_oof"]["r2"], "GREA pooled OOF R2")
    assert_close(oof["OOF_RMSE"], expected["pooled_oof"]["rmse"], "GREA pooled OOF RMSE")


def validate_all(claims_path: Path = CLAIMS_PATH) -> dict:
    claims = load_claims(claims_path)
    validate_dataset_scope(claims)
    validate_motif_scope(claims)
    validate_protocol(claims)
    validate_task_results(claims)
    validate_si_example(claims)
    validate_figures_and_mapping(claims)
    validate_grea(claims)
    return {
        "paper_title": claims["paper"]["title"],
        "public_rows": claims["dataset_scope"]["public_total"],
        "complete_rows": claims["dataset_scope"]["complete_total"],
        "mapped_items": len(claims["paper_items"]),
        "status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, default=CLAIMS_PATH, help="Paper-claims JSON to validate.")
    args = parser.parse_args()
    result = validate_all(args.claims.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
