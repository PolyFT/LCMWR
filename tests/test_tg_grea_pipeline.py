from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from rdkit import Chem


RUNNER = Path(__file__).resolve().parents[1] / "experiments" / "tg_grea" / "scripts" / "run_pipeline.py"
SPEC = importlib.util.spec_from_file_location("tg_grea_pipeline", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TgGreaPreparationTests(unittest.TestCase):
    def test_prepare_adds_header_and_creates_homopolymer_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Tg_GREA.csv"
            source.write_text("*C*,-54\n*CC(*)C,-3\n", encoding="utf-8")
            prepared = root / "input" / "prepared.csv"
            unique = root / "data" / "unique.csv"
            manifest = MODULE.prepare_input(source, prepared, unique)
            self.assertEqual(manifest["n_samples"], 2)
            with source.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(next(csv.reader(handle)), ["smiles1", "Tg"])
            output = pd.read_csv(prepared)
            self.assertEqual(output["type"].tolist(), ["homopolymer", "homopolymer"])
            self.assertEqual(output["wt1"].tolist(), [100.0, 100.0])
            self.assertTrue(output["source_record_id"].is_unique)
            self.assertEqual(pd.read_csv(unique)["smiles"].tolist(), ["*C*", "*CC(*)C"])

    def test_prepare_rejects_duplicate_smiles(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "Tg_GREA.csv"
            source.write_text("*C*,-54\n*C*,-53\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                MODULE.prepare_input(source, Path(directory) / "prepared.csv", Path(directory) / "unique.csv")

    def test_oof_validation_requires_one_prediction_per_sample_and_all_folds(self):
        oof = pd.DataFrame({
            "source_row_index": list(range(5)),
            "outer_fold": [1, 2, 3, 4, 5],
            "predicted_value": [-20.0, 10.0, 30.0, 50.0, 70.0],
        })
        result = MODULE.validate_oof_frame(oof, 5)
        self.assertTrue(result["oof_complete"])
        invalid = oof.copy()
        invalid.loc[4, "source_row_index"] = 3
        with self.assertRaisesRegex(ValueError, "exactly one row"):
            MODULE.validate_oof_frame(invalid, 5)

    def test_all_model_oof_is_annotated_with_stable_source_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "model_compare" / "Tg_GREA"
            model_dir.mkdir(parents=True)
            processed_path = root / "processed.csv"
            models = [f"model_{index}" for index in range(11)]
            processed = pd.DataFrame({
                "source_row_index": list(range(5)),
                "source_record_id": [f"record-{index}" for index in range(5)],
                "sample_id": [f"GREA_{index}" for index in range(5)],
                "smiles1": ["*C*"] * 5,
                "Tg": [-10.0, 0.0, 10.0, 20.0, 30.0],
            })
            processed.to_csv(processed_path, index=False)
            predictions = pd.DataFrame({"sample_index": list(range(5)), "true_value": processed["Tg"]})
            for index, model in enumerate(models):
                predictions[model] = processed["Tg"] + index
                predictions[f"{model}_fold"] = list(range(1, 6))
            predictions.to_csv(model_dir / "cv_predictions.csv", index=False)
            pd.DataFrame({"model": models, "status": ["success"] * len(models)}).to_csv(
                model_dir / "model_performance_summary.csv", index=False
            )
            original_model_dir = MODULE.MODEL_DIR
            original_processed = MODULE.PROCESSED_DATA
            original_root = MODULE.EXPERIMENT_ROOT
            try:
                MODULE.MODEL_DIR = model_dir
                MODULE.PROCESSED_DATA = processed_path
                MODULE.EXPERIMENT_ROOT = root
                audit = MODULE.augment_model_oof()
            finally:
                MODULE.MODEL_DIR = original_model_dir
                MODULE.PROCESSED_DATA = original_processed
                MODULE.EXPERIMENT_ROOT = original_root
            annotated = pd.read_csv(model_dir / "cv_predictions_with_source_ids.csv")
            self.assertTrue(audit["all_models_complete"])
            self.assertEqual(annotated["source_record_id"].tolist(), processed["source_record_id"].tolist())
            self.assertEqual(annotated["model_10_fold"].tolist(), [1, 2, 3, 4, 5])

    def test_candidate_screen_keeps_every_exact_substructure_match(self):
        mols = [Chem.MolFromSmiles(value) for value in ("CCO", "CCN", "c1ccccc1", "CCCl")]
        bitsets = [0] * 2048
        for index, mol in enumerate(mols):
            for bit in Chem.PatternFingerprint(mol, fpSize=2048).GetOnBits():
                bitsets[bit] |= 1 << index
        full_mask = (1 << len(mols)) - 1
        for smiles in ("C", "CO", "CN", "c1ccccc1", "Cl"):
            query = Chem.MolFromSmiles(smiles)
            screened = MODULE._candidate_molecule_mask(query, bitsets, full_mask)
            exact = {index for index, mol in enumerate(mols) if mol.HasSubstructMatch(query)}
            kept = {index for index in range(len(mols)) if screened & (1 << index)}
            self.assertTrue(exact.issubset(kept), smiles)


if __name__ == "__main__":
    unittest.main()
