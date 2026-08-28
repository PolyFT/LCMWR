from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "feature_selection"))

from dataset_provenance import compare_datasets, dataset_fingerprints  # noqa: E402


FIXTURES = ROOT / "tests" / "fixtures"


class DatasetProvenanceTests(unittest.TestCase):
    def test_doi_and_numeric_display_changes_preserve_scientific_hash(self):
        reference = dataset_fingerprints(FIXTURES / "reference.csv")
        updated = dataset_fingerprints(FIXTURES / "doi_updated.csv")
        self.assertEqual(
            reference["scientific_data_hash"], updated["scientific_data_hash"]
        )
        self.assertNotEqual(
            reference["source_metadata_hash"], updated["source_metadata_hash"]
        )

    def test_metadata_only_comparison_passes(self):
        report = compare_datasets(
            FIXTURES / "reference.csv", FIXTURES / "doi_updated.csv"
        )
        self.assertTrue(report["valid_metadata_only_change"])
        self.assertEqual(report["scientific_difference_rows"], [])
        self.assertEqual(report["metadata_difference_rows"], [2, 3])

    def test_scientific_change_fails(self):
        report = compare_datasets(
            FIXTURES / "reference.csv", FIXTURES / "scientific_changed.csv"
        )
        self.assertFalse(report["valid_metadata_only_change"])
        self.assertEqual(report["scientific_difference_rows"], [2])

    def test_row_reordering_fails(self):
        report = compare_datasets(
            FIXTURES / "reference.csv", FIXTURES / "reordered.csv"
        )
        self.assertFalse(report["valid_metadata_only_change"])
        self.assertEqual(report["scientific_difference_rows"], [2, 3])


if __name__ == "__main__":
    unittest.main()
