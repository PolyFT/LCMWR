from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "feature_selection"))

from motif_vocab_features import (  # noqa: E402
    _refresh_cached_source_metadata,
    run_composition_rule_self_test,
    run_fragment_query_self_test,
)


class MotifInvariantTests(unittest.TestCase):
    def test_fragment_query_invariants(self):
        run_fragment_query_self_test()

    def test_blend_weighting_invariants(self):
        run_composition_rule_self_test()

    def test_cached_doi_metadata_refresh_preserves_features(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_file = Path(temp_dir) / "LOI.csv"
            data_file.write_text(
                "DOI,smiles1,LOI\nnew-a,CC,20\nnew-b,CO,21\n",
                encoding="utf-8",
            )
            feature_matrix = pd.DataFrame({"C": [1.0, 2.0]})
            cached = {
                "processed_data": pd.DataFrame(
                    {"DOI": ["old-a", "old-b"], "smiles1": ["CC", "CO"], "LOI": [20, 21]}
                ),
                "feature_matrix": feature_matrix.copy(),
                "source_row_indices": [0, 1],
            }
            refreshed = _refresh_cached_source_metadata(
                cached,
                data_file,
                "LOI",
                {"data_hash": "raw", "source_metadata_hash": "source"},
                __import__("logging").getLogger("test"),
            )
            self.assertTrue(refreshed)
            self.assertEqual(cached["processed_data"]["DOI"].tolist(), ["new-a", "new-b"])
            pd.testing.assert_frame_equal(cached["feature_matrix"], feature_matrix)


if __name__ == "__main__":
    unittest.main()
