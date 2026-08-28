from __future__ import annotations

import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"


class PublicDatasetShapeTests(unittest.TestCase):
    def read_rows(self, name: str) -> tuple[list[str], list[list[str]]]:
        with (DATASET / name).open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            return next(reader), list(reader)

    def test_primary_release_table_shapes(self):
        expected = {
            "LOI.csv": (738, "LOI"),
            "T5.csv": (411, "T5"),
            "Tg.csv": (313, "Tg"),
            "UL-94.csv": (303, "UL-94"),
        }
        for name, (row_count, target) in expected.items():
            with self.subTest(name=name):
                header, rows = self.read_rows(name)
                self.assertEqual(len(rows), row_count)
                self.assertIn(target, header)
                self.assertNotIn("P_content", header)
                self.assertNotIn("Dripping", header)

    def test_tg_has_no_exact_duplicate_records(self):
        _, rows = self.read_rows("Tg.csv")
        self.assertEqual(len(rows), len({tuple(row) for row in rows}))

    def test_grea_is_structure_only_and_complete(self):
        header, rows = self.read_rows("Tg_GREA.csv")
        self.assertEqual(header, ["smiles1"])
        self.assertEqual(len(rows), 7174)
        self.assertTrue(all(len(row) == 1 and row[0] for row in rows))
        self.assertEqual(len(rows), len({row[0] for row in rows}))


if __name__ == "__main__":
    unittest.main()
