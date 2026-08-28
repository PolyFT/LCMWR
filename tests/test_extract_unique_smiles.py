from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_unique_smiles import collect_unique_smiles, write_outputs  # noqa: E402


class ExtractUniqueSmilesTests(unittest.TestCase):
    def test_collect_and_write_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            table = base / "LOI.csv"
            table.write_text(
                "smiles1,smiles2,mix_smiles1,LOI\n"
                "CC,CO,,20\n"
                "CC,,O,21\n",
                encoding="utf-8",
            )
            sources, report = collect_unique_smiles(base, ("LOI.csv",))
            self.assertEqual(sorted(sources), ["CC", "CO", "O"])
            self.assertEqual(len(report), 1)

            output_csv = base / "out" / "smiles.csv"
            output_txt = base / "out" / "smiles.txt"
            write_outputs(sources, output_csv, output_txt)
            with output_csv.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["smiles"] for row in rows], ["CC", "CO", "O"])
            self.assertEqual(output_txt.read_text().splitlines(), ["CC", "CO", "O"])


if __name__ == "__main__":
    unittest.main()

