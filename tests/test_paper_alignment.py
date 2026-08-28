from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_paper_alignment.py"
SPEC = importlib.util.spec_from_file_location("paper_alignment", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PaperAlignmentTests(unittest.TestCase):
    def test_repository_artifacts_match_paper_claims(self):
        result = MODULE.validate_all()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["public_rows"], 1765)
        self.assertEqual(result["complete_rows"], 2545)
        self.assertEqual(result["mapped_items"], 28)

if __name__ == "__main__":
    unittest.main()
