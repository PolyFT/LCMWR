from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WorkflowPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = load_module("retraining_pipeline", ROOT / "workflows/run_retraining_pipeline.py")
        cls.figure3 = load_module("figure3_motif_panels_portability", ROOT / "scripts/feature_selection/figure3_motif_panels.py")

    def test_preflight_is_repository_local_and_complete(self):
        result = self.workflow.preflight(self.workflow.commands())
        self.assertEqual(result["status"], "ready")
        self.assertEqual(Path(result["repository_root"]), ROOT)
        self.assertEqual(len(result["input_signature"]), 64)
        self.assertEqual(result["dataset_rows"]["LOI.csv"], 738)
        self.assertNotIn("CODEX_COMPOSITION_PROCESSING_AND_RETRAINING_SPEC", str(result))

    def test_figure3_root_does_not_depend_on_checkout_name(self):
        self.assertEqual(self.figure3.find_project_root(), ROOT)
        paths = self.figure3.task_paths("tg")
        self.assertEqual(paths["result_dir"], ROOT / "results/tg_motif_select")


if __name__ == "__main__":
    unittest.main()
