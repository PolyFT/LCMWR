from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "feature_selection"))

from composition_processing import (  # noqa: E402
    combine_blend_features,
    exact_molecular_weight,
    resolve_composition,
    resolve_copolymer_composition,
)


LOOKUP = {
    "CC": np.array([2.0, 0.0]),
    "O": np.array([0.0, 1.0]),
    "N": np.array([1.0, 3.0]),
    "CO": np.array([3.0, 2.0]),
}


def resolved(**values):
    return resolve_composition(pd.Series(values, dtype=object), LOOKUP)


class CompositionProcessingTests(unittest.TestCase):
    def assertFractions(self, encoded, expected):
        self.assertTrue(np.allclose(json.loads(encoded), expected, rtol=0, atol=1e-14))

    def test_01_two_copolymer_molar_amounts(self):
        result = resolved(smiles1="CC", mol1=3, smiles2="O", mol2=1)
        self.assertTrue(result["valid"])
        self.assertFractions(result["co_mole_fractions"], [0.75, 0.25])
        np.testing.assert_allclose(result["vector"], [1.5, 0.25])

    def test_02_two_copolymer_weight_amounts_convert_to_moles(self):
        result = resolved(smiles1="CC", wt1=30, smiles2="O", wt2=18)
        expected = np.array([30 / exact_molecular_weight("CC"), 18 / exact_molecular_weight("O")])
        expected /= expected.sum()
        self.assertFractions(result["co_mole_fractions"], expected)
        np.testing.assert_allclose(result["vector"], expected[0] * LOOKUP["CC"] + expected[1] * LOOKUP["O"])

    def test_03_different_copolymer_components_can_use_different_bases(self):
        result = resolved(smiles1="CC", mol1=2, smiles2="O", wt2=18)
        expected = np.array([2, 18 / exact_molecular_weight("O")]); expected /= expected.sum()
        self.assertTrue(result["valid"])
        self.assertFractions(result["co_mole_fractions"], expected)

    def test_04_single_base_plus_mix_weight(self):
        result = resolved(smiles1="CC", wt1=80, mix_smiles1="O", mix_wt1=20)
        self.assertFractions(result["blend_weight_fractions"], [0.8, 0.2])
        np.testing.assert_allclose(result["vector"], combine_blend_features([(0.8, LOOKUP["CC"]), (0.2, LOOKUP["O"])]))

    def test_05_copolymer_plus_mix_molar_amount(self):
        result = resolved(smiles1="CC", mol1=3, smiles2="O", mol2=1, mix_smiles1="N", mix_mol1=2)
        self.assertTrue(result["valid"])
        masses = np.array([3 * exact_molecular_weight("CC") + exact_molecular_weight("O"), 2 * exact_molecular_weight("N")]); masses /= masses.sum()
        self.assertFractions(result["blend_weight_fractions"], masses)

    def test_06_different_mix_components_can_use_different_bases(self):
        result = resolved(smiles1="CC", wt1=50, mix_smiles1="O", mix_wt1=20, mix_smiles2="N", mix_mol2=1)
        masses = np.array([50, 20, exact_molecular_weight("N")]); masses /= masses.sum()
        self.assertTrue(result["valid"])
        self.assertFractions(result["blend_weight_fractions"], masses)

    def test_07_same_mix_component_mol_and_wt_is_abnormal(self):
        result = resolved(smiles1="CC", wt1=10, mix_smiles1="O", mix_wt1=2, mix_mol1=1)
        self.assertFalse(result["valid"])
        self.assertIn("simultaneous_positive_mol_wt", result["review_reason"])

    def test_08_multiple_mix_components(self):
        result = resolved(smiles1="CC", wt1=7, mix_smiles1="O", mix_wt1=2, mix_smiles2="N", mix_wt2=1)
        self.assertFractions(result["blend_weight_fractions"], [0.7, 0.2, 0.1])

    def test_09_relative_amounts_summing_to_one(self):
        result = resolved(smiles1="CC", wt1=.8, mix_smiles1="O", mix_wt1=.2)
        self.assertFractions(result["blend_weight_fractions"], [.8, .2])

    def test_10_relative_amounts_summing_to_100(self):
        result = resolved(smiles1="CC", wt1=80, mix_smiles1="O", mix_wt1=20)
        self.assertFractions(result["blend_weight_fractions"], [.8, .2])

    def test_11_arbitrary_positive_relative_amounts(self):
        result = resolved(smiles1="CC", wt1=7, mix_smiles1="O", mix_wt1=4)
        self.assertFractions(result["blend_weight_fractions"], [7 / 11, 4 / 11])

    def test_12_zero_mix_is_ignored_and_recorded(self):
        result = resolved(smiles1="CC", mix_smiles1="O", mix_wt1=0)
        self.assertTrue(result["valid"])
        self.assertEqual(result["composition_mode"], "homopolymer")
        self.assertEqual(json.loads(result["blend_zero_components_ignored"]), [1])

    def test_13_invalid_amount_structure_and_molecular_weight(self):
        for values, reason in [
            ({"smiles1": "CC", "mol1": -1}, "invalid_amount"),
            ({"smiles1": "CC", "mol1": "bad"}, "invalid_amount"),
            ({"mol1": 1}, "missing_base_structure"),
            ({"smiles1": "not-smiles", "mol1": 1}, "missing_feature_vector"),
        ]:
            result = resolved(**values)
            self.assertFalse(result["valid"])
            self.assertIn(reason, result["review_reason"])
        bad_lookup = {**LOOKUP, "not-smiles": np.zeros(2)}
        result = resolve_composition(pd.Series({"smiles1": "not-smiles", "mol1": 1}), bad_lookup)
        self.assertFalse(result["valid"])
        self.assertEqual(result["review_reason"], "molecular_weight_failure")

    def test_14_mix_content_label_is_not_numeric(self):
        result = resolved(smiles1="CC", wt1=9, mix_smiles1="O", mix_wt1=1, mix_content1="SK")
        self.assertFractions(result["blend_weight_fractions"], [.9, .1])
        self.assertEqual(json.loads(result["mix_content_labels"])[0], "SK")

    def test_15_molar_100_to_10_is_mass_converted(self):
        result = resolved(smiles1="CC", mol1=100, mix_smiles1="O", mix_mol1=10)
        masses = np.array([100 * exact_molecular_weight("CC"), 10 * exact_molecular_weight("O")]); masses /= masses.sum()
        self.assertFractions(result["co_mole_fractions"], [1.0])
        self.assertFractions(result["blend_mole_fractions"], [100 / 110, 10 / 110])
        self.assertFalse(np.allclose(masses, [100 / 110, 10 / 110]))
        self.assertFractions(result["blend_weight_fractions"], masses)

    def test_16_pc_sk_low_formulations(self):
        for base, mix in [(99.9, .1), (99.7, .3), (99.5, .5)]:
            result = resolved(smiles1="CC", wt1=base, mix_smiles1="O", mix_wt1=mix)
            self.assertFractions(result["blend_weight_fractions"], [base / 100, mix / 100])

    def test_17_same_base_component_mol_and_wt_is_abnormal(self):
        result = resolved(smiles1="CC", mol1=1, wt1=30)
        self.assertFalse(result["valid"])
        self.assertEqual(result["mutual_exclusivity_check"], "failed")

    def test_17b_pc_sk1_deleted_and_lower_formulations_retained(self):
        doi = "10.1021/acsapm.3c00444"
        for filename in ("LOI.csv", "T5.csv", "UL-94.csv"):
            data = pd.read_csv(ROOT / "dataset" / filename)
            source = data[data["DOI"].eq(doi)]
            self.assertEqual(int(source["PolymerName"].eq("PC/SK1").sum()), 0)
            for formulation in ("PC/SK0.1", "PC/SK0.3", "PC/SK0.5"):
                self.assertEqual(int(source["PolymerName"].eq(formulation).sum()), 1)

    def test_18_all_successful_fractions_sum_to_one(self):
        result = resolved(smiles1="CC", wt1=7, smiles2="O", mol2=2, mix_smiles1="N", mix_mol1=3)
        self.assertEqual(sum(json.loads(result["co_mole_fractions"])), 1.0)
        self.assertEqual(sum(json.loads(result["blend_weight_fractions"])), 1.0)

    def test_19_harmonic_combination_is_finite(self):
        output = combine_blend_features([(0.3, LOOKUP["CC"]), (0.7, LOOKUP["O"])])
        self.assertTrue(np.isfinite(output).all())
        self.assertTrue((output >= 0).all())

    def test_multi_base_missing_amount_never_equal_falls_back(self):
        result = resolve_copolymer_composition(pd.Series({"smiles1": "CC", "smiles2": "O"}), LOOKUP)
        self.assertFalse(result["valid"])
        self.assertEqual(result["review_reason"], "missing_necessary_base_composition")


if __name__ == "__main__":
    unittest.main()
