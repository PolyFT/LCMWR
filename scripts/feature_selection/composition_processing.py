"""Strict hierarchical composition handling for LCMWR motif features.

The scientific rules in this module are summarized in
``configs/scientific_protocol.json``.  In particular,
``*_content`` columns are labels only, and a positive molar and weight amount
for the same component is a structural data error rather than two estimates to
compare.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors


COMPOSITION_RULE_VERSION = "hierarchical_mole_internal_mass_blend_v1"
MOLECULAR_WEIGHT_METHOD = "RDKit.Chem.rdMolDescriptors.CalcExactMolWt"
RDKit_VERSION = rdBase.rdkitVersion


def clean_smiles(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    return value.replace("[Fr]", "[H]").replace("[Rb]", "[H]")


def exact_molecular_weight(smiles):
    cleaned = clean_smiles(smiles)
    mol = Chem.MolFromSmiles(cleaned) if cleaned else None
    if mol is None:
        raise ValueError("molecular_weight_failure")
    value = float(rdMolDescriptors.CalcExactMolWt(mol))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("molecular_weight_failure")
    return value


@dataclass(frozen=True)
class ParsedAmount:
    state: str
    value: float | None


def _parse_amount(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ParsedAmount("missing", None)
    if isinstance(value, str) and not value.strip():
        return ParsedAmount("missing", None)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ParsedAmount("non_numeric", None)
    if not np.isfinite(number):
        return ParsedAmount("non_numeric", None)
    if number < 0:
        return ParsedAmount("negative", number)
    if number == 0:
        return ParsedAmount("zero", 0.0)
    return ParsedAmount("positive", number)


def _json(value):
    def plain(item):
        if isinstance(item, (np.integer, np.floating)):
            item = item.item()
        if item is None or (not isinstance(item, str) and pd.isna(item)):
            return None
        return item

    return json.dumps([plain(item) for item in value], ensure_ascii=False, separators=(",", ":"))


def _fraction_check(values):
    total = float(np.sum(np.asarray(values, dtype=np.float64)))
    # This bound only accounts for accumulated binary floating-point rounding;
    # it is not a configurable or scientific data-consistency tolerance.
    machine_bound = np.finfo(np.float64).eps * max(8, 4 * len(values))
    if abs(total - 1.0) > machine_bound:
        raise ArithmeticError(f"normalized_fraction_sum={total!r}")


def _invalid(reason, **details):
    return {
        **details,
        "valid": False,
        "vector": None,
        "review_reason": reason,
        "exclusion_reason": reason,
        "mutual_exclusivity_check": "failed" if "simultaneous_positive" in reason else "passed",
    }


def _component_amount(row, index, prefix, smiles):
    mol_field = f"{prefix}mol{index}" if prefix else f"mol{index}"
    wt_field = f"{prefix}wt{index}" if prefix else f"wt{index}"
    mol_amount = _parse_amount(row.get(mol_field))
    wt_amount = _parse_amount(row.get(wt_field))

    invalid = [
        f"{field}:{parsed.state}"
        for field, parsed in ((mol_field, mol_amount), (wt_field, wt_amount))
        if parsed.state in {"negative", "non_numeric"}
    ]
    if invalid:
        raise ValueError("invalid_amount:" + ",".join(invalid))
    if mol_amount.state == "positive" and wt_amount.state == "positive":
        raise ValueError(f"simultaneous_positive_mol_wt:{index}")

    molecular_weight = exact_molecular_weight(smiles)
    if mol_amount.state == "positive":
        moles = mol_amount.value
        mass = moles * molecular_weight
        source = mol_field
        conversion = "molar_amount_to_mass_by_exact_molecular_weight"
    elif wt_amount.state == "positive":
        mass = wt_amount.value
        moles = mass / molecular_weight
        source = wt_field
        conversion = "weight_amount_to_moles_by_exact_molecular_weight"
    elif mol_amount.state == "zero" or wt_amount.state == "zero":
        return {
            "active": False,
            "zero_ignored": True,
            "moles": 0.0,
            "mass": 0.0,
            "molecular_weight": molecular_weight,
            "source": mol_field if mol_amount.state == "zero" else wt_field,
            "conversion": "zero_amount_ignored",
        }
    else:
        return {
            "active": False,
            "zero_ignored": False,
            "moles": None,
            "mass": None,
            "molecular_weight": molecular_weight,
            "source": "missing",
            "conversion": "missing_amount",
        }
    return {
        "active": True,
        "zero_ignored": False,
        "moles": float(moles),
        "mass": float(mass),
        "molecular_weight": molecular_weight,
        "source": source,
        "conversion": conversion,
    }


def resolve_copolymer_composition(row, feature_lookup, *, require_explicit=False):
    """Convert base constituents to moles, normalize, and combine linearly."""
    components = []
    zero_components = []
    present_structures = []
    for index in (1, 2):
        smiles = clean_smiles(row.get(f"smiles{index}"))
        if not smiles:
            if _parse_amount(row.get(f"mol{index}")).state == "positive" or _parse_amount(row.get(f"wt{index}")).state == "positive":
                return _invalid(f"missing_base_structure:{index}")
            continue
        present_structures.append((index, smiles))
        if smiles not in feature_lookup:
            return _invalid(f"missing_feature_vector:smiles{index}")
        try:
            amount = _component_amount(row, index, "", smiles)
        except ValueError as exc:
            return _invalid(str(exc))
        item = {"index": index, "smiles": smiles, "feature": feature_lookup[smiles], **amount}
        if item["active"]:
            components.append(item)
        elif item["zero_ignored"]:
            zero_components.append(index)

    if not present_structures:
        return _invalid("missing_base_structure")
    missing = [item for item in present_structures if item[0] not in {x["index"] for x in components} and item[0] not in zero_components]
    if missing:
        if len(present_structures) == 1 and not require_explicit:
            index, smiles = missing[0]
            try:
                mw = exact_molecular_weight(smiles)
            except ValueError as exc:
                return _invalid(str(exc))
            components.append({
                "index": index,
                "smiles": smiles,
                "feature": feature_lookup[smiles],
                "active": True,
                "zero_ignored": False,
                "moles": 1.0,
                "mass": mw,
                "molecular_weight": mw,
                "source": "implicit_pure_polymer",
                "conversion": "single_structure_pure_polymer",
            })
        else:
            return _invalid("missing_necessary_base_composition")
    if not components:
        return _invalid("no_positive_base_component")

    total_moles = float(sum(item["moles"] for item in components))
    if not np.isfinite(total_moles) or total_moles <= 0:
        return _invalid("invalid_total_base_moles")
    fractions = [item["moles"] / total_moles for item in components]
    _fraction_check(fractions)
    vector = np.sum(
        [fraction * np.asarray(item["feature"], dtype=np.float64) for fraction, item in zip(fractions, components)],
        axis=0,
    )
    base_mass = float(sum(item["mass"] for item in components))
    mode = "homopolymer" if len(components) == 1 else "copolymer"
    return {
        "valid": True,
        "vector": vector,
        "base_mass": base_mass,
        "base_moles": total_moles,
        "composition_mode": mode,
        "review_reason": "",
        "exclusion_reason": "",
        "mutual_exclusivity_check": "passed",
        "co_n_components": len(components),
        "co_source_fields": _json([item["source"] for item in components]),
        "copolymer_amount_basis": _json([item["source"] for item in components]),
        "co_component_molecular_weights": _json([item["molecular_weight"] for item in components]),
        "copolymer_component_molecular_weights": _json([item["molecular_weight"] for item in components]),
        "co_component_moles": _json([item["moles"] for item in components]),
        "copolymer_component_moles": _json([item["moles"] for item in components]),
        "co_mole_fractions": _json(fractions),
        "copolymer_mole_fractions": _json(fractions),
        "co_component_masses": _json([item["mass"] for item in components]),
        "co_conversions": _json([item["conversion"] for item in components]),
        "copolymer_conversions": _json([item["conversion"] for item in components]),
        "co_zero_components_ignored": _json(zero_components),
    }


def combine_blend_features(components):
    """Harmonically combine feature vectors using supplied mass fractions."""
    if not components:
        raise ValueError("no_blend_components")
    weights = np.asarray([item[0] for item in components], dtype=np.float64)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("invalid_blend_mass_fraction")
    weights = weights / weights.sum()
    _fraction_check(weights)
    vectors = [np.asarray(item[1], dtype=np.float64) for item in components]
    if any(not np.isfinite(vector).all() for vector in vectors):
        raise ValueError("nonfinite_component_feature")
    if any((1.0 + vector <= 0).any() for vector in vectors):
        raise ValueError("invalid_harmonic_feature_domain")
    denominator = np.sum(
        [weight / (1.0 + vector) for weight, vector in zip(weights, vectors)],
        axis=0,
    )
    if not np.isfinite(denominator).all() or (denominator <= 0).any():
        raise ValueError("invalid_harmonic_denominator")
    output = 1.0 / denominator - 1.0
    if not np.isfinite(output).all():
        raise ValueError("nonfinite_combined_feature")
    return output


def resolve_blend_composition(row, base, feature_lookup):
    """Convert the base and mix constituents to masses and combine them."""
    if not base.get("valid"):
        return base
    mix_items = []
    zero_components = []
    has_mix_structure = False
    for index in range(1, 5):
        smiles = clean_smiles(row.get(f"mix_smiles{index}"))
        if not smiles:
            if _parse_amount(row.get(f"mix_mol{index}")).state == "positive" or _parse_amount(row.get(f"mix_wt{index}")).state == "positive":
                return _invalid(f"missing_mix_structure:{index}", **base)
            continue
        has_mix_structure = True
        if smiles not in feature_lookup:
            return _invalid(f"missing_feature_vector:mix_smiles{index}", **base)
        try:
            amount = _component_amount(row, index, "mix_", smiles)
        except ValueError as exc:
            return _invalid(str(exc), **base)
        if amount["active"]:
            mix_items.append({"index": index, "smiles": smiles, "feature": feature_lookup[smiles], **amount})
        elif amount["zero_ignored"]:
            zero_components.append(index)
        else:
            return _invalid(f"missing_necessary_mix_composition:{index}", **base)

    if not has_mix_structure or not mix_items:
        result = dict(base)
        result.update({
            "blend_source_fields": _json([]),
            "blend_amount_basis": _json([]),
            "blend_component_masses": _json([base["base_mass"]]),
            "blend_weight_fractions": _json([1.0]),
            "blend_component_moles": _json([base["base_moles"]]),
            "blend_mole_fractions": _json([1.0]),
            "blend_component_molecular_weights": _json([]),
            "blend_conversions": _json([]),
            "blend_zero_components_ignored": _json(zero_components),
        })
        return result

    masses = [base["base_mass"], *[item["mass"] for item in mix_items]]
    total_mass = float(sum(masses))
    if not np.isfinite(total_mass) or total_mass <= 0:
        return _invalid("invalid_total_blend_mass", **base)
    fractions = [mass / total_mass for mass in masses]
    _fraction_check(fractions)
    component_moles = [base["base_moles"], *[item["moles"] for item in mix_items]]
    total_component_moles = float(sum(component_moles))
    mole_fractions = [amount / total_component_moles for amount in component_moles]
    _fraction_check(mole_fractions)
    try:
        vector = combine_blend_features(
            [(fractions[0], base["vector"]), *[(fraction, item["feature"]) for fraction, item in zip(fractions[1:], mix_items)]]
        )
    except ValueError as exc:
        return _invalid(str(exc), **base)
    result = dict(base)
    result.update({
        "valid": True,
        "vector": vector,
        "composition_mode": "blend",
        "review_reason": "",
        "exclusion_reason": "",
        "mutual_exclusivity_check": "passed",
        "blend_source_fields": _json([item["source"] for item in mix_items]),
        "blend_amount_basis": _json(["base_mass", *[item["source"] for item in mix_items]]),
        "blend_component_masses": _json(masses),
        "blend_weight_fractions": _json(fractions),
        "blend_component_moles": _json(component_moles),
        "blend_mole_fractions": _json(mole_fractions),
        "blend_component_molecular_weights": _json([item["molecular_weight"] for item in mix_items]),
        "blend_conversions": _json([item["conversion"] for item in mix_items]),
        "blend_zero_components_ignored": _json(zero_components),
    })
    return result


def resolve_composition(row, feature_lookup):
    has_positive_mix = any(
        _parse_amount(row.get(f"mix_mol{i}")).state == "positive"
        or _parse_amount(row.get(f"mix_wt{i}")).state == "positive"
        for i in range(1, 5)
    )
    base = resolve_copolymer_composition(row, feature_lookup, require_explicit=has_positive_mix)
    result = resolve_blend_composition(row, base, feature_lookup)
    result["composition_rule_version"] = COMPOSITION_RULE_VERSION
    result["molecular_weight_method"] = MOLECULAR_WEIGHT_METHOD
    result["rdkit_version"] = RDKit_VERSION
    result["co_content_labels"] = _json([row.get("co_content1"), row.get("co_content2")],)
    result["mix_content_labels"] = _json([row.get(f"mix_content{i}") for i in range(1, 5)])
    return result
