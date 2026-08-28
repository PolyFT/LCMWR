# Data policy and provenance

## Public and complete analysis scopes

The four public source tables contain 1,765 records. The reported analysis uses 2,545 records; the remaining 780 target-bearing records are restricted because they include unpublished laboratory measurements or collaboration-governed data.

| Public file | Property | Public rows | Complete analysis rows | Restricted rows | Release role |
|---|---|---:|---:|---:|---|
| `dataset/LOI.csv` | Limiting oxygen index | 738 | 948 | 210 | Public source subset |
| `dataset/T5.csv` | 5% mass-loss temperature | 411 | 584 | 173 | Public source subset |
| `dataset/Tg.csv` | Glass-transition temperature | 313 | 518 | 205 | Public source subset |
| `dataset/UL-94.csv` | UL-94 class | 303 | 495 | 192 | Public source subset |
| `dataset/Tg_GREA.csv` | GREA GlassTemp input | 7,174 | 7,174 | Targets withheld | Structure-only benchmark input |

The public checkout therefore supports source-table inspection, deterministic unique-SMILES extraction, smoke tests and the IA vectorization check. It does not contain every target-bearing row required to retrain the reported four-task models.

Complete selected feature matrices are retained as structure-derived evidence with row counts 948, 584, 518 and 495. They do not replace the restricted source targets needed for end-to-end model recreation.

## Source provenance

The first `DOI` column records literature provenance. DOI additions or corrections are metadata changes; row order and all non-DOI scientific fields must remain invariant unless a separate scientific-data revision is declared.

Compare a frozen reference table with a DOI-enriched candidate using:

```bash
python scripts/validate_dataset_metadata.py reference.csv candidate.csv
```

The command accepts numeric display normalization such as `100.0` to `100`, but fails if scientific values, text fields, schema, row count or row order change. Use `--json-output` to retain a machine-readable report.

## Derived data

- `data/unique_smiles_for_fragments.*`: deterministic extraction from the four public source tables.
- `data/task_unique_smiles/`: task-specific unique structures used by motif generation.
- `results/local_vocab_*.csv`: generated motif vocabularies and support data.
- `results/*_motif_select/`: processed tables, feature matrices, selection tables and figure evidence.
- `results/model_compare/`: aggregate nested-CV summaries, parameters and pooled OOF metrics.
- `results/interpretability/`: descriptive full-data SHAP rankings and publication figures.
- `experiments/tg_grea/results/`: selected aggregate GREA summaries and figures.

Runtime caches, logs, fitted models, raw SHAP arrays, sample-level OOF predictions and target-bearing GREA derivatives remain local unless explicitly listed as published evidence in `results/README.md`.

## Rights and redistribution

Inclusion in this repository does not itself grant an open-data license. The DOI field documents provenance but does not replace the access terms of the underlying sources. The following remain excluded:

- 780 restricted primary-property records;
- GREA Tg target values;
- target-bearing GREA intermediate tables;
- sample-level GREA OOF predictions and fitted models;
- collaborator- or institution-restricted material.

No blanket software or data reuse license is currently granted. Users must independently respect source, institutional and confidentiality restrictions.
