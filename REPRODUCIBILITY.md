# Reproducibility

## Evidence levels

Use these terms precisely:

- **Documented**: the command, inputs and expected outputs are recorded.
- **Runnable**: the command has executed in the existing project environment.
- **Reproduced**: a clean environment recreated the declared reference result from declared inputs.

The repository is documented and locally smoke-tested. Paper-facing aggregate artifacts pass the alignment gate, but the complete production workflow has not been recreated in a clean environment and cannot be run publicly without restricted target-bearing inputs.

## Maintained validation

```bash
bash workflows/run_smoke_test.sh
```

The smoke command writes caches to a temporary directory and checks:

- Python source and notebook-cell syntax;
- deterministic SMILES extraction and dataset metadata behavior;
- motif-query and hierarchical-composition invariants;
- public and complete analysis row counts;
- the IA Supporting Information vectorization example;
- screening counts, thresholds, best models, OOF metrics and SHAP rankings;
- GREA counts, nested-CV metrics and pooled OOF metrics;
- the complete paper/Supporting Information mapping.

The paper-specific gate can also be run independently:

```bash
python scripts/validate_paper_alignment.py
```

## Recorded environment

The reported workflow used a Conda environment with Python 3.10.13.

| Component | Version |
|---|---:|
| Python | 3.10.13 |
| NumPy | 1.23.5 |
| pandas | 1.5.3 |
| SciPy | 1.9.3 |
| scikit-learn | 1.2.2 |
| RDKit | 2022.09.5 |
| XGBoost | 1.6.2 |
| LightGBM | 4.6.0 |
| Matplotlib | 3.5.3 |
| seaborn | 0.12.2 |
| SHAP | 0.42.1 |
| UMAP-learn | 0.5.7 |

`environment.yml` provides the supported reconstruction environment; it is not an exact cross-platform lock. Operating system, CPU, memory, optional GPU/CUDA backend and measured stage runtimes must be recorded for any complete rerun.

## Scientific controls

The publication-alignment work does not change:

- random seed 48;
- five outer and three inner CV folds;
- 20 randomized-search iterations;
- motif maximum atom count 17;
- CHON-only minimum support ratio 0.01;
- other-element minimum support ratio 0.005;
- the 6×5×5×5 feature-selection grid;
- UL-94 `V-0` as the positive class;
- ROC-AUC as the UL-94 selection metric;
- the hierarchical mole-internal/mass-blend composition rule.

The documented protocol is in `configs/scientific_protocol.json`. Runtime scientific behavior remains owned by the listed modules and notebooks until configuration centralization has been regression-tested.

## Data boundary

The reported four-task models use 2,545 rows; the public source tables contain 1,765 rows. The released complete structure-derived matrices support feature-list and row-count validation, but the 780 restricted target-bearing records are required to recreate screening scores, model fits, pooled OOF predictions and descriptive SHAP values.

`dataset/Tg_GREA.csv` contains 7,174 structures without Tg targets. Aggregate GREA results are published; target-bearing intermediates remain local.

The production workflow intentionally uses the tables found under `dataset/` without changing output namespaces. A run using the public tables is therefore a new public-subset run, not a recreation of the reported complete-data metrics.

## Production stages

Run the read-only preflight first:

```bash
python workflows/run_retraining_pipeline.py --preflight
```

The guarded full command is:

```bash
bash workflows/run_full.sh --confirm-production-run
```

To resume a verified range of stages:

```bash
python workflows/run_retraining_pipeline.py \
  --from-stage feature_matrices \
  --through-stage nested_model_comparison
```

The orchestrator records the input signature, environment, commands, state, logs and output checksums. It writes into `results/` and must not be confused with the temporary-output smoke test.

## Paper outputs

Every main and Supporting Information item is listed in `reproducibility/paper_reproduction_matrix.md`. Publication-facing numerical expectations are stored in `reproducibility/paper_claims.json`.

Figure 4c–f use pooled held-out outer-fold predictions or probabilities. Figure 5, Figure 6, Figure S5 and Figure S6 use descriptive SHAP values from models refitted on the complete task-specific datasets; they are explicitly not OOF-SHAP.

Publication SVG rendering requires Arial. Set `LCMWR_ARIAL_REGULAR` and `LCMWR_ARIAL_BOLD` when the WSL defaults are unavailable. Font substitution is not performed silently.

## Known nondeterminism and limits

- Parallel numerical libraries can introduce small floating-point differences across platforms.
- UMAP is seeded, but library and numeric-backend changes can affect coordinates.
- Complete 750-combination searches are too expensive for routine lightweight validation.
- Restricted targets prevent a fully public end-to-end rerun.
- The current evidence does not support labeling the checkout a cleanly reproduced release candidate.
