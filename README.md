# LCMWR

LCMWR constructs composition-weighted local chemical-motif representations for homopolymers, copolymers and polymer blends, then evaluates structure–property models for limiting oxygen index (LOI), 5% mass-loss temperature (T5), glass-transition temperature (Tg) and UL-94 classification.

## Associated paper

**A Component-Weighted Local Chemical Motif Representation for Unified Polymer Informatics**

Run-Fang Gao, Ya-Jie Yang, Ran Wang, Ya-Lin Song, Xiu-Li Wang, Teng Fu, and Yu-Zhong Wang

Canonical repository: <https://github.com/PolyFT/LCMWR>

## Workflow

The repository:

1. extracts unique component SMILES from four property tables;
2. generates and support-filters a shared local-motif vocabulary;
3. constructs hierarchical composition-weighted motif matrices;
4. selects features through a fixed 750-combination threshold grid;
5. compares 11 machine-learning models with nested 5×3 cross-validation;
6. produces screening, pooled OOF and descriptive SHAP evidence;
7. validates the method independently on the 7,174-structure GREA GlassTemp benchmark.

The declared paper-facing counts, thresholds, model choices, metrics and feature rankings are stored in `reproducibility/paper_claims.json`. Every main-text and Supporting Information item is mapped in [the paper reproduction matrix](reproducibility/paper_reproduction_matrix.md).

## Validated scope

The current checkout is **partially reproducible paper-supporting research software**:

- the public datasets, source syntax, scientific unit tests, IA vectorization example, tracked aggregate results and paper-facing invariants are locally validated;
- the complete target-bearing datasets contain restricted records and are not fully distributed;
- a clean-environment recreation of the complete production workflow has not been demonstrated;
- SHAP results describe models refitted on each complete task dataset and are not OOF-SHAP.

Run the maintained lightweight validation:

```bash
conda env create -f environment.yml
conda activate lcmwr
bash workflows/run_smoke_test.sh
```

Run only the paper-alignment gate:

```bash
python scripts/validate_paper_alignment.py
```

## Data scope

The reported four-task analysis uses 2,545 records. The public tables contain 1,765 records; 780 records are excluded from redistribution.

| Task | Public rows | Complete analysis rows |
|---|---:|---:|
| LOI | 738 | 948 |
| T5 | 411 | 584 |
| Tg | 313 | 518 |
| UL-94 | 303 | 495 |

The selected structure-derived feature matrices retain the complete analysis row counts without publishing the restricted target-bearing source rows. `dataset/Tg_GREA.csv` contains 7,174 unique structures only; its Tg targets, sample-level OOF predictions and fitted models remain local.

See [DATA.md](DATA.md) for file roles, provenance and redistribution limits.

## Full workflow

Check the production inputs and environment without writing files:

```bash
python workflows/run_retraining_pipeline.py --preflight
```

The guarded production command is:

```bash
bash workflows/run_full.sh --confirm-production-run
```

It is expensive and writes into `results/`. The command uses whichever task tables are present under `dataset/`; running it with the public tables does not recreate the complete-data paper metrics. Stage commands and recovery points are documented in [docs/workflow.md](docs/workflow.md).

Publication plotting requires regular and bold Arial files. The WSL defaults can be overridden with `LCMWR_ARIAL_REGULAR` and `LCMWR_ARIAL_BOLD`; see [docs/configuration.md](docs/configuration.md).

## Repository layout

- `dataset/`: public source tables
- `data/`: deterministic unique-structure extracts
- `scripts/`: motif, modeling, figure, SHAP and validation entry points
- `results/`: selected publication evidence and local generated artifacts
- `experiments/tg_grea/`: isolated GREA GlassTemp validation
- `reproducibility/`: claims, paper mapping, audits and run metadata
- `tests/`: scientific unit and repository-alignment tests
- `workflows/`: smoke, production and release wrappers

## Known limitations

- Restricted target-bearing records prevent public recreation of the complete four-task fits and pooled OOF predictions.
- GREA targets and target-bearing derivatives are not redistributed.
- Arial is required to reproduce the recorded publication typography.
- UMAP and parallel numerical libraries may show small platform-dependent differences.
- Full 750-combination searches and nested model comparisons are intentionally excluded from routine lightweight validation.
- Data and software are distributed without a blanket reuse license.

## License and reuse

No software or data license has been granted. Unless and until the authors and relevant institutions select explicit terms, the repository must not be treated as authorizing reuse, modification or redistribution. Third-party source rights and confidentiality restrictions continue to apply independently.
