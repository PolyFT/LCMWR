# Workflow

## Stage 0: validate the checkout

```bash
bash workflows/run_smoke_test.sh
```

Run the production preflight without writing outputs:

```bash
python workflows/run_retraining_pipeline.py --preflight
```

## Stage 1: extract source SMILES

```bash
python scripts/extract_unique_smiles.py \
  --per-table-output-dir data/task_unique_smiles
```

Inputs: `dataset/*.csv`  
Outputs: `data/unique_smiles_for_fragments.*` and task-specific extracts.

## Stage 2: build the motif vocabulary

The current source of truth is `scripts/motif_generating.ipynb`. The repository-local cell executor runs it without `nbconvert` and records the stage log and output checksums.

## Stage 3: generate and select task features

```bash
PYTHONPATH=scripts/feature_selection \
python scripts/feature_selection/run_motif_selection_notebook.py loi
```

Replace `loi` with `tg`, `t5`, or `ul94`. Use `--stop-after-cell 0` to build and audit only the feature matrix. A subsequent full invocation reloads that validated cache and runs the 750-combination search.

## Stage 4: compare models

```bash
PYTHONPATH=scripts/feature_selection \
python scripts/feature_selection/run_motif_selection_notebook.py loi_model_compare
```

Available tasks are `loi_model_compare`, `tg_model_compare`, `t5_model_compare`, and `ul94_model_compare`.

## Stage 5: Figure 3

```bash
PYTHONPATH=scripts/feature_selection \
python scripts/feature_selection/run_figure3_analysis_notebook.py loi
```

The source notebook remains unchanged. The executed copy is local-only under `results/executed_notebooks/figure3/`. Use `--in-place` only for the legacy behavior.

## Stage 6: Figure 4 and SHAP

The model-comparison and interpretation path is:

- `scripts/model_comparison/rebuild_actual_best_oof.py` determines each actual best model, reconstructs pooled OOF predictions/probabilities, and refits it on all data;
- `scripts/model_comparison/figure4_actual_best.py` generates Figure 4c–f from those OOF artifacts: Tg is 4c, T5 is 4d, LOI is 4e, and UL-94 is 4f;
- `scripts/shap_analysis/run_actual_best_shap.py` dispatches an appropriate explainer for the selected model and generates Figure 5/6 assets. UL-94 always explains `P(V-0)`.

Figure 4a–b are externally assembled and have no repository renderer. Publication plotting requires regular and bold Arial files. Set `LCMWR_ARIAL_REGULAR` and `LCMWR_ARIAL_BOLD` when the WSL defaults are unavailable.

## Full guarded command

```bash
bash workflows/run_full.sh --confirm-production-run
```

This command runs through release validation. Each expensive stage must pass its artifact gate before the next stage starts.

The workflow uses the task tables currently present under `dataset/` and writes to the established `results/` locations. The public tables are smaller than the complete analysis inputs, so a public-table run does not recreate the reported complete-data metrics.
