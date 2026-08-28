# Tg_GREA independent experiment

This directory contains an isolated Tg_GREA experiment. It does not write into
the repository-level `results/` directory. Selected aggregate performance summaries and figures are
published; target-bearing tables, OOF predictions, full-fit models, caches,
and logs remain local-only.

The released `dataset/Tg_GREA.csv` contains structures only. Supply the target
column before running model-building stages.

Run stages from the repository root:

```bash
python experiments/tg_grea/scripts/run_pipeline.py prepare
python experiments/tg_grea/scripts/run_pipeline.py all
```

Stages are `prepare`, `vocab`, `features`, `select`, `model_compare`,
`best_model`, `validate`, and `all`.  Long stages are resumable through the
experiment-local caches and checkpoint files, which are not published.

The experiment reproduces the current Tg settings: a GREA-only vocabulary,
the 750-combination feature-selection grid, and 11-model nested 5x3 model
comparison with random seed 48.
