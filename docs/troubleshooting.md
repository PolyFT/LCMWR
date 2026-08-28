# Troubleshooting

## `python: command not found` or imports fail

- Diagnosis: the LCMWR environment is not active.
- Confirm: `python --version` and `python -c "import rdkit, sklearn, xgboost"`.
- Action: create/activate `environment.yml`, or set `LCMWR_PYTHON` to the intended interpreter.

## UMAP/Numba or Matplotlib cannot write a cache

- Diagnosis: the default user cache directory is not writable.
- Confirm: the traceback mentions Numba caching or Matplotlib configuration.
- Action: set writable temporary directories, for example `NUMBA_CACHE_DIR=/tmp/lcmwr-numba` and `MPLCONFIGDIR=/tmp/lcmwr-mpl`.

## A cache rebuild occurs after a DOI update

- Diagnosis: the cache lacks the source-row metadata required by the current workflow.
- Confirm: the log reports a provenance upgrade.
- Action: allow one rebuild. Later DOI-only changes reuse scientific features and refresh DOI-bearing derived tables.

## Dataset metadata validation fails

- Diagnosis: schema, row count, row order, or at least one non-DOI field differs.
- Confirm: inspect `scientific_difference_rows` in the JSON report.
- Action: do not accept the update as DOI-only; review it as a scientific-data revision.

## Final plotting reports missing Arial files

- Diagnosis: the WSL Arial defaults are unavailable on the current machine.
- Confirm: check `/mnt/c/Windows/Fonts/arial.ttf` and `/mnt/c/Windows/Fonts/arialbd.ttf`, or inspect the configured overrides.
- Action: set `LCMWR_ARIAL_REGULAR` and `LCMWR_ARIAL_BOLD` to the regular and bold Arial files. Do not silently substitute a different font for publication output.

## Production preflight reports a missing input

- Diagnosis: an expected environment, configuration or dataset file is absent.
- Confirm: run `python workflows/run_retraining_pipeline.py --preflight` and inspect the named path.
- Action: restore the declared input before launching the guarded production workflow. The preflight performs no writes.

## The production workflow takes too long

- Diagnosis: feature selection evaluates the full 750-combination grid and nested model searches.
- Confirm: distinguish `workflows/run_full.sh` from `workflows/run_smoke_test.sh`.
- Action: use the smoke test for code validation. Do not present reduced settings as the production protocol.
