# External software and platform requirements

## Required

- Conda-compatible environment manager for `environment.yml`.
- Python 3.10 and the scientific packages listed in the environment file.
- Python notebook dependencies recorded in `environment.yml`; the repository-local executor runs production notebook stages.

## Optional orchestration

`scripts/feature_selection/run_remaining_motif_selection.sh` can wait on a named tmux session. tmux is not required by the scientific algorithms and is not used by the smoke test.

## Publication fonts

Several current Figure 4 and SHAP notebooks load:

- regular Arial font file
- bold Arial font file

The WSL defaults are `/mnt/c/Windows/Fonts/arial.ttf` and `/mnt/c/Windows/Fonts/arialbd.ttf`. Override them with `LCMWR_ARIAL_REGULAR` and `LCMWR_ARIAL_BOLD`. Missing files cause an actionable error; the maintained publication entry points do not silently substitute another font.

Do not substitute a font for publication figures without recording and visually validating the change.

## GPU and CUDA

The recorded environment included a CUDA-enabled LightGBM build, while XGBoost was CPU-built. The code does not establish a general GPU requirement. The final release must state the exact backend used for the reported results and must not imply that `environment.yml` captures driver/CUDA compatibility.
