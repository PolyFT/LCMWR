# Configuration

## Runtime configuration

`configs/runtime.example.json` documents machine and resource settings. Copy it to `configs/runtime.local.json` for local notes; the local file is ignored by Git.

Supported environment overrides:

| Variable | Purpose | Default |
|---|---|---|
| `LCMWR_PYTHON` | Python executable used by wrappers | `python` from the active environment |
| `LCMWR_ARIAL_REGULAR` | Regular Arial file for publication figures | `/mnt/c/Windows/Fonts/arial.ttf` |
| `LCMWR_ARIAL_BOLD` | Bold Arial file for publication figures | `/mnt/c/Windows/Fonts/arialbd.ttf` |

## Scientific configuration

`configs/scientific_protocol.json` is a documented snapshot, not yet the runtime source of truth. This avoids silently changing the established notebooks during cleanup.

Scientific settings and machine settings must remain separate. A future centralization change must compare regenerated features, selected feature names, fold assignments, metrics, and expected figures before the notebooks may stop being authoritative.

## Precedence

Current entry-point precedence is:

1. explicit CLI options;
2. supported `LCMWR_*` environment variables;
3. repository-relative defaults;
4. documented notebook constants for scientific parameters.

Repository discovery in maintained entry points is based on script location or repository markers, not the checkout directory name.
