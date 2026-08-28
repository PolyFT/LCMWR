# Generated results

Everything under `results/` is generated or derived from source tables and workflow code. The production composition rule is `hierarchical_mole_internal_mass_blend_v1`.

## Artifact classes

| Class | Examples | Default release treatment |
|---|---|---|
| Published evidence | selected feature tables, model summaries, parameters, metrics, and SVG figures | Retain in Git |
| Local-only data | raw feature/SHAP matrices, per-sample predictions, and target-bearing GREA derivatives | Ignore and retain locally |
| Runtime artifacts | binary models, caches, logs, locks, backups, and plotting caches | Ignore and retain locally |
| Historical results | archived metric protocols and prior comparisons | Ignore and retain locally |

The local release review records the publication decision for each result artifact. Update that review before adding new result families.

Local run records and caches are excluded by `.gitignore`.
