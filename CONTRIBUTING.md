# Contributing

Keep repository cleanup separate from scientific changes.

- Do not change datasets, thresholds, seeds, units, filtering rules, target definitions, or numerical protocols under a formatting/refactor commit.
- Run `bash workflows/run_smoke_test.sh` before proposing changes.
- Treat DOI-only edits as source-metadata revisions and validate them with `scripts/validate_dataset_metadata.py`.
- Record behavior-changing scientific work explicitly, with justification and regenerated evidence.
- Do not commit credentials, machine-local paths, restricted data, temporary caches, or unreviewed large artifacts.
- Do not use `git add .`; stage reviewable groups by purpose.

The project is not yet open for general redistribution or contribution until its license and data-rights decisions are complete.

