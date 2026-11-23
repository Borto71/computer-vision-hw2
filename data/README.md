# Data Workspace

This directory houses all datasets and data-handling assets required by the project. It should be treated as the authoritative location for documenting provenance, transformation steps, and storage conventions.

## Scope
- Raw extracts sourced from external systems (preserve original files without modification).
- Intermediate artifacts generated during exploratory analysis or preprocessing.
- Curated datasets that are ready for model training or evaluation, accompanied by provenance notes.
- Automation scripts (for example, `data.py`) that coordinate ingestion, cleaning, validation, or schema checks.

## Recommended Layout
- `raw/` — immutable snapshots received directly from the source.
- `interim/` — temporary outputs created during exploratory or partial processing.
- `processed/` — versioned datasets that have completed the preparation workflow.
- `external/` — third-party resources such as benchmarks or pretrained embeddings.

Additional subdirectories may be introduced when necessary; select descriptive names that make intent clear to future contributors.

## Operating Guidelines
- Record each dataset update through an experiment log, commit message, or equivalent tracking mechanism.
- Exclude large binaries from version control; rely on DVC, Git LFS, or external object storage instead.
- Align preprocessing logic with the implementation in `data.py` so that notebooks and pipelines remain reproducible.
- Provide lightweight documentation (for example, a nested `README` or `schema.json`) when dataset structure becomes complex.

## Suggested Workflow
1. Extend `data.py` or companion scripts with any new acquisition or transformation steps.
2. Execute the relevant scripts to populate or refresh the directory layout.
3. Commit metadata, manifests, and code updates only; raw assets should stay outside of the repository.

Maintaining rigorous discipline in this directory is essential for producing dependable machine learning outcomes.
