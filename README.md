# Electronic Nose Thesis Workspace

This repository is a GitHub/Codex handoff package for the local e-nose thesis work.
It contains the stable training code, current thesis result tables, figures, reports,
and a documented boundary between data already used in training and newly recorded
data that has not yet entered any experiment.

## Current Scope

- Task: three-class gas classification and drift/domain-adaptation analysis.
- Classes: `air`, `alcohol`, `acetone`.
- Current training data: `data/training_enose_data/`, copied from `D:\thesis\enose_data`.
- Current feature tables: `tables/manifest_baseline_as_air_features.csv` (`54D`) and
  `tables/paper8_48d_s1_s7/paper8_48d_features.csv` (`48D`).
- Current split family: `S1-S7` plus per-batch `PB1-PB5` for the classical ML baseline.
- New recorded data: listed in `docs/new_untrained_records_manifest.csv`; not used in
  any current training, validation, test, figure, or conclusion.

## Repository Layout

- `code/training/`: main training and reporting scripts.
- `code/analysis/`: feature extraction, sensor analysis, and figure generation scripts.
- `data/training_enose_data/`: 303 CSV files used by the current thesis pipeline.
- `processed/`: compact processed matrices and metadata used by analysis scripts.
- `tables/`: current result tables and protocol JSON files.
- `figures/`: thesis figures, dashboards, architecture diagrams, and comparison charts.
- `reports/`: HTML/PDF/Markdown thesis reports and GPT handoff notes.
- `docs/`: repository-level handoff documentation and machine-readable manifests.

## Start Here For Codex

Read these files first:

1. `docs/CODEX_HANDOFF.md`
2. `docs/MODELS_AND_RESULTS.md`
3. `docs/NEW_DATA_STATUS.md`
4. `docs/artifact_manifest.csv`

The most important scripts are:

- `code/training/shared_experiment_protocol.py`
- `code/training/run_manifest_da_baseline_splits.py`
- `code/training/run_opensource_da_48_54_benchmark.py`
- `code/training/run_deep_coral_multi_splits.py`
- `code/training/run_cdan_multi_splits.py`
- `code/training/run_dann_multi_splits.py`

## Important Exclusion

`D:\thesis\tables\opensource_da_48_54\predictions.csv` was intentionally not
included because it is about 102.6 MB and exceeds the normal GitHub single-file
limit. The corresponding summary files, protocol, runs, and figures are included.

