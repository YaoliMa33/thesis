# Codex Handoff

This repository is intended to let another Codex session continue the thesis work
without needing the original Windows `D:\thesis` tree.

## Operating Assumptions

- Keep the thesis scope as three-class classification unless explicitly changed:
  `air`, `alcohol`, `acetone`.
- Use `train / validation / test` wording for classical ML baselines.
- Keep domain-adaptation method names distinct: `DANN`, `CDAN`, `C-DANN`,
  `Deep CORAL`, `MK-MMD`, and explicitly named hybrid objectives.
- Do not treat the newly recorded data as part of the existing training set.
- Do not tune, select, or justify methods from the final test labels.

## Current Training Data Boundary

The current trained/reported pipeline uses:

- `data/training_enose_data/acetone/`
- `data/training_enose_data/air/`
- `data/training_enose_data/alcohol/`

This copied training set has 303 CSV files. The current 54D feature table
`tables/manifest_baseline_as_air_features.csv` also has 303 rows.

The newer recording found under the local path `D:\enose_data\enose_data` is only
listed in `docs/new_untrained_records_manifest.csv`. It is not included in the
current train/validation/test definitions and was not used to make the figures or
conclusions in this repository.

## Main Reproducibility Entry Points

Classical ML baseline:

```powershell
python code/training/run_manifest_da_baseline_splits.py
```

Main DA 48D/54D benchmark:

```powershell
python code/training/run_opensource_da_48_54_benchmark.py
```

Independent Deep CORAL branch:

```powershell
python code/training/run_deep_coral_multi_splits.py
```

Unified DA suite and related reports:

```powershell
python code/training/run_unified_da_source_target_suite.py
python code/training/build_unified_thesis_dashboard.py
```

When rerunning, inspect each script's path constants first. Some scripts were
developed against the original absolute `D:\thesis` layout and may need path
normalization before running on another machine.

## Key Artifact Map

- `tables/manifest_da_baseline_split_results.csv`: classical ML `S1-S7` results.
- `tables/manifest_da_baseline_period_i_per_batch_results.csv`: `PB1-PB5` batch
  stress-test results.
- `tables/opensource_da_48_54/focused_bottleneck16_accuracy_results.csv`: compact
  48D/54D DA comparison.
- `tables/opensource_da_48_54/best_by_feature_and_split_descriptive.csv`: best
  descriptive result per feature/split.
- `figures/ml_learning/`: classical ML summary figures.
- `figures/opensource_model_comparison/`: 48D/54D comparison figures.
- `figures/opensource_model_explanatory/`: explanatory DA figures.
- `reports/ml_results_confusion_explorer.html`: interactive classical ML confusion
  report.
- `figures/unified_results_dashboard.html`: large unified dashboard.

## Large File Note

The full `tables/opensource_da_48_54/predictions.csv` file is excluded from Git
because it is larger than 100 MB. Use the included `runs.csv`, `summary.csv`,
`focused_bottleneck16_accuracy_results.csv`, and figures for normal continuation.
If row-level predictions are needed later, restore them from the local source
`D:\thesis\tables\opensource_da_48_54\predictions.csv` or use Git LFS.

