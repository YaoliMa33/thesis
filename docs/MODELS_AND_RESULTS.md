# Models And Current Conclusions

This document summarizes what is in the repository. It is a handoff note, not a
new experiment.

## Feature Families

- `54D`: custom six-sensor feature table with 9 features per sensor, stored at
  `tables/manifest_baseline_as_air_features.csv`.
- `48D`: paper-style six-sensor feature table with 8 features per sensor, stored
  at `tables/paper8_48d_s1_s7/paper8_48d_features.csv`.

Do not introduce a `56D` branch unless a new feature table is explicitly created
and verified.

## Classical ML Models

The classical baseline is implemented mainly in
`code/training/run_manifest_da_baseline_splits.py` and shares constants through
`code/training/shared_experiment_protocol.py`.

Included models:

- `LDA`
- `Logistic Regression`
- `kNN`
- `Random Forest`
- `RBF SVM`
- `MLP`

Current protocol:

- Protocol version: `enose_three_class_ml_v2_2026-07-01`.
- Preprocessing: `Pipeline(StandardScaler, classifier)`.
- Stochastic repeated models: RF and MLP use seeds `0,1,2,3,4`.
- RF bootstrap is normal RF bagging, not bootstrap uncertainty estimation.
- If validation exists, selection is validation-only followed by final refit on
  `train+validation`; test remains untouched.
- `S1` has no validation and stays `train_only_no_validation`.

Best classical rows in the current `S1-S7` table include:

- Logistic Regression on `S6_train_I_D1_2_val_D3_5`: test accuracy about `0.9707`.
- Logistic Regression on `S7_train_I_D1_3_val_D4_5`: test accuracy about `0.9707`.
- RBF SVM on `S7_train_I_D1_3_val_D4_5`: test accuracy about `0.9659`.

## Domain Adaptation Models

The DA family is implemented across several scripts, especially:

- `code/training/run_opensource_da_48_54_benchmark.py`
- `code/training/opensource_da_components.py`
- `code/training/run_dann_multi_splits.py`
- `code/training/run_cdan_multi_splits.py`
- `code/training/run_deep_coral_multi_splits.py`
- `code/training/run_unified_da_source_target_suite.py`

Included DA methods:

- `source-only`
- `DANN`
- `CDAN`
- `C-DANN`
- `Deep CORAL`
- `MK-MMD`
- `CDAN+C-DANN`
- `MK-MMD+CDAN`
- `MK-MMD+C-DANN`
- `MK-MMD+CDAN+C-DANN`
- `MK-MMD+LC` and related explicitly named local variants

Current DA benchmark protocol:

- Classes: `air`, `alcohol`, `acetone`.
- Feature sets: `48D` and `54D`.
- Splits: `S1-S7`.
- Seeds: `0,1,2,3,4`.
- Epochs: 400.
- Optimizer: Adam.
- Learning rate: `0.01`.
- Weight decay: `0.0001`.
- Target test labels are evaluation-only.

## Current Interpretation

The strongest general thesis signal is that `54D` is usually better than `48D`
for this local dataset. The compact benchmark table shows that `54D` wins the
overwhelming majority of matched DA configurations, while a few `S7` cases still
favor `48D`.

The focused 54D branch shows strong DA performance with bottleneck16 + linear
head variants. The top rows in
`tables/opensource_da_48_54/focused_bottleneck16_accuracy_results.csv` include
`54D` `S7` `source-only`/`Deep CORAL` and `54D` `S6` `Deep CORAL` or
`source-only`, with test accuracy around `0.965-0.972` in the best rows.

The results do not support a blanket claim that stacked or hybrid losses always
beat single-loss methods. Negative or mixed results should be preserved as thesis
evidence for method selection under drift.

## Suggested Method-Selection Framing

Use local evidence rather than generic claims:

- Small observed drift and high source confidence: classical LR or source-only
  neural baselines may be enough.
- Large marginal drift with weak target confidence: `MK-MMD` or `Deep CORAL` are
  defensible candidates.
- Strong class-conditional mismatch: `CDAN` or `C-DANN` can be considered.
- A few reliable labeled target samples: semi-supervised DA branches are relevant.

This is a framing for future validation. It must not become post-hoc test-set
tuning.

