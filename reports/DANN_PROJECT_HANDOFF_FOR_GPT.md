# DANN Project Handoff for GPT

## 1. Current scope

The active thesis scope is now limited to:

1. Classical machine-learning baselines.
2. MLP source-only and source-plus-target-label baselines.
3. DANN-UDA and DANN-semi.
4. The custom centroid-alignment C-DANN variants.
5. Official-style CDAN variants and CDAN + centroid-alignment hybrids.

Ordinary Random Forest remains a valid model inside the classical ML baseline. The removed scope comprises DANN-latent Random Forest, RF-54D domain-adaptation extensions, CORAL + RF, Domain-Invariant RF, and DNDF.

## 2. Data and protocol

- Task: three-class classification (`air`, `alcohol`, `acetone`).
- Input per trial: one 54-dimensional vector, consisting of 9 baseline-corrected response features from each of 6 sensors.
- Current feature table: `D:\thesis\tables\manifest_baseline_as_air_features.csv`.
- Current table dimensions: 303 trials and 63 columns, including metadata and 54 model features.
- Raw trial data: `D:\thesis\enose_data`.
- Trial metadata: `D:\thesis\tables\feature_space_current_manifest_samples.csv`.
- Date record supplied by the researcher: `C:\Users\yaoli\Desktop\record_data.xlsx`.
- Shared split, seed, architecture, and RF/ML constants: `D:\thesis\code\training\shared_experiment_protocol.py`.
- Fixed model seeds: `0, 1, 2, 3, 4`. A seed is an independent stochastic repeat and must never be selected using test performance.

### Exact construction of the 54-dimensional input

The neural networks do not receive the complete time series. Each trial is converted into one 54-dimensional feature vector by `extract_legacy9_features()` in `D:\thesis\code\training\run_manifest_da_baseline_splits.py`.

For each sensor `s1` to `s6`:

1. Let `x(t)` be its recorded time series.
2. Compute the sensor baseline `b` as the median of `x(t)` over the trial's baseline phase.
3. Convert every time point to relative baseline response:

   `z(t) = (x(t) - b) / |b|`

4. Split `z(t)` into exposure and recovery phases using the phase rules in `phase_masks()`.
5. Extract nine scalar features:

| Feature | Definition in the current code |
|---|---|
| `exp_mean` | Mean relative response during exposure |
| `exp_absmax` | Maximum absolute relative response during exposure |
| `exp_span` | Exposure maximum minus exposure minimum |
| `exp_tail` | Mean of the final 25% of exposure samples |
| `exp_auc_abs` | Mean absolute exposure response; despite the name, the code does not use trapezoidal time integration |
| `rec_mean` | Mean relative response during recovery |
| `rec_tail` | Mean of the final 25% of recovery samples |
| `rec_slope` | Least-squares slope of relative response against time during recovery |
| `overall_std` | Standard deviation of the relative response over the complete trial |

Therefore:

`6 sensors x 9 features per sensor = 54 input dimensions`.

The resulting model input for `N` trials is a matrix `X` with shape `N x 54`. The gas labels are stored separately as `y` and are not a 55th input feature.

The phase boundaries depend on gas label and recorded duration. For alcohol/acetone, baseline generally ends by 60 seconds, exposure begins at 60 seconds, and exposure end is selected from the duration rules in `phase_masks()`. Air/reference/baseline trials use the air-specific proportional phase rule. These rules must be reviewed as preprocessing assumptions; they must not be silently changed when rerunning experiments.

### Seven chronological splits

- S1: train Period I; no validation; test all later-period trials.
- S2: train 25 and validate 5 fixed Period I trials; test all later-period trials.
- S3: train 20 and validate 10 fixed Period I trials; test all later-period trials.
- S4: train Period I; validate target Day 1-2; test Day 3 onward.
- S5: train Period I plus labeled target Day 1; validate Day 2-3; test Day 4 onward.
- S6: train Period I plus labeled target Day 1-2; validate Day 3-5; test Day 6 onward.
- S7: train Period I plus labeled target Day 1-3; validate Day 4-5; test Day 6 onward.

The target-domain feature matrix currently contains all later-period target trials, including unlabeled test features. Therefore, the current DANN experiments are transductive domain adaptation experiments. They must not be described as prediction for a completely unseen future domain.

## 3. Algorithm background and current notation

### Source-only MLP

- Feature extractor `G_f`: `54 -> 32 -> 16`, ReLU after each linear layer and dropout `0.05` between the two layers.
- Latent representation: `h = G_f(x)` with shape `N x 16`.
- Gas classifier `G_y`: `16 -> C`, where `C=3`.
- Gas logits: `z_y = G_y(h)`, shape `N x 3`.
- Classification loss: `L = CE(z_y, y_cls)`.

### DANN

- Source and target features both pass through the shared `G_f`.
- Source latent features enter `G_y` for gas classification. In semi-supervised variants, the selected labeled target subset also enters the gas-classification loss.
- All source and target latent features enter the domain classifier `G_d` through a gradient-reversal layer.
- Domain labels are `y_dom=0` for source and `y_dom=1` for target.
- Domain loss: `Ld = CE(G_d(GRL_lambda(h_d)), y_dom)`.
- The scalar used for ordinary backpropagation is `L + Ld`; because GRL reverses the gradient reaching `G_f`, the effective feature-extractor objective is `min G_f: L - lambda*Ld`.

### Custom C-DANN used in this project

This is not official CDAN. It adds class-wise latent centroid alignment to DANN:

- Source centroid for class `k`: `mu_s,k`.
- Target centroid for class `k`: `mu_t,k`.
- Conditional loss: `Lc = mean_k ||mu_s,k - mu_t,k||^2`.
- UDA uses target predicted class probabilities as soft class weights.
- Semi-supervised training can use known labels from the selected labeled target subset.
- Effective feature objective: `L - lambda*Ld + alpha*Lc`.

### Official-style CDAN

CDAN conditions the domain discriminator on both the latent representation and gas prediction:

- `g = softmax(z_y)`, shape `N x 3`.
- `h_d`, shape `N x 16`.
- Multilinear condition map: `T(h_d,g) = g outer h_d`, flattened to `N x 48`.
- Conditional domain loss: `Ld_cdan = CE(G_cd(T(h_d,g)), y_dom)`.
- The current implementation detaches `g` for the domain-loss path, so `Ld_cdan` updates the conditional domain discriminator and `G_f`, but not `G_y` through `g`.
- The hybrid `CDAN+C-DANN` adds the centroid loss `Lc` to official-style CDAN.

### Training constants

- Epochs: 400.
- Optimizer: Adam.
- Learning rate: 0.01.
- Weight decay: `1e-4`.
- Dropout: 0.05.
- Seeds: `0-4`.
- Candidate `lambda`: `0.05, 0.2, 0.5` where validation exists.
- Candidate centroid strength `alpha`: `0.05, 0.2, 0.5` where applicable.
- S1 has no validation set and therefore uses fixed defaults. Test labels must not be used to choose hyperparameters.

## 4. Confirmed numerical observations

### DANN-UDA compared with source-only MLP

| Split | MLP test accuracy | DANN-UDA test accuracy | DANN change |
|---|---:|---:|---:|
| S1 | 0.7846 | 0.7473 | -0.0374 |
| S2 | 0.6645 | 0.7040 | +0.0396 |
| S3 | 0.6549 | 0.6740 | +0.0190 |
| S4 | 0.7853 | 0.7829 | -0.0024 |
| S5 | 0.7861 | 0.7810 | -0.0052 |
| S6 | 0.7405 | 0.6976 | -0.0429 |
| S7 | 0.7298 | 0.6976 | -0.0322 |

DANN-UDA improves only S2 and S3 in the current selected results. It is worse than source-only MLP in five of seven splits. Domain-adversarial training therefore does not reliably improve the current representation.

### Instability across seeds

- C-DANN-UDA test-accuracy standard deviation is `0.0807` in S6 and `0.0898` in S7.
- CDAN+C-DANN-UDA test-accuracy standard deviation is `0.1063` in S3.
- These ranges are too large to justify conclusions from one favorable seed.

### Validation does not represent later test periods perfectly

Examples:

- C-DANN-UDA S4: validation accuracy `0.9714`, test accuracy `0.8514`.
- CDAN-UDA S2: validation accuracy `0.9200`, test accuracy `0.7158`.
- High validation accuracy therefore does not imply stable performance under later temporal drift.

### Semi-supervised versus UDA

The statement "semi is often worse than UDA" is not supported as a general conclusion by the current selected means.

- Across DANN, C-DANN, CDAN, and CDAN+C-DANN for S5-S7, semi has higher mean test accuracy in 11 of 12 matched method/split comparisons.
- The only lower selected mean is CDAN+C-DANN in S5: semi `0.8511` versus UDA `0.8528`, a difference of `-0.0017`.
- In matched per-seed comparisons available for DANN, CDAN, and CDAN+C-DANN, semi is lower in 2 of 45 cases. DANN-semi is higher than DANN-UDA in all 15 matched seeds.
- Per-seed all-run results are not currently available for the standalone C-DANN report, so no seed-level claim should be made for that family.

Thus, semi can be worse for an individual seed or method, but the current evidence does not show that this happens frequently. Semi also uses additional target labels, so its comparison with UDA must be described as a different supervision setting rather than a pure algorithmic ablation.

## 5. Confirmed methodological problems

### Domain-accuracy metric is currently misleading

The reported `domain_discriminator_accuracy` is ordinary accuracy on the concatenation of source and target samples. These groups are strongly imbalanced. For example, S1 contains 30 source trials and 273 target trials, so a discriminator predicting "target" for every sample already obtains `273/303 = 0.9010` accuracy. Many reported domain accuracies are approximately 0.90.

Consequences:

1. A reported domain accuracy near 0.90 does not prove that the discriminator learned strong domain separation; it may be the majority-domain baseline.
2. It also does not prove successful domain confusion.
3. The domain cross-entropy itself is currently computed on the imbalanced full source-target matrix, which can bias optimization toward the target domain.

Required general correction:

- Train the domain branch using balanced source/target minibatches or a justified class-weighted domain cross-entropy.
- Report domain balanced accuracy, source-domain recall, target-domain recall, and the majority-domain baseline.
- Do not interpret raw domain accuracy alone.

This correction changes the training algorithm and requires a new experiment family. Old and corrected results must be retained separately and clearly named.

### Per-trial DANN predictions are missing

Current DANN result tables save seed-level aggregate metrics but not `sample_id`, `target_file`, true label, and predicted label for every test trial. Therefore, the following question cannot currently be answered from saved evidence:

> Which exact trials were classified incorrectly by every seed?

Do not infer this from aggregate accuracy, confusion matrices, or PCA plots. The next run must write a prediction table containing at least:

`split, variant, lambda, alpha, seed, sample_id, target_file, period, day, batch, true_label, predicted_label, correct`

Afterward, group by `split + variant + sample_id` and identify rows for which `correct=False` for all five seeds. Also summarize persistent errors by true gas, predicted gas, day, batch, and trial duration.

### Feature-space plots are diagnostic only

PCA source-target overlap does not by itself prove domain invariance or good class discrimination. Report at least one quantitative domain-discrepancy measure and class-conditional separation metric, while keeping target test gas labels excluded from training and hyperparameter selection.

## 6. Files the next GPT should inspect first

### Protocol and data

1. `D:\thesis\code\training\shared_experiment_protocol.py`
2. `D:\thesis\tables\manifest_baseline_as_air_features.csv`
3. `D:\thesis\tables\feature_space_current_manifest_samples.csv`
4. `D:\thesis\enose_data`
5. `C:\Users\yaoli\Desktop\record_data.xlsx`

### ML code and results

1. `D:\thesis\code\training\run_manifest_da_baseline_splits.py`
2. `D:\thesis\tables\manifest_da_baseline_split_results.csv`
3. `D:\thesis\tables\manifest_da_baseline_split_test_predictions.csv`
4. `D:\thesis\figures\ml_learning`

### DANN/C-DANN/CDAN code

1. `D:\thesis\code\training\run_dann_multi_splits.py`
2. `D:\thesis\code\training\build_dann_conditional_report.py`
3. `D:\thesis\code\training\run_cdan_multi_splits.py`
4. `D:\thesis\code\training\run_cdan_hybrid_report.py`
5. `D:\thesis\code\training\build_dann_interactive_report.py`
6. `D:\thesis\code\training\audit_shared_experiment_protocol.py`

### DANN result evidence

1. `D:\thesis\tables\dann_multi_split_all_runs.csv`
2. `D:\thesis\tables\dann_multi_split_selected_summary.csv`
3. `D:\thesis\tables\dann_multi_split_partitions.csv`
4. `D:\thesis\tables\dann_conditional_selected_summary.csv`
5. `D:\thesis\tables\cdan_multi_split_all_runs.csv`
6. `D:\thesis\tables\cdan_multi_split_selected_summary.csv`
7. `D:\thesis\tables\cdan_hybrid_all_runs.csv`
8. `D:\thesis\tables\cdan_hybrid_selected_summary.csv`
9. `D:\thesis\tables\dann_interactive_training_curves.csv`
10. `D:\thesis\tables\dann_interactive_feature_space.csv`
11. `D:\thesis\figures\dann_learning`

## 7. Prompt for the next GPT

Use the following prompt together with this handoff file:

> Continue the e-nose thesis project from the existing files under `D:\thesis`. The active scope is classical machine learning plus MLP, DANN, the custom centroid-alignment C-DANN, official-style CDAN, and CDAN+C-DANN. Do not restore RF domain-adaptation extensions or DNDF. Preserve the three-class task (`air/alcohol/acetone`), the shared S1-S7 chronological protocol, 54-dimensional trial features, and fixed repeat seeds 0-4. First audit the current code and result tables listed in `D:\thesis\reports\DANN_PROJECT_HANDOFF_FOR_GPT.md`. Do not claim that semi is generally worse than UDA: current selected means show semi is better in 11/12 matched comparisons. The current DANN tables do not contain per-trial predictions, so add a reproducible prediction export and rerun before identifying samples misclassified by all seeds. Also audit the imbalanced domain loss and misleading raw domain accuracy: use a separate, clearly named corrected experiment with balanced source/target domain training and report balanced domain metrics. Never use target test gas labels for training, model selection, stopping, or hyperparameter tuning. Keep old result files separate from corrected results and document every protocol change.

## 8. Immediate recommended next experiment

1. Add per-trial prediction export without changing training behavior and rerun the current selected configurations to establish the persistent-error baseline.
2. Implement balanced source/target domain minibatches or class-weighted domain loss as a separately named DANN correction.
3. Use validation metrics only for hyperparameter selection; S1 must remain fixed-default or be excluded from hyperparameter claims.
4. Compare corrected DANN against the unchanged MLP baseline using mean, sample standard deviation, minimum, and maximum over seeds 0-4.
5. Report both gas-class metrics and balanced domain-discriminator metrics.
