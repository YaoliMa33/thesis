"""Protocol-matched PyTorch MLP/DANN refit after validation model selection.

This is a separate experiment and never overwrites established DANN artifacts.
Hyperparameters and epoch budgets are selected from the existing train/validation
runs. The final model is then initialized from scratch and fitted on train +
validation. Test gas labels remain evaluation-only.

For S4-S7, target validation labels enter final classifier training; those final
models are therefore semi-supervised refits and must not be described as UDA.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_dann_multi_splits as base
import run_dann_one_layer_ablation as one_layer
import shared_experiment_protocol as protocol


ROOT = Path(r"D:\thesis")
TABLE_DIR = ROOT / "tables"
OUTPUT_DIR = TABLE_DIR / "dann_train_val_refit"
FEATURE_PATH = TABLE_DIR / "manifest_baseline_as_air_features.csv"
ML_RESULT_PATH = TABLE_DIR / "manifest_da_baseline_split_results.csv"

ARCHITECTURES = {
    "two-layer G_f: 54->32->16": {
        "extractor": base.FeatureExtractor,
        "summary": TABLE_DIR / "dann_multi_split_selected_summary.csv",
        "all_runs": TABLE_DIR / "dann_multi_split_all_runs.csv",
    },
    "one-layer G_f: 54->16": {
        "extractor": one_layer.OneLayerFeatureExtractor,
        "summary": TABLE_DIR / "dann_one_layer" / "dann_multi_split_selected_summary.csv",
        "all_runs": TABLE_DIR / "dann_one_layer" / "dann_multi_split_all_runs.csv",
    },
}


def standardize_from_final_train(
    x_final: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    mean = x_final.mean(axis=0)
    scale = x_final.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    return tuple((array - mean) / scale for array in (x_final, *others))


def domain_metrics(model: base.DANN, x_source: np.ndarray, x_target: np.ndarray) -> dict:
    x_domain = np.vstack([x_source, x_target])
    y_domain = np.concatenate(
        [np.zeros(len(x_source), dtype=int), np.ones(len(x_target), dtype=int)]
    )
    prediction = base.predict_domain(model, base.to_tensor(x_domain))
    source_recall = float((prediction[: len(x_source)] == 0).mean())
    target_recall = float((prediction[len(x_source) :] == 1).mean())
    return {
        "domain_accuracy": float((prediction == y_domain).mean()),
        "domain_balanced_accuracy": 0.5 * (source_recall + target_recall),
        "domain_majority_baseline": max(len(x_source), len(x_target)) / len(y_domain),
    }


def selected_configuration(
    summary: pd.DataFrame,
    all_runs: pd.DataFrame,
    split_name: str,
    selection_variant: str,
    seed: int,
) -> tuple[float, int]:
    selected = summary[
        summary["split"].eq(split_name) & summary["variant"].eq(selection_variant)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one selected row for {split_name} / {selection_variant}, found {len(selected)}."
        )
    lambda_value = float(selected.iloc[0]["lambda"])
    run = all_runs[
        all_runs["split"].eq(split_name)
        & all_runs["variant"].eq(selection_variant)
        & np.isclose(all_runs["lambda"].astype(float), lambda_value)
        & all_runs["seed"].eq(seed)
    ]
    if len(run) != 1:
        raise ValueError(
            f"Expected one seed run for {split_name} / {selection_variant} / seed {seed}, found {len(run)}."
        )
    return lambda_value, int(run.iloc[0]["best_epoch"])


def train_refit(
    extractor_class,
    x_classifier: np.ndarray,
    y_classifier: np.ndarray,
    x_source_domain: np.ndarray,
    x_target_domain: np.ndarray,
    n_classes: int,
    use_domain_loss: bool,
    lambda_value: float,
    epochs: int,
    seed: int,
) -> tuple[base.DANN, dict]:
    if epochs <= 0 or epochs > base.EPOCHS:
        raise ValueError(f"Invalid selected epoch budget: {epochs}")
    original_extractor = base.FeatureExtractor
    try:
        base.FeatureExtractor = extractor_class
        base.set_random_seed(seed)
        model = base.DANN(x_classifier.shape[1], n_classes)
    finally:
        base.FeatureExtractor = original_extractor

    optimizer = torch.optim.Adam(
        model.parameters(), lr=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY
    )
    x_classifier_t = base.to_tensor(x_classifier)
    y_classifier_t = base.to_long(y_classifier)
    class_weight = base.class_weights(y_classifier, n_classes)
    x_domain_t = torch.cat(
        [base.to_tensor(x_source_domain), base.to_tensor(x_target_domain)], dim=0
    )
    y_domain_t = torch.cat(
        [
            torch.zeros(len(x_source_domain), dtype=torch.long),
            torch.ones(len(x_target_domain), dtype=torch.long),
        ],
        dim=0,
    )
    last_l = last_ld = 0.0
    for _epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model.forward_class(x_classifier_t)
        loss_l = F.cross_entropy(logits, y_classifier_t, weight=class_weight)
        loss_ld = torch.tensor(0.0)
        backward_scalar = loss_l
        if use_domain_loss:
            domain_logits = model.forward_domain(x_domain_t, lambda_value)
            loss_ld = F.cross_entropy(domain_logits, y_domain_t)
            backward_scalar = loss_l + loss_ld
        backward_scalar.backward()
        optimizer.step()
        last_l = float(loss_l.detach().cpu())
        last_ld = float(loss_ld.detach().cpu())
    return model, {"L_last": last_l, "Ld_last": last_ld}


def metric_rows(values: pd.Series, prefix: str) -> dict:
    return {
        prefix: float(values.mean()),
        f"{prefix}_std": float(values.std(ddof=1)),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
    }


def summarize(all_runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["architecture", "split", "final_variant"]
    for key, group in all_runs.groupby(keys, sort=False):
        first = group.iloc[0]
        row = {
            "architecture": key[0],
            "split": key[1],
            "final_variant": key[2],
            "description": first["description"],
            "selection_variant": first["selection_variant"],
            "selected_lambda": first["selected_lambda"],
            "seed_values": ",".join(str(value) for value in sorted(group["seed"].unique())),
            "seed_count": int(group["seed"].nunique()),
            "selection_train_n": int(first["selection_train_n"]),
            "validation_n_consumed": int(first["validation_n_consumed"]),
            "final_train_n": int(first["final_train_n"]),
            "target_labels_used_in_final_refit": bool(first["target_labels_used_in_final_refit"]),
            "validation_status": "consumed by final refit; no independent validation metric",
            "test_labels_used_for_training": False,
            "transductive_target_features_used": bool(first["transductive_target_features_used"]),
        }
        row.update(metric_rows(group["selected_epoch"], "selected_epoch"))
        for metric in [
            "train_accuracy", "train_balanced_accuracy", "train_macro_f1",
            "test_accuracy", "test_balanced_accuracy", "test_macro_f1",
            "domain_accuracy", "domain_balanced_accuracy",
        ]:
            row.update(metric_rows(group[metric], metric))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    torch.set_num_threads(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(FEATURE_PATH)
    frame = frame[frame["label"].isin(protocol.THREE_CLASSES)].reset_index(drop=True)
    feature_columns = base.feature_columns(frame)
    all_rows = []
    prediction_rows = []

    for architecture, config in ARCHITECTURES.items():
        summary = pd.read_csv(config["summary"])
        all_runs = pd.read_csv(config["all_runs"])
        for da_split, ml_split in zip(protocol.DA_SPLITS, protocol.ML_SPLITS, strict=True):
            split_name = da_split["split"]
            train_mask = ml_split["train"](frame)
            validation_mask = ml_split["val"](frame)
            test_mask = ml_split["test"](frame)
            final_mask = train_mask | validation_mask
            source_domain_mask = da_split["source_train"](frame) | da_split["source_val"](frame)
            target_domain_mask = da_split["target_domain"](frame)

            final_df = frame[final_mask].copy()
            test_df = frame[test_mask].copy()
            x_final_raw = final_df[feature_columns].to_numpy(float)
            x_source_domain_raw = frame.loc[source_domain_mask, feature_columns].to_numpy(float)
            x_target_domain_raw = frame.loc[target_domain_mask, feature_columns].to_numpy(float)
            x_test_raw = test_df[feature_columns].to_numpy(float)
            x_final, x_source_domain, x_target_domain, x_test = standardize_from_final_train(
                x_final_raw, x_source_domain_raw, x_target_domain_raw, x_test_raw
            )
            y_final = base.encode(final_df["label"].to_numpy(str), protocol.THREE_CLASSES)
            y_test = base.encode(test_df["label"].to_numpy(str), protocol.THREE_CLASSES)
            has_target_labels = bool(
                (final_df["period"].astype(str) != "Period_I").any()
            )
            selection_has_target_labels = bool(da_split["target_labeled_train"](frame).any())
            source_mlp_variant = (
                "MLP source + target labels" if selection_has_target_labels else "MLP source only"
            )
            source_dann_variant = "DANN-semi" if selection_has_target_labels else "DANN-UDA"

            for final_variant, selection_variant, use_domain_loss in [
                ("PyTorch MLP train+validation refit", source_mlp_variant, False),
                ("DANN train+validation refit", source_dann_variant, True),
            ]:
                for seed in protocol.MODEL_SEEDS:
                    lambda_value, selected_epoch = selected_configuration(
                        summary, all_runs, split_name, selection_variant, seed
                    )
                    model, losses = train_refit(
                        config["extractor"],
                        x_final,
                        y_final,
                        x_source_domain,
                        x_target_domain,
                        len(protocol.THREE_CLASSES),
                        use_domain_loss,
                        lambda_value,
                        selected_epoch,
                        seed,
                    )
                    train_prediction = base.predict(model, base.to_tensor(x_final))
                    test_prediction = base.predict(model, base.to_tensor(x_test))
                    train_metrics = base.metric_dict(y_final, train_prediction, len(protocol.THREE_CLASSES))
                    test_metrics = base.metric_dict(y_test, test_prediction, len(protocol.THREE_CLASSES))
                    domain = domain_metrics(model, x_source_domain, x_target_domain)
                    row = {
                        "architecture": architecture,
                        "split": split_name,
                        "description": da_split["description"],
                        "final_variant": final_variant,
                        "selection_variant": selection_variant,
                        "selected_lambda": lambda_value,
                        "selected_epoch": selected_epoch,
                        "seed": seed,
                        "selection_train_n": int(train_mask.sum()),
                        "validation_n_consumed": int(validation_mask.sum()),
                        "final_train_n": int(final_mask.sum()),
                        "test_n": int(test_mask.sum()),
                        "target_labels_used_in_final_refit": has_target_labels,
                        "test_labels_used_for_training": False,
                        "transductive_target_features_used": use_domain_loss,
                        "train_accuracy": train_metrics["accuracy"],
                        "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                        "train_macro_f1": train_metrics["macro_f1"],
                        "test_accuracy": test_metrics["accuracy"],
                        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                        "test_macro_f1": test_metrics["macro_f1"],
                        **domain,
                        **losses,
                    }
                    all_rows.append(row)
                    for local_index, (_, test_record) in enumerate(test_df.iterrows()):
                        prediction_rows.append(
                            {
                                "architecture": architecture,
                                "split": split_name,
                                "final_variant": final_variant,
                                "seed": seed,
                                "selected_lambda": lambda_value,
                                "selected_epoch": selected_epoch,
                                "sample_id": test_record["sample_id"],
                                "target_file": test_record["target_file"],
                                "period": test_record["period"],
                                "day_label": test_record["day_label"],
                                "batch": test_record["batch"],
                                "true_label": protocol.THREE_CLASSES[int(y_test[local_index])],
                                "predicted_label": protocol.THREE_CLASSES[int(test_prediction[local_index])],
                                "correct": bool(y_test[local_index] == test_prediction[local_index]),
                            }
                        )
                    print(f"{architecture} | {split_name} | {final_variant} | seed={seed}")

    all_runs_frame = pd.DataFrame(all_rows)
    summary_frame = summarize(all_runs_frame)
    predictions_frame = pd.DataFrame(prediction_rows)
    all_runs_frame.to_csv(OUTPUT_DIR / "dann_train_val_refit_all_runs.csv", index=False)
    summary_frame.to_csv(OUTPUT_DIR / "dann_train_val_refit_summary.csv", index=False)
    predictions_frame.to_csv(OUTPUT_DIR / "dann_train_val_refit_test_predictions.csv", index=False)

    ml = pd.read_csv(ML_RESULT_PATH)
    ml = ml[(ml["task"].eq("three_class_with_air")) & (ml["model"].eq("MLP"))]
    ml = ml[["split", "test_accuracy", "test_accuracy_std", "test_balanced_accuracy", "test_macro_f1"]].rename(
        columns={
            "test_accuracy": "sklearn_mlp_test_accuracy",
            "test_accuracy_std": "sklearn_mlp_test_accuracy_std",
            "test_balanced_accuracy": "sklearn_mlp_test_balanced_accuracy",
            "test_macro_f1": "sklearn_mlp_test_macro_f1",
        }
    )
    comparison = summary_frame.merge(ml, on="split", how="left")
    comparison["test_accuracy_minus_sklearn_mlp"] = (
        comparison["test_accuracy"] - comparison["sklearn_mlp_test_accuracy"]
    )
    comparison.to_csv(OUTPUT_DIR / "dann_train_val_refit_vs_sklearn_mlp.csv", index=False)

    metadata = {
        "protocol": "validation-only selection followed by from-scratch train+validation refit",
        "validation_after_refit": "consumed; not an independent evaluation partition",
        "test_label_usage": "evaluation only",
        "target_feature_usage": "transductive for DANN domain loss",
        "warning": "S4-S7 final refits use target validation labels and are not UDA",
        "architectures": list(ARCHITECTURES),
        "seeds": list(protocol.MODEL_SEEDS),
    }
    (OUTPUT_DIR / "protocol.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
