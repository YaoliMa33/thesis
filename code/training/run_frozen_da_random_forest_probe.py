"""Evaluate frozen one-layer DA representations with a Random Forest probe.

The Random Forest is post-hoc and non-differentiable. It does not participate
in domain adaptation or update G_f. Each selected DA model is reconstructed
with its recorded seed/configuration, G_f is frozen, and RF is fitted on the
allowed labeled latent training features only. Test labels remain evaluation-only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_unified_da_source_target_suite as suite
import shared_experiment_protocol as protocol


ROOT = Path(r"D:\thesis")
INPUT_DIR = ROOT / "tables" / "unified_da_source_target"
OUTPUT_DIR = ROOT / "tables" / "frozen_da_random_forest_probe"
FEATURE_PATH = ROOT / "tables" / "manifest_baseline_as_air_features.csv"

RUN_FILES = [
    INPUT_DIR / "runs__one-layer__source-only_dann_deep-coral_mk-mmd__uda_semi.csv",
    INPUT_DIR / "runs__one-layer__c-dann_cdan_cdan+c-dann__uda_semi.csv",
]
INCLUDED_METHODS = {"source-only", "dann", "cdan", "c-dann", "mk-mmd", "deep-coral"}


def load_selected_runs() -> pd.DataFrame:
    missing = [path for path in RUN_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing unified DA run files: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in RUN_FILES], ignore_index=True, sort=False)
    frame = frame[frame["method"].isin(INCLUDED_METHODS)].copy()
    frame["adversarial_strength"] = frame.get("adversarial_strength", 0.0)
    frame["adversarial_strength"] = frame["adversarial_strength"].fillna(0.0)
    expected = set(protocol.MODEL_SEEDS)
    for key, group in frame.groupby(["split", "variant"]):
        seeds = set(group["seed"].astype(int))
        if seeds != expected:
            raise ValueError(f"{key} has seeds {sorted(seeds)}, expected {sorted(expected)}.")
    return frame


def rf(seed: int) -> RandomForestClassifier:
    params = protocol.RF_DEFAULT
    return RandomForestClassifier(
        n_estimators=protocol.RF_N_ESTIMATORS,
        criterion=protocol.RF_CRITERION,
        max_features=params["max_features"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        class_weight="balanced",
        bootstrap=protocol.RF_BOOTSTRAP,
        random_state=seed,
        n_jobs=protocol.RF_N_JOBS,
    )


def reconstruct(
    row: pd.Series,
    raw: pd.DataFrame,
    columns: list[str],
    split: dict,
) -> tuple[suite.UnifiedModel, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    parts = suite.partitions(raw, split)
    initial_arrays = suite.make_arrays(parts, columns)
    mode = str(row["mode"])
    include_validation = mode == "semi" or len(initial_arrays["x"]["source_val"]) > 0
    if mode == "semi":
        arrays = suite.make_arrays(parts, columns, ("source_train", "target_labeled_train", "target_val"))
    elif include_validation:
        arrays = suite.make_arrays(parts, columns, ("source_train", "source_val"))
    else:
        arrays = initial_arrays
    candidate = suite.Candidate(
        alignment_strength=float(row["alignment_strength"]),
        centroid_strength=float(row["centroid_strength"]),
        adversarial_strength=float(row.get("adversarial_strength", 0.0) or 0.0),
    )
    model, info = suite.train_candidate(
        arrays=arrays,
        hidden_dims=suite.ARCHITECTURES["one-layer"],
        method=str(row["method"]),
        mode=mode,
        candidate=candidate,
        seed=int(row["seed"]),
        include_validation=include_validation,
        fixed_epochs=int(row["best_epoch"]),
    )
    x_train, y_train, label_usage = suite.classifier_data(arrays, mode, include_validation)
    x_test = arrays["x"]["target_test"]
    y_test = arrays["y"]["target_test"]
    model.eval()
    with suite.torch.no_grad():
        h_train = model.features(suite.existing_dann.to_tensor(x_train)).cpu().numpy()
        h_test = model.features(suite.existing_dann.to_tensor(x_test)).cpu().numpy()
    return model, arrays, h_train, y_train, h_test, y_test, {**info, "label_usage": label_usage, "parts": parts}


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in runs.groupby(["split", "variant"], sort=False):
        first = group.iloc[0]
        row = {
            "split": key[0],
            "variant": key[1],
            "probe_variant": first["probe_variant"],
            "method": first["method"],
            "mode": first["mode"],
            "final_training": first["final_training"],
            "label_usage": first["label_usage"],
            "rf_params": first["rf_params"],
            "seed_values": ",".join(map(str, sorted(group.seed.astype(int).unique()))),
        }
        for metric in [
            "neural_test_accuracy", "rf_train_accuracy", "rf_test_accuracy",
            "rf_test_balanced_accuracy", "rf_test_macro_f1", "rf_minus_neural_accuracy",
        ]:
            values = group[metric].astype(float)
            row[metric] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    selected = load_selected_runs()
    raw = pd.read_csv(FEATURE_PATH)
    columns = suite.feature_columns(raw)
    split_map = {item["split"]: item for item in protocol.DA_SPLITS}
    run_rows, prediction_rows = [], []
    for row in selected.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        split = split_map[str(row_series["split"])]
        _, _, h_train, y_train, h_test, y_test, info = reconstruct(row_series, raw, columns, split)
        classifier = rf(int(row_series["seed"]))
        classifier.fit(h_train, y_train)
        train_pred, test_pred = classifier.predict(h_train), classifier.predict(h_test)
        train_metric = suite.metric(y_train, train_pred)
        test_metric = suite.metric(y_test, test_pred)
        neural_accuracy = float(row_series["test_accuracy"])
        probe_variant = f"Frozen {row_series['variant']} features + RF"
        run_rows.append({
            "architecture": "one-layer G_f: 54->16",
            "split": row_series["split"],
            "variant": row_series["variant"],
            "probe_variant": probe_variant,
            "method": row_series["method"],
            "mode": row_series["mode"],
            "seed": int(row_series["seed"]),
            "best_epoch": int(row_series["best_epoch"]),
            "alignment_strength": float(row_series["alignment_strength"]),
            "centroid_strength": float(row_series["centroid_strength"]),
            "final_training": row_series["final_training"],
            "label_usage": info["label_usage"],
            "latent_dimension": h_train.shape[1],
            "rf_params": json.dumps(protocol.RF_DEFAULT, sort_keys=True),
            "neural_test_accuracy": neural_accuracy,
            "rf_train_accuracy": train_metric["accuracy"],
            "rf_test_accuracy": test_metric["accuracy"],
            "rf_test_balanced_accuracy": test_metric["balanced_accuracy"],
            "rf_test_macro_f1": test_metric["macro_f1"],
            "rf_minus_neural_accuracy": test_metric["accuracy"] - neural_accuracy,
        })
        test_frame = info["parts"]["target_test"]
        for position, (index, sample) in enumerate(test_frame.iterrows()):
            prediction_rows.append({
                "architecture": "one-layer G_f: 54->16",
                "split": row_series["split"],
                "variant": row_series["variant"],
                "probe_variant": probe_variant,
                "seed": int(row_series["seed"]),
                "sample_index": int(index),
                "sample_id": sample.get("sample_id", index),
                "target_file": sample.get("source_file", sample.get("file", "")),
                "period": sample.get("period", ""),
                "day_label": sample.get("day_label", ""),
                "batch": sample.get("batch", ""),
                "true_label": suite.CLASSES[int(y_test[position])],
                "predicted_label": suite.CLASSES[int(test_pred[position])],
                "correct": bool(y_test[position] == test_pred[position]),
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(run_rows)
    summary = summarize(runs)
    predictions = pd.DataFrame(prediction_rows)
    runs.to_csv(OUTPUT_DIR / "frozen_da_rf_probe_all_runs.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "frozen_da_rf_probe_summary.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "frozen_da_rf_probe_test_predictions.csv", index=False)
    protocol_payload = {
        "interpretation": "post-hoc Random Forest probe on frozen 16D DA features; RF is not part of DA training",
        "architecture": "one-layer G_f: 54->16",
        "methods": sorted(INCLUDED_METHODS),
        "seeds": list(protocol.MODEL_SEEDS),
        "rf": {"n_estimators": protocol.RF_N_ESTIMATORS, **protocol.RF_DEFAULT, "class_weight": "balanced", "bootstrap": protocol.RF_BOOTSTRAP},
        "selection": "fixed RF defaults; no test-based RF tuning",
    }
    (OUTPUT_DIR / "protocol.json").write_text(json.dumps(protocol_payload, indent=2), encoding="utf-8")
    print(summary[["split", "variant", "rf_test_accuracy", "neural_test_accuracy", "rf_minus_neural_accuracy"]].to_string(index=False))
    print(f"Outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
