"""Frozen one/two-layer MLP latent features with sklearn Logistic Regression.

Each selected PyTorch MLP is reconstructed with its recorded seed and training
protocol. G_f is then frozen, 16D latent features are extracted, and a fixed
multinomial Logistic Regression probe is fitted on allowed labeled training
data. Rows and predictions are checkpointed after every seed and resumable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_unified_da_source_target_suite as suite
import shared_experiment_protocol as protocol


ROOT = Path(r"D:\thesis")
INPUT = ROOT / "tables" / "unified_da_source_target"
OUTPUT = ROOT / "tables" / "frozen_mlp_logistic_probe"
FEATURE_PATH = ROOT / "tables" / "manifest_baseline_as_air_features.csv"
RUN_PATHS = {
    "one-layer": INPUT / "runs__one-layer__source-only_dann_deep-coral_mk-mmd__uda_semi.csv",
    "two-layer": INPUT / "runs__two-layer__source-only_dann_deep-coral_mk-mmd__uda_semi.csv",
}
RUNS_OUT = OUTPUT / "frozen_mlp_logistic_probe_all_runs.csv"
PRED_OUT = OUTPUT / "frozen_mlp_logistic_probe_test_predictions.csv"
SUMMARY_OUT = OUTPUT / "frozen_mlp_logistic_probe_summary.csv"


def selected_rows() -> pd.DataFrame:
    frames = []
    for architecture, path in RUN_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame = frame[frame["method"].eq("source-only")].copy()
        frame["probe_architecture"] = architecture
        frames.append(frame)
    selected = pd.concat(frames, ignore_index=True, sort=False)
    selected["adversarial_strength"] = 0.0
    return selected.sort_values(["probe_architecture", "split", "variant", "seed"]).reset_index(drop=True)


def checkpoint(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def existing() -> tuple[pd.DataFrame, pd.DataFrame, set[tuple]]:
    runs = pd.read_csv(RUNS_OUT) if RUNS_OUT.exists() else pd.DataFrame()
    predictions = pd.read_csv(PRED_OUT) if PRED_OUT.exists() else pd.DataFrame()
    done = set()
    if not runs.empty:
        done = set(zip(runs["architecture"], runs["split"], runs["variant"], runs["seed"].astype(int)))
    return runs, predictions, done


def reconstruct(row: pd.Series, raw: pd.DataFrame, columns: list[str], split: dict):
    parts = suite.partitions(raw, split)
    initial = suite.make_arrays(parts, columns)
    mode = str(row["mode"])
    include_validation = mode == "semi" or len(initial["x"]["source_val"]) > 0
    if mode == "semi":
        arrays = suite.make_arrays(parts, columns, ("source_train", "target_labeled_train", "target_val"))
    elif include_validation:
        arrays = suite.make_arrays(parts, columns, ("source_train", "source_val"))
    else:
        arrays = initial
    candidate = suite.Candidate(0.0, 0.0, 0.0)
    model, _ = suite.train_candidate(
        arrays=arrays,
        hidden_dims=suite.ARCHITECTURES[str(row["probe_architecture"])],
        method="source-only",
        mode=mode,
        candidate=candidate,
        seed=int(row["seed"]),
        include_validation=include_validation,
        fixed_epochs=int(row["best_epoch"]),
    )
    x_train, y_train, label_usage = suite.classifier_data(arrays, mode, include_validation)
    x_test, y_test = arrays["x"]["target_test"], arrays["y"]["target_test"]
    model.eval()
    with suite.torch.no_grad():
        h_train = model.features(suite.existing_dann.to_tensor(x_train)).cpu().numpy()
        h_test = model.features(suite.existing_dann.to_tensor(x_test)).cpu().numpy()
    return parts, h_train, y_train, h_test, y_test, label_usage


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in runs.groupby(["architecture", "split", "variant"], sort=False):
        first = group.iloc[0]
        row = {
            "architecture": key[0], "split": key[1], "variant": key[2],
            "probe_variant": first["probe_variant"], "mode": first["mode"],
            "final_training": first["final_training"], "label_usage": first["label_usage"],
            "logistic_params": first["logistic_params"],
            "seed_values": ",".join(map(str, sorted(group.seed.astype(int).unique()))),
        }
        for metric in ["neural_test_accuracy", "lr_train_accuracy", "lr_test_accuracy", "lr_test_balanced_accuracy", "lr_test_macro_f1", "lr_minus_neural_accuracy"]:
            values = group[metric].astype(float)
            row[metric] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    selected = selected_rows()
    raw = pd.read_csv(FEATURE_PATH)
    columns = suite.feature_columns(raw)
    split_map = {item["split"]: item for item in protocol.DA_SPLITS}
    runs, predictions, done = existing()
    run_records = runs.to_dict("records") if not runs.empty else []
    prediction_records = predictions.to_dict("records") if not predictions.empty else []
    total = len(selected)
    for number, row in selected.iterrows():
        key = (row["probe_architecture"], row["split"], row["variant"], int(row["seed"]))
        if key in done:
            print(f"[{number + 1}/{total}] skip completed {key}", flush=True)
            continue
        print(f"[{number + 1}/{total}] train {key}", flush=True)
        parts, h_train, y_train, h_test, y_test, label_usage = reconstruct(row, raw, columns, split_map[str(row["split"])])
        probe = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs")
        probe.fit(h_train, y_train)
        train_pred, test_pred = probe.predict(h_train), probe.predict(h_test)
        train_m, test_m = suite.metric(y_train, train_pred), suite.metric(y_test, test_pred)
        record = {
            "architecture": row["probe_architecture"], "split": row["split"], "variant": row["variant"],
            "probe_variant": f"Frozen {row['variant']} features + Logistic Regression",
            "mode": row["mode"], "seed": int(row["seed"]), "best_epoch": int(row["best_epoch"]),
            "final_training": row["final_training"], "label_usage": label_usage, "latent_dimension": h_train.shape[1],
            "logistic_params": "C=1.0;class_weight=balanced;solver=lbfgs;max_iter=2000",
            "neural_test_accuracy": float(row["test_accuracy"]),
            "lr_train_accuracy": train_m["accuracy"], "lr_test_accuracy": test_m["accuracy"],
            "lr_test_balanced_accuracy": test_m["balanced_accuracy"], "lr_test_macro_f1": test_m["macro_f1"],
            "lr_minus_neural_accuracy": test_m["accuracy"] - float(row["test_accuracy"]),
        }
        run_records.append(record)
        test_frame = parts["target_test"]
        for position, (index, sample) in enumerate(test_frame.iterrows()):
            prediction_records.append({
                "architecture": row["probe_architecture"], "split": row["split"], "variant": row["variant"],
                "probe_variant": record["probe_variant"], "seed": int(row["seed"]), "sample_index": int(index),
                "sample_id": sample.get("sample_id", index), "target_file": sample.get("source_file", sample.get("file", "")),
                "period": sample.get("period", ""), "day_label": sample.get("day_label", ""), "batch": sample.get("batch", ""),
                "true_label": suite.CLASSES[int(y_test[position])], "predicted_label": suite.CLASSES[int(test_pred[position])],
                "correct": bool(y_test[position] == test_pred[position]),
            })
        checkpoint(pd.DataFrame(run_records), RUNS_OUT)
        checkpoint(pd.DataFrame(prediction_records), PRED_OUT)
        done.add(key)
    runs = pd.DataFrame(run_records)
    summary = summarize(runs)
    checkpoint(summary, SUMMARY_OUT)
    protocol_data = {
        "interpretation": "sklearn Logistic Regression probe on frozen 16D MLP representations",
        "architectures": {key: list(value) for key, value in suite.ARCHITECTURES.items()},
        "variants": ["MLP Source-only", "MLP Source+Target"], "seeds": list(protocol.MODEL_SEEDS),
        "logistic": {"C": 1.0, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 2000},
        "checkpointing": "runs and predictions saved after every seed; reruns skip completed keys",
        "test_policy": "test labels evaluation-only",
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol_data, indent=2), encoding="utf-8")
    print(summary[["architecture", "split", "variant", "lr_test_accuracy", "neural_test_accuracy", "lr_minus_neural_accuracy"]].to_string(index=False))
    print(f"Outputs written to {OUTPUT}")


if __name__ == "__main__":
    main()
