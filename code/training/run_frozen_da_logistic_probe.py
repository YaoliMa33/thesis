"""Logistic Regression probes on frozen one/two-layer DA representations.

This is a post-hoc representation evaluation. The sklearn classifier never
participates in DA training and cannot update G_f. Selected DA configurations
are reconstructed from the completed unified run tables. Results checkpoint
after every seed and can be resumed safely.
"""

from __future__ import annotations

import argparse
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
OUTPUT = ROOT / "tables" / "frozen_da_logistic_probe"
FEATURE_PATH = ROOT / "tables" / "manifest_baseline_as_air_features.csv"
ARCHITECTURES = ["one-layer", "two-layer"]
METHODS = ["dann", "cdan", "c-dann", "mk-mmd", "deep-coral", "cdan+c-dann"]
RUN_FILES = {
    "one-layer": [
        INPUT / "runs__one-layer__source-only_dann_deep-coral_mk-mmd__uda_semi.csv",
        INPUT / "runs__one-layer__c-dann_cdan_cdan+c-dann__uda_semi.csv",
    ],
    "two-layer": [
        INPUT / "runs__two-layer__source-only_dann_deep-coral_mk-mmd__uda_semi.csv",
        INPUT / "runs__two-layer__c-dann_cdan_cdan+c-dann__uda_semi.csv",
    ],
}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=ARCHITECTURES)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    return parser.parse_args()


def paths(tag: str) -> tuple[Path, Path, Path]:
    return (
        OUTPUT / f"runs__{tag}.csv",
        OUTPUT / f"predictions__{tag}.csv",
        OUTPUT / f"summary__{tag}.csv",
    )


def load_selected(architectures: list[str], methods: list[str]) -> pd.DataFrame:
    frames = []
    for architecture in architectures:
        for path in RUN_FILES[architecture]:
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            frame = frame[frame["method"].isin(methods)].copy()
            frame["probe_architecture"] = architecture
            frames.append(frame)
    selected = pd.concat(frames, ignore_index=True, sort=False)
    if selected.empty:
        raise ValueError("No completed DA rows match the requested methods/architectures.")
    if "adversarial_strength" not in selected:
        selected["adversarial_strength"] = 0.0
    selected["adversarial_strength"] = selected["adversarial_strength"].fillna(0.0)
    selected = selected.drop_duplicates(["probe_architecture", "split", "variant", "seed"], keep="last")
    return selected.sort_values(["probe_architecture", "method", "split", "variant", "seed"]).reset_index(drop=True)


def checkpoint(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


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
    candidate = suite.Candidate(
        float(row["alignment_strength"]),
        float(row["centroid_strength"]),
        float(row.get("adversarial_strength", 0.0) or 0.0),
    )
    model, _ = suite.train_candidate(
        arrays=arrays,
        hidden_dims=suite.ARCHITECTURES[str(row["probe_architecture"])],
        method=str(row["method"]),
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
            "probe_variant": first["probe_variant"], "method": first["method"], "mode": first["mode"],
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
    options = args()
    tag = "_".join(options.architectures) + "__" + "_".join(options.methods)
    runs_path, predictions_path, summary_path = paths(tag)
    selected = load_selected(options.architectures, options.methods)
    raw = pd.read_csv(FEATURE_PATH)
    columns = suite.feature_columns(raw)
    split_map = {item["split"]: item for item in protocol.DA_SPLITS}
    runs = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame()
    predictions = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()
    run_records = runs.to_dict("records") if not runs.empty else []
    prediction_records = predictions.to_dict("records") if not predictions.empty else []
    done = set(zip(runs.architecture, runs.split, runs.variant, runs.seed.astype(int))) if not runs.empty else set()
    total = len(selected)
    for position, row in selected.iterrows():
        key = (row["probe_architecture"], row["split"], row["variant"], int(row["seed"]))
        if key in done:
            print(f"[{position + 1}/{total}] skip {key}", flush=True)
            continue
        print(f"[{position + 1}/{total}] train {key}", flush=True)
        parts, h_train, y_train, h_test, y_test, label_usage = reconstruct(row, raw, columns, split_map[str(row["split"])])
        classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs")
        classifier.fit(h_train, y_train)
        train_pred, test_pred = classifier.predict(h_train), classifier.predict(h_test)
        train_m, test_m = suite.metric(y_train, train_pred), suite.metric(y_test, test_pred)
        probe_variant = f"Frozen {row['variant']} features + Logistic Regression"
        run_records.append({
            "architecture": row["probe_architecture"], "split": row["split"], "variant": row["variant"],
            "probe_variant": probe_variant, "method": row["method"], "mode": row["mode"], "seed": int(row["seed"]),
            "best_epoch": int(row["best_epoch"]), "alignment_strength": float(row["alignment_strength"]),
            "centroid_strength": float(row["centroid_strength"]), "adversarial_strength": float(row["adversarial_strength"]),
            "final_training": row["final_training"], "label_usage": label_usage, "latent_dimension": h_train.shape[1],
            "logistic_params": "C=1.0;class_weight=balanced;solver=lbfgs;max_iter=2000",
            "neural_test_accuracy": float(row["test_accuracy"]), "lr_train_accuracy": train_m["accuracy"],
            "lr_test_accuracy": test_m["accuracy"], "lr_test_balanced_accuracy": test_m["balanced_accuracy"],
            "lr_test_macro_f1": test_m["macro_f1"], "lr_minus_neural_accuracy": test_m["accuracy"] - float(row["test_accuracy"]),
        })
        test_frame = parts["target_test"]
        for i, (index, sample) in enumerate(test_frame.iterrows()):
            prediction_records.append({
                "architecture": row["probe_architecture"], "split": row["split"], "variant": row["variant"],
                "probe_variant": probe_variant, "seed": int(row["seed"]), "sample_index": int(index),
                "sample_id": sample.get("sample_id", index), "target_file": sample.get("source_file", sample.get("file", "")),
                "period": sample.get("period", ""), "day_label": sample.get("day_label", ""), "batch": sample.get("batch", ""),
                "true_label": suite.CLASSES[int(y_test[i])], "predicted_label": suite.CLASSES[int(test_pred[i])],
                "correct": bool(y_test[i] == test_pred[i]),
            })
        checkpoint(pd.DataFrame(run_records), runs_path)
        checkpoint(pd.DataFrame(prediction_records), predictions_path)
        done.add(key)
    final_runs = pd.DataFrame(run_records)
    checkpoint(summarize(final_runs), summary_path)
    protocol_data = {
        "interpretation": "post-hoc Logistic Regression probe on frozen 16D DA features; not part of DA training",
        "architectures": options.architectures, "methods": options.methods, "seeds": list(protocol.MODEL_SEEDS),
        "logistic": {"C": 1.0, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 2000},
        "checkpointing": "save after every seed and skip completed keys on resume", "test_policy": "evaluation-only",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"protocol__{tag}.json").write_text(json.dumps(protocol_data, indent=2), encoding="utf-8")
    print(f"Outputs written to {OUTPUT}")


if __name__ == "__main__":
    main()
