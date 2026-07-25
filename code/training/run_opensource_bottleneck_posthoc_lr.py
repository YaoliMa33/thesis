"""Post-hoc sklearn Logistic Regression on frozen 16D DA checkpoints.

This is the third classifier strategy in the focused comparison:
1. end-to-end MLP head;
2. end-to-end linear/logistic head;
3. frozen MLP-head representation followed by a post-hoc sklearn LR probe.

The neural checkpoints come from ``run_opensource_da_48_54_benchmark.py``.
No neural model is retrained. Test labels are evaluation-only. Every completed
feature-set/split/method/seed probe is saved immediately and skipped on resume.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_opensource_da_48_54_benchmark as benchmark
import run_unified_da_source_target_suite as suite
import shared_experiment_protocol as protocol


ROOT = Path(r"D:\thesis\tables\opensource_da_48_54")
SOURCE_RUNS = ROOT / "runs.csv"
OUTPUT = ROOT / "posthoc_lr_bottleneck16"
RUNS = OUTPUT / "runs.csv"
PREDICTIONS = OUTPUT / "predictions.csv"
SUMMARY = OUTPUT / "summary.csv"
STATUS = OUTPUT / "status.json"
METHODS = (
    "source-only", "dann", "deep-coral", "mk-mmd", "cdan", "c-dann", "cdan+c-dann",
    "mk-mmd+cdan", "mk-mmd+c-dann", "mk-mmd+cdan+c-dann",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-sets", nargs="+", choices=("48D", "54D"), default=["48D", "54D"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--splits", nargs="+", default=[f"S{i}" for i in range(1, 8)])
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False); os.replace(temporary, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8"); os.replace(temporary, path)


def records(path: Path) -> list[dict]:
    return pd.read_csv(path).to_dict("records") if path.exists() else []


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def frozen_features(model: benchmark.TabularDAModel, values: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.features(torch.as_tensor(values, dtype=torch.float32)).cpu().numpy()


def arrays_for_checkpoint(parts: dict, columns: list[str], mode: str, final_training: str):
    was_refit = "refit" in final_training and "no refit" not in final_training
    return benchmark.final_arrays(parts, columns, mode, was_refit)


def summarize(run_records: list[dict]) -> None:
    frame = pd.DataFrame(run_records)
    rows = []
    groups = ["feature_set", "split", "method", "representation", "classifier_strategy"]
    for keys, group in frame.groupby(groups, sort=False):
        row = dict(zip(groups, keys)); row["seed_count"] = len(group)
        row["seeds"] = ",".join(str(int(value)) for value in sorted(group.seed.unique()))
        for metric in ("train_accuracy", "test_accuracy", "test_balanced_accuracy", "test_macro_f1"):
            values = group[metric].astype(float)
            row[metric] = float(values.mean()); row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)
    atomic_csv(pd.DataFrame(rows), SUMMARY)


def main() -> None:
    args = parse_args(); args.splits = benchmark.resolve_splits(args.splits)
    if not SOURCE_RUNS.exists():
        raise FileNotFoundError(SOURCE_RUNS)
    source = pd.read_csv(SOURCE_RUNS)
    source = source[
        source["family"].eq("domain-adaptation")
        & source["feature_set"].isin(args.feature_sets)
        & source["split"].isin(args.splits)
        & source["method"].isin(args.methods)
        & source["representation"].eq("bottleneck16")
        & source["head"].eq("mlp")
    ].copy()
    expected = len(args.feature_sets) * len(args.splits) * len(args.methods) * len(protocol.MODEL_SEEDS)
    if len(source) != expected:
        counts = source.groupby(["feature_set", "split", "method"]).size()
        raise ValueError(f"Expected {expected} complete neural checkpoints, found {len(source)}. Counts:\n{counts}")
    run_records, prediction_records = records(RUNS), records(PREDICTIONS)
    requested_keys = {
        (row.feature_set, row.split, row.method, int(row.seed))
        for row in source.itertuples()
    }
    completed = {
        (row["feature_set"], row["split"], row["method"], int(row["seed"]))
        for row in run_records
    } & requested_keys
    split_map = {item["split"]: item for item in protocol.DA_SPLITS}
    feature_cache = {}
    status = {"state": "running", "total": expected, "completed": len(completed), "current": None}
    atomic_json(status, STATUS)
    try:
        for number, row in enumerate(source.sort_values(["feature_set", "split", "method", "seed"]).itertuples(), 1):
            key = (row.feature_set, row.split, row.method, int(row.seed))
            if key in completed:
                continue
            status.update({"current": list(key), "task_number": number}); atomic_json(status, STATUS)
            print(f"[{number}/{expected}] {' | '.join(map(str, key))}", flush=True)
            if row.feature_set not in feature_cache:
                feature_cache[row.feature_set] = benchmark.load_feature_set(row.feature_set)
            frame, columns = feature_cache[row.feature_set]
            parts = suite.partitions(frame, split_map[row.split])
            arrays, include_validation, _ = arrays_for_checkpoint(parts, columns, row.mode, row.final_training)
            x_train, y_train, label_usage = suite.classifier_data(arrays, row.mode, include_validation)
            x_test, y_test = arrays["x"]["target_test"], arrays["y"]["target_test"]
            checkpoint = Path(row.checkpoint)
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = benchmark.TabularDAModel(len(columns), "bottleneck16", "mlp")
            model.load_state_dict(payload["state_dict"])
            train_h, test_h = frozen_features(model, x_train), frozen_features(model, x_test)
            classifier = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs")
            classifier.fit(train_h, y_train)
            train_pred, test_pred = classifier.predict(train_h), classifier.predict(test_h)
            probe_path = OUTPUT / "checkpoints" / row.feature_set / row.split / row.method / f"seed_{int(row.seed)}.joblib"
            probe_path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(classifier, probe_path)
            train_metric, test_metric = metrics(y_train, train_pred), metrics(y_test, test_pred)
            run_record = {
                "feature_set": row.feature_set, "split": row.split, "mode": row.mode,
                "method": row.method, "representation": "bottleneck16",
                "classifier_strategy": "posthoc-sklearn-lr-on-frozen-mlp-representation",
                "seed": int(row.seed), "input_dimensions": len(columns), "latent_dimensions": 16,
                "neural_checkpoint": str(checkpoint), "probe_checkpoint": str(probe_path),
                "label_usage": label_usage, "final_training": row.final_training,
                "lr_parameters": "C=1.0;class_weight=balanced;solver=lbfgs;max_iter=2000",
                "confusion_matrix": json.dumps(confusion_matrix(
                    y_test, test_pred, labels=range(len(protocol.THREE_CLASSES))
                ).tolist()),
                **{f"train_{name}": value for name, value in train_metric.items()},
                **{f"test_{name}": value for name, value in test_metric.items()},
            }
            run_records.append(run_record)
            for position, (index, sample) in enumerate(parts["target_test"].iterrows()):
                prediction_records.append({
                    "feature_set": row.feature_set, "split": row.split, "method": row.method,
                    "representation": "bottleneck16", "seed": int(row.seed),
                    "sample_index": int(index), "sample_id": sample["sample_id"],
                    "true_label": protocol.THREE_CLASSES[int(y_test[position])],
                    "predicted_label": protocol.THREE_CLASSES[int(test_pred[position])],
                    "correct": bool(y_test[position] == test_pred[position]),
                })
            atomic_csv(pd.DataFrame(run_records), RUNS); atomic_csv(pd.DataFrame(prediction_records), PREDICTIONS)
            summarize(run_records); completed.add(key)
            status.update({"completed": len(completed), "current": None}); atomic_json(status, STATUS)
        status.update({"state": "complete", "current": None}); atomic_json(status, STATUS)
    except BaseException as error:
        status.update({"state": "interrupted", "error": repr(error)}); atomic_json(status, STATUS)
        raise


if __name__ == "__main__":
    main()
