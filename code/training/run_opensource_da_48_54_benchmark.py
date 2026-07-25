"""Reproducible 48D/54D ML and domain-adaptation benchmark.

The core DA losses are tabular adapters of published open-source code in
``opensource_da_components.py``. Project extensions are explicitly separated
from standard algorithms in every result row.

Every completed model/seed is written immediately. Re-running the same command
skips completed keys, so an interrupted run loses at most the current seed.
Existing thesis result families are never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import opensource_da_components as osc
import run_manifest_da_baseline_splits as mlbase
import run_paper8_48d_s1_s7_models as paper48
import run_unified_da_source_target_suite as existing_suite
import shared_experiment_protocol as protocol


ROOT = Path(r"D:\thesis")
OUTPUT = ROOT / "tables" / "opensource_da_48_54"
CHECKPOINTS = OUTPUT / "checkpoints"
RUNS_PATH = OUTPUT / "runs.csv"
PREDICTIONS_PATH = OUTPUT / "predictions.csv"
CURVES_PATH = OUTPUT / "curves.csv"
SUMMARY_PATH = OUTPUT / "summary.csv"
BEST_PATH = OUTPUT / "best_by_feature_and_split_descriptive.csv"
STATUS_PATH = OUTPUT / "status.json"
PROTOCOL_PATH = OUTPUT / "protocol.json"

FEATURE_PATHS = {
    "54D": ROOT / "tables" / "manifest_baseline_as_air_features.csv",
    "48D": ROOT / "tables" / "paper8_48d_s1_s7" / "paper8_48d_features.csv",
}
ML_MODELS = ("logistic-regression", "mlp")
METHODS = (
    "source-only", "dann", "deep-coral", "mk-mmd", "c-dann", "cdan",
    "cdan+c-dann", "mk-mmd+lc", "mk-mmd+cdan", "mk-mmd+lc+cdan",
    "mk-mmd+c-dann", "mk-mmd+cdan+c-dann",
)
CORE_METHODS = ("source-only", "dann", "deep-coral", "mk-mmd", "cdan")
EXTENSION_METHODS = tuple(method for method in METHODS if method not in CORE_METHODS)
REPRESENTATIONS = ("bottleneck16", "residual-d", "fusion16")
HEADS = ("linear", "mlp")
SEEDS = tuple(int(seed) for seed in protocol.MODEL_SEEDS)
EPOCHS = int(protocol.NEURAL_EPOCHS)
EVAL_EVERY = 10
CORAL_WEIGHT = 1.0
MMD_WEIGHT = 1.0
CENTROID_WEIGHT = 0.1
ML_PARAM_CACHE: dict[tuple[str, str, str], tuple[dict, str]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-sets", nargs="+", choices=FEATURE_PATHS, default=list(FEATURE_PATHS))
    parser.add_argument("--splits", nargs="+", default=[f"S{i}" for i in range(1, 8)])
    parser.add_argument("--ml-models", nargs="*", choices=ML_MODELS, default=list(ML_MODELS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=["bottleneck16"])
    parser.add_argument("--heads", nargs="+", choices=HEADS, default=list(HEADS))
    parser.add_argument("--no-refit", action="store_true", help="Do not refit train+validation after epoch selection.")
    parser.add_argument("--force", action="store_true", help="Recompute requested completed keys.")
    return parser.parse_args()


def resolve_splits(requested: list[str]) -> list[str]:
    available = [item["split"] for item in protocol.DA_SPLITS]
    aliases = {name.split("_", 1)[0].upper(): name for name in available}
    resolved = [aliases.get(value.upper(), value) for value in requested]
    unknown = sorted(set(resolved) - set(available))
    if unknown:
        raise ValueError(f"Unknown S1-S7 splits: {unknown}")
    if len(resolved) != len(set(resolved)):
        raise ValueError("Duplicate splits were requested after alias resolution.")
    return resolved


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_records(path: Path) -> list[dict]:
    return pd.read_csv(path).to_dict("records") if path.exists() else []


def sensor_columns(frame: pd.DataFrame, expected: int) -> list[str]:
    columns = [column for column in frame.columns if any(column.startswith(f"s{i}_") for i in range(1, 7))]
    if len(columns) != expected:
        raise ValueError(f"Expected {expected} sensor features, found {len(columns)}.")
    if not np.isfinite(frame[columns].to_numpy(float)).all():
        raise ValueError(f"{expected}D table contains non-finite feature values.")
    return columns


def load_feature_set(name: str) -> tuple[pd.DataFrame, list[str]]:
    path = FEATURE_PATHS[name]
    if not path.exists():
        if name == "48D":
            paper48.build_or_load_features(True)
        else:
            raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    expected = int(name[:-1])
    columns = sensor_columns(frame, expected)
    if len(frame) != 303:
        raise ValueError(f"{name} must contain 303 trials, found {len(frame)}.")
    counts = frame["label"].value_counts().to_dict()
    if counts != {"acetone": 108, "alcohol": 108, "air": 87}:
        raise ValueError(f"Unexpected {name} class counts: {counts}")
    return frame, columns


def encode_labels(values: pd.Series | np.ndarray) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(protocol.THREE_CLASSES)}
    return np.asarray([mapping[str(value)] for value in values], dtype=np.int64)


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class Representation(nn.Module):
    def __init__(self, input_dim: int, kind: str) -> None:
        super().__init__()
        self.kind = kind
        if kind in {"bottleneck16", "fusion16"}:
            self.network = nn.Sequential(
                nn.Linear(input_dim, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(protocol.NEURAL_DROPOUT),
                nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(),
            )
            self.output_dim = 16
        elif kind == "residual-d":
            self.network = nn.Linear(input_dim, input_dim)
            nn.init.zeros_(self.network.weight); nn.init.zeros_(self.network.bias)
            self.output_dim = input_dim
        else:
            raise ValueError(kind)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.kind == "residual-d":
            return values + self.network(values)
        return self.network(values)


class GasHead(nn.Module):
    def __init__(self, input_dim: int, kind: str, classes: int) -> None:
        super().__init__()
        if kind == "linear":
            self.network = nn.Linear(input_dim, classes)
        elif kind == "mlp":
            self.network = nn.Sequential(
                nn.Linear(input_dim, 16), nn.ReLU(), nn.Dropout(protocol.NEURAL_DROPOUT), nn.Linear(16, classes)
            )
        else:
            raise ValueError(kind)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class TabularDAModel(nn.Module):
    def __init__(self, input_dim: int, representation: str, head: str) -> None:
        super().__init__()
        self.representation_kind = representation
        self.representation = Representation(input_dim, representation)
        class_dim = input_dim + 16 if representation == "fusion16" else self.representation.output_dim
        self.gas_head = GasHead(class_dim, head, len(protocol.THREE_CLASSES))
        self.dann_discriminator = osc.DomainDiscriminator(self.representation.output_dim, 32)
        self.cdan_discriminator = osc.DomainDiscriminator(
            self.representation.output_dim * len(protocol.THREE_CLASSES), 32
        )

    def features(self, values: torch.Tensor) -> torch.Tensor:
        return self.representation(values)

    def class_input(self, values: torch.Tensor, feature: torch.Tensor | None = None) -> torch.Tensor:
        feature = self.features(values) if feature is None else feature
        return torch.cat([values, feature], dim=1) if self.representation_kind == "fusion16" else feature

    def logits(self, values: torch.Tensor, feature: torch.Tensor | None = None) -> torch.Tensor:
        return self.gas_head(self.class_input(values, feature))


@dataclass(frozen=True)
class TrainingSpec:
    feature_set: str
    split: str
    mode: str
    method: str
    method_family: str
    representation: str
    head: str
    seed: int


def mode_for(split_name: str) -> str:
    return "uda" if int(split_name[1]) <= 3 else "semi"


def has_method(method: str, component: str) -> bool:
    mappings = {
        "dann": {"dann", "c-dann", "cdan+c-dann", "mk-mmd+c-dann", "mk-mmd+cdan+c-dann"},
        "cdan": {
            "cdan", "cdan+c-dann", "mk-mmd+cdan", "mk-mmd+lc+cdan", "mk-mmd+cdan+c-dann"
        },
        "mmd": {
            "mk-mmd", "mk-mmd+lc", "mk-mmd+cdan", "mk-mmd+lc+cdan",
            "mk-mmd+c-dann", "mk-mmd+cdan+c-dann",
        },
        "centroid": {
            "c-dann", "cdan+c-dann", "mk-mmd+lc", "mk-mmd+lc+cdan",
            "mk-mmd+c-dann", "mk-mmd+cdan+c-dann",
        },
        "coral": {"deep-coral"},
    }
    return method in mappings[component]


def balanced_domain_batch(source: torch.Tensor, source_y: torch.Tensor,
                          target: torch.Tensor, epoch: int, seed: int):
    count = min(len(source), len(target))
    if count < 2:
        raise ValueError("Domain alignment requires at least two source and target samples.")
    generator = torch.Generator().manual_seed(seed * 1000003 + epoch)
    source_index = torch.randperm(len(source), generator=generator)[:count]
    target_index = torch.randperm(len(target), generator=generator)[:count]
    return source[source_index], source_y[source_index], target[target_index]


def predict(model: nn.Module, values: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(values, dtype=torch.float32)
        return model.logits(tensor).argmax(1).cpu().numpy()


def train_network(arrays: dict, spec: TrainingSpec, epochs: int, include_validation: bool,
                  selection_x: np.ndarray | None, selection_y: np.ndarray | None,
                  curve_phase: str) -> tuple[TabularDAModel, int, list[dict]]:
    set_seed(spec.seed)
    input_dim = arrays["x"]["source_train"].shape[1]
    model = TabularDAModel(input_dim, spec.representation, spec.head)
    dann_loss = osc.DomainAdversarialLoss(model.dann_discriminator, max_iters=epochs)
    cdan_loss = osc.ConditionalDomainAdversarialLoss(model.cdan_discriminator, max_iters=epochs)
    mmd_loss = osc.MultipleKernelMaximumMeanDiscrepancy([
        osc.GaussianKernel(0.5), osc.GaussianKernel(1.0), osc.GaussianKernel(2.0)
    ])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=protocol.NEURAL_LEARNING_RATE, weight_decay=protocol.NEURAL_WEIGHT_DECAY
    )
    x_cls, y_cls, _ = existing_suite.classifier_data(arrays, spec.mode, include_validation)
    x_cls_t = torch.as_tensor(x_cls, dtype=torch.float32)
    y_cls_t = torch.as_tensor(y_cls, dtype=torch.long)
    counts = np.bincount(y_cls, minlength=len(protocol.THREE_CLASSES)).astype(float)
    if np.any(counts == 0):
        raise ValueError(f"Classifier training data are missing classes: {counts.tolist()}")
    class_weight = torch.as_tensor(len(y_cls) / (len(counts) * counts), dtype=torch.float32)
    source_x = torch.as_tensor(arrays["x"]["source_train"], dtype=torch.float32)
    source_y = torch.as_tensor(arrays["y"]["source_train"], dtype=torch.long)
    target_x = torch.as_tensor(arrays["x"]["target_domain"], dtype=torch.float32)
    best_state, best_epoch = copy.deepcopy(model.state_dict()), epochs
    best_score = (-1.0, -1.0, -1.0)
    history = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad()
        classification_loss = F.cross_entropy(model.logits(x_cls_t), y_cls_t, weight=class_weight)
        total = classification_loss
        losses = {"classification_loss": float(classification_loss.detach()), "dann_loss": 0.0,
                  "cdan_loss": 0.0, "coral_loss": 0.0, "mmd_loss": 0.0, "centroid_loss": 0.0}
        if spec.method != "source-only":
            source_batch, source_batch_y, target_batch = balanced_domain_batch(
                source_x, source_y, target_x, epoch, spec.seed
            )
            source_h, target_h = model.features(source_batch), model.features(target_batch)
            source_logits = model.logits(source_batch, source_h)
            target_logits = model.logits(target_batch, target_h)
        if has_method(spec.method, "dann"):
            value = dann_loss(source_h, target_h); total = total + value
            losses["dann_loss"] = float(value.detach())
        if has_method(spec.method, "cdan"):
            value = cdan_loss(source_logits, source_h, target_logits, target_h); total = total + value
            losses["cdan_loss"] = float(value.detach())
        if has_method(spec.method, "coral"):
            value = osc.deep_coral_loss(source_h, target_h); total = total + CORAL_WEIGHT * value
            losses["coral_loss"] = float(value.detach())
        if has_method(spec.method, "mmd"):
            value = mmd_loss(source_h, target_h); total = total + MMD_WEIGHT * value
            losses["mmd_loss"] = float(value.detach())
        if has_method(spec.method, "centroid"):
            value = osc.class_centroid_loss(
                source_h, source_batch_y, target_h, target_logits, len(protocol.THREE_CLASSES)
            )
            total = total + CENTROID_WEIGHT * value
            losses["centroid_loss"] = float(value.detach())
        total.backward(); optimizer.step()
        if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == epochs:
            validation = (
                metric_values(selection_y, predict(model, selection_x))
                if selection_x is not None else {"accuracy": np.nan, "balanced_accuracy": np.nan, "macro_f1": np.nan}
            )
            history.append({
                **asdict(spec), "phase": curve_phase, "epoch": epoch,
                **losses, "total_loss": float(total.detach()),
                "validation_accuracy": validation["accuracy"],
                "validation_balanced_accuracy": validation["balanced_accuracy"],
                "validation_macro_f1": validation["macro_f1"],
                "domain_balanced_accuracy": (
                    dann_loss.domain_discriminator_accuracy if has_method(spec.method, "dann")
                    else cdan_loss.domain_discriminator_accuracy if has_method(spec.method, "cdan") else np.nan
                ),
                "elapsed_seconds": time.time() - start,
            })
            if selection_x is not None:
                score = (validation["macro_f1"], validation["balanced_accuracy"], validation["accuracy"])
                if score > best_score:
                    best_score, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
    if selection_x is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, history


def final_arrays(parts: dict, columns: list[str], mode: str, do_refit: bool) -> tuple[dict, bool, str]:
    if not do_refit:
        return existing_suite.make_arrays(parts, columns), False, "selected checkpoint; no refit"
    if mode == "semi" and len(parts["target_val"]):
        arrays = existing_suite.make_arrays(parts, columns, ("source_train", "target_labeled_train", "target_val"))
        return arrays, True, "source+target labeled train+validation refit"
    if mode == "uda" and len(parts["source_val"]):
        arrays = existing_suite.make_arrays(parts, columns, ("source_train", "source_val"))
        return arrays, True, "source train+validation refit"
    return existing_suite.make_arrays(parts, columns), False, "no validation; fixed 400 epochs"


def checkpoint_path(spec: TrainingSpec) -> Path:
    return CHECKPOINTS / spec.feature_set / spec.split / spec.representation / spec.head / spec.method / f"seed_{spec.seed}.pt"


def run_da(frame: pd.DataFrame, columns: list[str], split: dict, spec: TrainingSpec,
           do_refit: bool) -> tuple[dict, list[dict], list[dict]]:
    parts = existing_suite.partitions(frame, split)
    initial = existing_suite.make_arrays(parts, columns)
    selection_x, selection_y, selection_usage = existing_suite.selection_data(initial, spec.mode)
    selected_model, selected_epoch, selection_history = train_network(
        initial, spec, EPOCHS, False, selection_x, selection_y, "selection"
    )
    if selection_x is not None and do_refit:
        arrays, include_validation, final_training = final_arrays(parts, columns, spec.mode, True)
        model, _, final_history = train_network(
            arrays, spec, selected_epoch, include_validation, None, None, "refit"
        )
    else:
        arrays, include_validation, final_training = final_arrays(parts, columns, spec.mode, False)
        model, final_history = selected_model, []
    x_train, y_train, label_usage = existing_suite.classifier_data(arrays, spec.mode, include_validation)
    train_pred = predict(model, x_train)
    test_x, test_y = arrays["x"]["target_test"], arrays["y"]["target_test"]
    test_pred = predict(model, test_x)
    train_metric, test_metric = metric_values(y_train, train_pred), metric_values(test_y, test_pred)
    checkpoint = checkpoint_path(spec); checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "spec": asdict(spec), "input_dim": len(columns),
                "selected_epoch": selected_epoch, "final_training": final_training}, checkpoint)
    run = {
        "family": "domain-adaptation", **asdict(spec), "model": f"{spec.method}:{spec.representation}:{spec.head}",
        "input_dimensions": len(columns), "selected_epoch": selected_epoch,
        "selection_usage": selection_usage, "label_usage": label_usage,
        "final_training": final_training, "checkpoint": str(checkpoint),
        "confusion_matrix": json.dumps(confusion_matrix(test_y, test_pred, labels=range(len(protocol.THREE_CLASSES))).tolist()),
        **{f"train_{key}": value for key, value in train_metric.items()},
        **{f"test_{key}": value for key, value in test_metric.items()},
    }
    predictions = []
    for position, (index, sample) in enumerate(parts["target_test"].iterrows()):
        predictions.append({
            **asdict(spec), "family": "domain-adaptation", "sample_index": int(index),
            "sample_id": sample["sample_id"], "true_label": protocol.THREE_CLASSES[int(test_y[position])],
            "predicted_label": protocol.THREE_CLASSES[int(test_pred[position])],
            "correct": bool(test_y[position] == test_pred[position]),
        })
    return run, predictions, selection_history + final_history


def run_ml(frame: pd.DataFrame, columns: list[str], split: dict, feature_set: str,
           model_slug: str, seed: int | None) -> tuple[dict, list[dict], object]:
    model_name = {"logistic-regression": "Logistic Regression", "mlp": "MLP"}[model_slug]
    train, val, test = frame.loc[split["train"](frame)], frame.loc[split["val"](frame)], frame.loc[split["test"](frame)]
    x_train, y_train = train[columns].to_numpy(float), encode_labels(train["label"].to_numpy())
    x_val, y_val = val[columns].to_numpy(float), encode_labels(val["label"].to_numpy())
    cache_key = (feature_set, split["split"], model_slug)
    if cache_key not in ML_PARAM_CACHE:
        if len(val):
            params, _ = mlbase.select_params(
                model_name, x_train, y_train, x_val, y_val, len(protocol.THREE_CLASSES)
            )
            ML_PARAM_CACHE[cache_key] = (params, "train+validation refit")
        else:
            ML_PARAM_CACHE[cache_key] = (mlbase.DEFAULT_PARAMS[model_name], "train only; fixed default")
    params, final_training = ML_PARAM_CACHE[cache_key]
    final = pd.concat([train, val], axis=0) if len(val) else train
    x_final, y_final = final[columns].to_numpy(float), encode_labels(final["label"].to_numpy())
    x_test, y_test = test[columns].to_numpy(float), encode_labels(test["label"].to_numpy())
    estimator = mlbase.fit_estimator(mlbase.make_estimator(model_name, params, seed), x_final, y_final)
    train_pred, test_pred = estimator.predict(x_final), estimator.predict(x_test)
    run = {
        "family": "classical-ml", "feature_set": feature_set, "split": split["split"], "mode": "ml",
        "method": model_slug, "method_family": "classical-ml", "representation": "raw",
        "head": model_slug, "seed": -1 if seed is None else seed, "model": model_slug,
        "input_dimensions": len(columns), "selected_epoch": np.nan,
        "selection_usage": "validation" if len(val) else "fixed default; no validation",
        "label_usage": "train labels; validation labels for selection/refit",
        "final_training": final_training, "checkpoint": "",
        "confusion_matrix": json.dumps(confusion_matrix(y_test, test_pred, labels=range(len(protocol.THREE_CLASSES))).tolist()),
        **{f"train_{key}": value for key, value in metric_values(y_final, train_pred).items()},
        **{f"test_{key}": value for key, value in metric_values(y_test, test_pred).items()},
    }
    predictions = [{
        "family": "classical-ml", "feature_set": feature_set, "split": split["split"], "mode": "ml",
        "method": model_slug, "method_family": "classical-ml", "representation": "raw", "head": model_slug,
        "seed": -1 if seed is None else seed, "sample_index": int(index), "sample_id": sample["sample_id"],
        "true_label": protocol.THREE_CLASSES[int(y_test[position])],
        "predicted_label": protocol.THREE_CLASSES[int(test_pred[position])],
        "correct": bool(y_test[position] == test_pred[position]),
    } for position, (index, sample) in enumerate(test.iterrows())]
    return run, predictions, estimator


def key_from(row: dict) -> tuple:
    return tuple(row.get(name) for name in (
        "family", "feature_set", "split", "method", "representation", "head", "seed"
    ))


def replace_key(records: list[dict], new_rows: list[dict], key: tuple) -> list[dict]:
    return [row for row in records if key_from(row) != key] + new_rows


def write_summary(runs: list[dict]) -> None:
    frame = pd.DataFrame(runs)
    if frame.empty:
        return
    group_columns = ["family", "feature_set", "split", "method", "method_family", "representation", "head"]
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=False):
        row = dict(zip(group_columns, keys)); row["seeds"] = ",".join(str(int(v)) for v in group["seed"])
        for metric in ("train_accuracy", "test_accuracy", "test_balanced_accuracy", "test_macro_f1"):
            values = pd.to_numeric(group[metric], errors="raise")
            row[metric] = float(values.mean()); row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    atomic_csv(summary, SUMMARY_PATH)
    best = (
        summary.sort_values(
            ["feature_set", "split", "test_balanced_accuracy", "test_macro_f1", "test_accuracy"],
            ascending=[True, True, False, False, False],
        )
        .groupby(["feature_set", "split"], as_index=False)
        .first()
    )
    best.insert(2, "interpretation", "descriptive test ranking only; not used for model selection")
    atomic_csv(best, BEST_PATH)


def persist(runs: list[dict], predictions: list[dict], curves: list[dict], status: dict) -> None:
    atomic_csv(pd.DataFrame(runs), RUNS_PATH)
    atomic_csv(pd.DataFrame(predictions), PREDICTIONS_PATH)
    if curves:
        atomic_csv(pd.DataFrame(curves), CURVES_PATH)
    write_summary(runs); atomic_json(status, STATUS_PATH)


def main() -> None:
    args = parse_args(); args.splits = resolve_splits(args.splits)
    OUTPUT.mkdir(parents=True, exist_ok=True); CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    split_da = {item["split"]: item for item in protocol.DA_SPLITS}
    split_ml = {item["split"]: item for item in protocol.ML_SPLITS}
    runs, predictions, curves = load_records(RUNS_PATH), load_records(PREDICTIONS_PATH), load_records(CURVES_PATH)
    complete = {key_from(row) for row in runs}
    tasks = []
    for feature_set in args.feature_sets:
        for split_name in args.splits:
            for model in args.ml_models:
                seeds = mlbase.seeds_for_model({"logistic-regression": "Logistic Regression", "mlp": "MLP"}[model])
                tasks.extend(("ml", feature_set, split_name, model, "raw", model, -1 if seed is None else seed) for seed in seeds)
            for representation in args.representations:
                for head in args.heads:
                    for method in args.methods:
                        tasks.extend(("da", feature_set, split_name, method, representation, head, seed) for seed in SEEDS)
    protocol_payload = {
        "feature_sets": {name: str(path) for name, path in FEATURE_PATHS.items()},
        "classes": protocol.THREE_CLASSES, "splits": args.splits, "seeds": list(SEEDS),
        "epochs": EPOCHS, "evaluation_interval": EVAL_EVERY,
        "optimizer": "Adam", "learning_rate": protocol.NEURAL_LEARNING_RATE,
        "weight_decay": protocol.NEURAL_WEIGHT_DECAY,
        "opensource_core": list(CORE_METHODS), "project_extensions": list(EXTENSION_METHODS),
        "representations": list(args.representations), "heads": list(args.heads),
        "refit_after_validation_epoch_selection": not args.no_refit,
        "transductive_protocol": "target-domain test features may enter alignment without gas labels",
        "test_labels": "evaluation only; never used for selection or training",
        "upstream": {
            "TLlib": "https://github.com/thuml/Transfer-Learning-Library",
            "CORAL": "https://github.com/VisionLearningGroup/CORAL",
        },
        "hybrid_objectives": {
            "mk-mmd+cdan": "L_cls + L_mkmmd + L_cdan",
            "mk-mmd+c-dann": "L_cls + L_mkmmd + L_dann + 0.1*L_centroid",
            "mk-mmd+cdan+c-dann": "L_cls + L_mkmmd + L_cdan + L_dann + 0.1*L_centroid",
        },
    }
    atomic_json(protocol_payload, PROTOCOL_PATH)
    status = {"state": "running", "total_tasks": len(tasks), "completed_before_start": len(complete),
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "current": None}
    atomic_json(status, STATUS_PATH)
    frames = {}
    try:
        for number, task in enumerate(tasks, start=1):
            family, feature_set, split_name, method, representation, head, seed = task
            key = ("classical-ml" if family == "ml" else "domain-adaptation",
                   feature_set, split_name, method, representation, head, seed)
            if key in complete and not args.force:
                continue
            if feature_set not in frames:
                frames[feature_set] = load_feature_set(feature_set)
            frame, columns = frames[feature_set]
            status.update({"current": list(key), "task_number": number, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            atomic_json(status, STATUS_PATH)
            print(f"[{number}/{len(tasks)}] {' | '.join(map(str, key))}", flush=True)
            if family == "ml":
                run, pred, estimator = run_ml(frame, columns, split_ml[split_name], feature_set, method,
                                               None if seed == -1 else seed)
                checkpoint = CHECKPOINTS / feature_set / split_name / "classical-ml" / method / f"seed_{seed}.joblib"
                checkpoint.parent.mkdir(parents=True, exist_ok=True); joblib.dump(estimator, checkpoint)
                run["checkpoint"] = str(checkpoint); new_curves = []
            else:
                spec = TrainingSpec(
                    feature_set, split_name, mode_for(split_name), method,
                    "opensource-core" if method in CORE_METHODS else "project-extension",
                    representation, head, seed,
                )
                run, pred, new_curves = run_da(frame, columns, split_da[split_name], spec, not args.no_refit)
            runs = replace_key(runs, [run], key)
            predictions = replace_key(predictions, pred, key)
            curves = replace_key(curves, new_curves, key)
            complete.add(key)
            status["completed_tasks"] = len(complete); status["current"] = None
            persist(runs, predictions, curves, status)
        status.update({"state": "complete", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "current": None})
        persist(runs, predictions, curves, status)
    except BaseException as error:
        status.update({"state": "interrupted", "error": repr(error),
                       "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        persist(runs, predictions, curves, status)
        raise


if __name__ == "__main__":
    main()
