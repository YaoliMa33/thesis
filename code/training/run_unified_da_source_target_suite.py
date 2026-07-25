"""Unified source-only, UDA, and semi-supervised DA experiments.

The established experiment files are not modified. This runner provides the
same two PyTorch backbones for every method and writes a new result family.

Label policy
------------
* UDA: gas classification uses source labels only. Target features are used by
  the alignment loss without target gas labels.
* Semi: classification uses source plus the target labels explicitly assigned
  to training by the split. After validation-only selection, target validation
  labels are included in a fresh Source+Target refit.
* Test gas labels are evaluation-only.

Algorithm provenance
--------------------
* DANN, C-DANN, CDAN, and CDAN+C-DANN follow the existing project modules.
* Deep CORAL uses the covariance loss from VisionLearningGroup/CORAL.
* MK-MMD follows THU Transfer-Learning-Library's DAN Gaussian-kernel design;
  the quadratic estimator below supports unequal source/target sample counts.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_cdan_multi_splits as existing_cdan
import run_cdan_hybrid_report as existing_hybrid
import run_dann_multi_splits as existing_dann
import run_deep_coral_multi_splits as existing_coral
import shared_experiment_protocol as protocol


ROOT = Path(r"D:\thesis")
FEATURE_PATH = ROOT / "tables" / "manifest_baseline_as_air_features.csv"
OUTPUT_DIR = ROOT / "tables" / "unified_da_source_target"
CLASSES = list(protocol.THREE_CLASSES)
SEEDS = list(protocol.MODEL_SEEDS)
EPOCHS = protocol.NEURAL_EPOCHS
EVAL_EVERY = 10

ARCHITECTURES = {
    "one-layer": (16,),
    "two-layer": (32, 16),
}
METHODS = [
    "source-only", "dann", "deep-coral", "mk-mmd", "c-dann", "cdan", "cdan+c-dann",
    "mk-mmd+lc", "mk-mmd+cdan", "mk-mmd+lc+cdan",
]
MODES = ["uda", "semi"]

DANN_LAMBDAS = tuple(existing_dann.LAMBDA_VALUES)
CORAL_LAMBDAS = (0.001, 0.01, 0.1, 1.0)
MMD_LAMBDAS = (0.01, 0.1, 1.0)
CENTROID_ALPHAS = tuple(existing_hybrid.ALIGN_STRENGTHS)
DEFAULT_LAMBDA = 0.2
DEFAULT_CORAL_LAMBDA = 0.1
DEFAULT_MMD_LAMBDA = 0.1
DEFAULT_ALPHA = existing_hybrid.DEFAULT_ALPHA
MMD_KERNEL_ALPHAS = (0.5, 1.0, 2.0)

PROVENANCE = {
    "dann": "existing run_dann_multi_splits.py",
    "deep-coral": "existing run_deep_coral_multi_splits.py; VisionLearningGroup/CORAL",
    "mk-mmd": "THU Transfer-Learning-Library tllib/alignment/dan.py and tllib/modules/kernels.py",
    "c-dann": "existing class-conditional centroid alignment plus DANN domain loss",
    "cdan": "existing run_cdan_multi_splits.py official-style multilinear condition T(h,g)",
    "cdan+c-dann": "existing run_cdan_hybrid_report.py",
    "mk-mmd+lc": "THU DAN MK-MMD plus existing class-conditional latent centroid loss Lc",
    "mk-mmd+cdan": "THU DAN MK-MMD plus existing official-style CDAN conditional adversarial loss",
    "mk-mmd+lc+cdan": "THU DAN MK-MMD plus existing centroid Lc and CDAN conditional adversarial loss",
}


@dataclass(frozen=True)
class Candidate:
    alignment_strength: float = 0.0
    centroid_strength: float = 0.0
    adversarial_strength: float = 0.0


class FeatureExtractor(nn.Module):
    def __init__(self, n_features: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = n_features
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.ReLU(), nn.Dropout(protocol.NEURAL_DROPOUT)])
            previous = width
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UnifiedModel(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        latent_dim = hidden_dims[-1]
        if latent_dim != 16:
            raise ValueError("Existing domain heads require a 16-dimensional latent representation.")
        self.feature_extractor = FeatureExtractor(n_features, hidden_dims)
        self.label_classifier = nn.Linear(latent_dim, n_classes)
        self.dann_domain_classifier = nn.Sequential(nn.Linear(latent_dim, 16), nn.ReLU(), nn.Linear(16, 2))
        self.cdan_domain_classifier = existing_cdan.CDANDomainDiscriminator(latent_dim, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def class_logits_from_h(self, h: torch.Tensor) -> torch.Tensor:
        return self.label_classifier(h)

    def class_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.class_logits_from_h(self.features(x))

    def dann_domain_logits(self, h: torch.Tensor, strength: float, reverse: bool = True) -> torch.Tensor:
        value = existing_dann.grad_reverse(h, strength) if reverse else h
        return self.dann_domain_classifier(value)

    def cdan_domain_logits(self, h: torch.Tensor, strength: float, reverse: bool = True) -> torch.Tensor:
        gas_logits = self.class_logits_from_h(h)
        conditioned = existing_cdan.multilinear_condition(h, gas_logits, detach_predictions=True)
        value = existing_dann.grad_reverse(conditioned, strength) if reverse else conditioned
        return self.cdan_domain_classifier(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--architectures", nargs="+", choices=list(ARCHITECTURES), default=list(ARCHITECTURES))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--splits", nargs="+", default=[item["split"] for item in protocol.DA_SPLITS])
    return parser.parse_args()


def feature_columns(df: pd.DataFrame) -> list[str]:
    columns = existing_dann.feature_columns(df)
    if len(columns) != 54:
        raise ValueError(f"Expected 54 features, found {len(columns)}.")
    if not np.isfinite(df[columns].to_numpy(dtype=float)).all():
        raise ValueError("Feature table contains NaN or infinite values.")
    return columns


def frame_for(df: pd.DataFrame, fn) -> pd.DataFrame:
    mask = fn(df)
    if not isinstance(mask, pd.Series) or len(mask) != len(df):
        raise ValueError("Invalid split mask.")
    return df.loc[mask].copy()


def partitions(df: pd.DataFrame, split: dict) -> dict[str, pd.DataFrame]:
    parts = {key: frame_for(df, split[key]) for key in [
        "source_train", "source_val", "target_labeled_train", "target_domain", "target_val", "target_test"
    ]}
    if parts["source_train"].empty or parts["target_test"].empty or parts["target_domain"].empty:
        raise ValueError(f"{split['split']} has an empty required partition.")
    labeled_keys = ["source_train", "source_val", "target_labeled_train", "target_val", "target_test"]
    for i, left in enumerate(labeled_keys):
        for right in labeled_keys[i + 1:]:
            overlap = set(parts[left].index) & set(parts[right].index)
            if overlap:
                raise ValueError(f"{split['split']} overlaps {left} and {right} by {len(overlap)} rows.")
    return parts


def encode(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = frame[columns].to_numpy(dtype=np.float32)
    y = existing_dann.encode(frame["label"].astype(str).to_numpy(), CLASSES)
    return x, y


def fit_standardizer(x_reference: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mean = x_reference.mean(axis=0)
    scale = x_reference.std(axis=0, ddof=0)
    if np.any(scale < 1e-9):
        raise ValueError("Training reference contains near-constant feature columns.")
    return {key: ((value - mean) / scale).astype(np.float32) for key, value in arrays.items()}


def make_arrays(
    parts: dict[str, pd.DataFrame],
    columns: list[str],
    standardization_reference: tuple[str, ...] = ("source_train",),
) -> dict:
    x, y = {}, {}
    for key, frame in parts.items():
        x[key], y[key] = encode(frame, columns)
    reference = concatenate([x[key] for key in standardization_reference], len(columns))
    standardized = fit_standardizer(reference, x)
    return {"x": standardized, "y": y}


def concatenate(items: list[np.ndarray], width: int | None = None) -> np.ndarray:
    nonempty = [item for item in items if len(item)]
    if nonempty:
        return np.concatenate(nonempty, axis=0)
    return np.empty((0, width), dtype=np.float32) if width is not None else np.empty((0,), dtype=np.int64)


def classifier_data(arrays: dict, mode: str, include_validation: bool) -> tuple[np.ndarray, np.ndarray, str]:
    x_items = [arrays["x"]["source_train"]]
    y_items = [arrays["y"]["source_train"]]
    usage = "source labels only"
    if mode == "semi":
        x_items.append(arrays["x"]["target_labeled_train"])
        y_items.append(arrays["y"]["target_labeled_train"])
        if include_validation:
            x_items.append(arrays["x"]["target_val"])
            y_items.append(arrays["y"]["target_val"])
        usage = "source + labeled target"
    elif include_validation and len(arrays["x"]["source_val"]):
        x_items.append(arrays["x"]["source_val"])
        y_items.append(arrays["y"]["source_val"])
        usage = "source train + source validation labels"
    return concatenate(x_items, arrays["x"]["source_train"].shape[1]), concatenate(y_items), usage


def selection_data(arrays: dict, mode: str) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if mode == "uda":
        if len(arrays["x"]["source_val"]):
            return arrays["x"]["source_val"], arrays["y"]["source_val"], "source validation labels"
        return None, None, "fixed hyperparameters; target labels prohibited for UDA selection"
    if len(arrays["x"]["target_val"]):
        return arrays["x"]["target_val"], arrays["y"]["target_val"], "target validation labels"
    return None, None, "fixed hyperparameters; no labeled target validation"


def method_candidates(
    method: str,
    has_validation: bool,
    architecture: str,
    split: str,
    mode: str,
) -> list[Candidate]:
    if method == "source-only":
        return [Candidate()]
    if method in {"mk-mmd+lc", "mk-mmd+cdan", "mk-mmd+lc+cdan"}:
        if not has_validation:
            return [Candidate(DEFAULT_MMD_LAMBDA, DEFAULT_ALPHA if "+lc" in method else 0.0, DEFAULT_LAMBDA if "+cdan" in method else 0.0)]
        if method == "mk-mmd+lc":
            return [Candidate(mmd_strength, alpha, 0.0) for mmd_strength in MMD_LAMBDAS for alpha in CENTROID_ALPHAS]
        if method == "mk-mmd+cdan":
            return [Candidate(mmd_strength, 0.0, adversarial) for mmd_strength in MMD_LAMBDAS for adversarial in DANN_LAMBDAS]
        return [
            Candidate(mmd_strength, alpha, adversarial)
            for mmd_strength in MMD_LAMBDAS
            for alpha in CENTROID_ALPHAS
            for adversarial in DANN_LAMBDAS
        ]
    if not has_validation:
        default = DEFAULT_CORAL_LAMBDA if method == "deep-coral" else DEFAULT_MMD_LAMBDA if method == "mk-mmd" else DEFAULT_LAMBDA
        return [Candidate(default, DEFAULT_ALPHA if method in {"c-dann", "cdan+c-dann"} else 0.0)]
    if method == "deep-coral":
        return [Candidate(value) for value in CORAL_LAMBDAS]
    if method == "mk-mmd":
        return [Candidate(value) for value in MMD_LAMBDAS]
    if method in {"c-dann", "cdan+c-dann"}:
        return [Candidate(lambd, alpha) for lambd in DANN_LAMBDAS for alpha in CENTROID_ALPHAS]
    return [Candidate(value) for value in DANN_LAMBDAS]


def covariance(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("Deep CORAL requires at least two 2D feature samples per domain.")
    centered = features - features.mean(dim=0, keepdim=True)
    return centered.T @ centered / (features.shape[0] - 1)


def coral_loss(source_h: torch.Tensor, target_h: torch.Tensor) -> torch.Tensor:
    d = source_h.shape[1]
    return (covariance(source_h) - covariance(target_h)).pow(2).sum() / (4.0 * d * d)


def gaussian_kernel(x: torch.Tensor, alpha: float) -> torch.Tensor:
    squared = ((x.unsqueeze(0) - x.unsqueeze(1)) ** 2).sum(dim=2)
    sigma_square = alpha * squared.detach().mean()
    if not bool(torch.isfinite(sigma_square)) or float(sigma_square) <= 0:
        raise ValueError("MK-MMD Gaussian bandwidth is non-positive or non-finite.")
    return torch.exp(-squared / (2.0 * sigma_square))


def mkmmd_loss(source_h: torch.Tensor, target_h: torch.Tensor) -> torch.Tensor:
    combined = torch.cat([source_h, target_h], dim=0)
    kernel = sum(gaussian_kernel(combined, alpha) for alpha in MMD_KERNEL_ALPHAS)
    n_source = source_h.shape[0]
    k_ss = kernel[:n_source, :n_source]
    k_tt = kernel[n_source:, n_source:]
    k_st = kernel[:n_source, n_source:]
    return k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean()


def centroid_loss(
    model: UnifiedModel,
    source_h: torch.Tensor,
    y_source: np.ndarray,
    target_h: torch.Tensor,
    target_y: np.ndarray | None,
) -> torch.Tensor:
    source_weights = F.one_hot(existing_dann.to_long(y_source), num_classes=len(CLASSES)).float()
    if target_y is None:
        with torch.no_grad():
            target_weights = F.softmax(model.class_logits_from_h(target_h), dim=1)
    else:
        target_weights = F.one_hot(existing_dann.to_long(target_y), num_classes=len(CLASSES)).float()
    source_mean = existing_hybrid.weighted_mean(source_h, source_weights)
    target_mean = existing_hybrid.weighted_mean(target_h, target_weights)
    return (source_mean - target_mean).pow(2).mean()


def predictions(model: UnifiedModel, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.class_logits(existing_dann.to_tensor(x)).argmax(dim=1).cpu().numpy()


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return existing_dann.metric_dict(y, pred, len(CLASSES))


def domain_metrics(model: UnifiedModel, method: str, source_h: torch.Tensor, target_h: torch.Tensor) -> dict[str, float]:
    cdan_methods = {"cdan", "mk-mmd+cdan", "mk-mmd+lc+cdan"}
    if method not in {"dann", "c-dann", "cdan", "cdan+c-dann", "mk-mmd+cdan", "mk-mmd+lc+cdan"}:
        return {"domain_accuracy": np.nan, "domain_balanced_accuracy": np.nan, "domain_source_recall": np.nan, "domain_target_recall": np.nan}
    with torch.no_grad():
        h = torch.cat([source_h, target_h], dim=0)
        logits = model.cdan_domain_logits(h, 0.0, reverse=False) if method in cdan_methods else model.dann_domain_logits(h, 0.0, reverse=False)
        pred = logits.argmax(dim=1).cpu().numpy()
    source_recall = float((pred[:len(source_h)] == 0).mean())
    target_recall = float((pred[len(source_h):] == 1).mean())
    truth = np.concatenate([np.zeros(len(source_h), dtype=int), np.ones(len(target_h), dtype=int)])
    return {
        "domain_accuracy": float((pred == truth).mean()),
        "domain_balanced_accuracy": 0.5 * (source_recall + target_recall),
        "domain_source_recall": source_recall,
        "domain_target_recall": target_recall,
    }


def train_candidate(
    arrays: dict,
    hidden_dims: tuple[int, ...],
    method: str,
    mode: str,
    candidate: Candidate,
    seed: int,
    include_validation: bool,
    fixed_epochs: int | None,
) -> tuple[UnifiedModel, dict]:
    x_cls, y_cls, label_usage = classifier_data(arrays, mode, include_validation)
    x_select, y_select, selection_usage = selection_data(arrays, mode)
    if include_validation:
        x_select = y_select = None
    existing_dann.set_random_seed(seed)
    model = UnifiedModel(x_cls.shape[1], len(CLASSES), hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol.NEURAL_LEARNING_RATE, weight_decay=protocol.NEURAL_WEIGHT_DECAY)
    x_cls_t, y_cls_t = existing_dann.to_tensor(x_cls), existing_dann.to_long(y_cls)
    class_weight = existing_dann.class_weights(y_cls, len(CLASSES))
    x_source = arrays["x"]["source_train"]
    y_source = arrays["y"]["source_train"]
    x_target = arrays["x"]["target_domain"]
    x_source_t, x_target_t = existing_dann.to_tensor(x_source), existing_dann.to_tensor(x_target)
    y_domain = torch.cat([torch.zeros(len(x_source), dtype=torch.long), torch.ones(len(x_target), dtype=torch.long)])

    if mode == "semi" and len(arrays["x"]["target_labeled_train"]) + len(arrays["x"]["target_val"]) == 0:
        raise ValueError("Semi mode requires labeled target training or validation data.")

    best_state = copy.deepcopy(model.state_dict())
    best_score = (-1.0, -1.0, -1.0)
    best_epoch = 0
    epochs_to_run = int(fixed_epochs or EPOCHS)
    last = {}
    for epoch in range(1, epochs_to_run + 1):
        model.train()
        optimizer.zero_grad()
        class_loss = F.cross_entropy(model.class_logits(x_cls_t), y_cls_t, weight=class_weight)
        source_h, target_h = model.features(x_source_t), model.features(x_target_t)
        alignment = torch.tensor(0.0)
        centroid = torch.tensor(0.0)
        total = class_loss

        if method in {"dann", "c-dann", "cdan+c-dann"}:
            domain_h = torch.cat([source_h, target_h], dim=0)
            domain_loss = F.cross_entropy(model.dann_domain_logits(domain_h, candidate.alignment_strength), y_domain)
            total = total + domain_loss
            alignment = domain_loss
        if method in {"cdan", "cdan+c-dann", "mk-mmd+cdan", "mk-mmd+lc+cdan"}:
            domain_h = torch.cat([source_h, target_h], dim=0)
            adversarial_strength = candidate.adversarial_strength or candidate.alignment_strength
            cdan_loss = F.cross_entropy(model.cdan_domain_logits(domain_h, adversarial_strength), y_domain)
            total = total + cdan_loss
            alignment = alignment + cdan_loss
        if method == "deep-coral":
            alignment = coral_loss(source_h, target_h)
            total = total + candidate.alignment_strength * alignment
        if method in {"mk-mmd", "mk-mmd+lc", "mk-mmd+cdan", "mk-mmd+lc+cdan"}:
            mmd_value = mkmmd_loss(source_h, target_h)
            total = total + candidate.alignment_strength * mmd_value
            alignment = alignment + mmd_value
        if method in {"c-dann", "cdan+c-dann", "mk-mmd+lc", "mk-mmd+lc+cdan"}:
            if mode == "semi":
                target_keys = ["target_labeled_train"] + (["target_val"] if include_validation else [])
                target_x = concatenate([arrays["x"][key] for key in target_keys], x_cls.shape[1])
                target_y = concatenate([arrays["y"][key] for key in target_keys])
                if len(target_x) == 0:
                    target_x, target_y = x_target, None
                target_centroid_h = model.features(existing_dann.to_tensor(target_x))
                centroid = centroid_loss(model, source_h, y_source, target_centroid_h, target_y)
            else:
                centroid = centroid_loss(model, source_h, y_source, target_h, None)
            total = total + candidate.centroid_strength * centroid

        total.backward()
        optimizer.step()
        last = {"L": float(class_loss.detach()), "alignment_loss": float(alignment.detach()), "centroid_loss": float(centroid.detach()), "total_loss": float(total.detach())}
        if x_select is not None and (epoch % EVAL_EVERY == 0 or epoch == epochs_to_run):
            selected = metric(y_select, predictions(model, x_select))
            score = (selected["macro_f1"], selected["balanced_accuracy"], selected["accuracy"])
            if score > best_score:
                best_score, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        elif x_select is None and epoch == epochs_to_run:
            best_epoch, best_state = epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, {**last, "best_epoch": best_epoch, "label_usage": label_usage, "selection_usage": selection_usage}


def selected_candidate(rows: pd.DataFrame) -> Candidate:
    grouped = rows.groupby(["alignment_strength", "centroid_strength", "adversarial_strength"], as_index=False)[
        ["validation_macro_f1", "validation_balanced_accuracy", "validation_accuracy"]
    ].mean()
    chosen = grouped.sort_values(
        ["validation_macro_f1", "validation_balanced_accuracy", "validation_accuracy", "alignment_strength", "centroid_strength", "adversarial_strength"],
        ascending=[False, False, False, True, True, True], kind="mergesort"
    ).iloc[0]
    return Candidate(float(chosen.alignment_strength), float(chosen.centroid_strength), float(chosen.adversarial_strength))


def evaluate_model(model: UnifiedModel, arrays: dict, method: str, y_train: np.ndarray, x_train: np.ndarray) -> dict:
    train_m = metric(y_train, predictions(model, x_train))
    test_m = metric(arrays["y"]["target_test"], predictions(model, arrays["x"]["target_test"]))
    model.eval()
    with torch.no_grad():
        source_h = model.features(existing_dann.to_tensor(arrays["x"]["source_train"]))
        target_h = model.features(existing_dann.to_tensor(arrays["x"]["target_domain"]))
    return {
        **{f"train_{key}": value for key, value in train_m.items()},
        **{f"test_{key}": value for key, value in test_m.items()},
        **domain_metrics(model, method, source_h, target_h),
    }


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["architecture", "split", "variant"]
    metrics_to_summarize = [column for column in runs.columns if column.startswith(("train_", "test_", "domain_")) and pd.api.types.is_numeric_dtype(runs[column])]
    for key, group in runs.groupby(keys, sort=False):
        first = group.iloc[0]
        row = {"architecture": key[0], "split": key[1], "variant": key[2]}
        for column in ["description", "method", "mode", "label_usage", "selection_usage", "final_training", "alignment_strength", "centroid_strength", "adversarial_strength", "algorithm_provenance"]:
            row[column] = first[column]
        row["seeds"] = ",".join(map(str, sorted(group.seed.astype(int).unique())))
        for column in metrics_to_summarize + ["best_epoch"]:
            values = group[column].astype(float)
            row[column] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1))
            row[f"{column}_min"] = float(values.min())
            row[f"{column}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def run_experiment(df: pd.DataFrame, columns: list[str], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows, prediction_rows = [], []
    split_map = {item["split"]: item for item in protocol.DA_SPLITS}
    unknown = set(args.splits) - set(split_map)
    if unknown:
        raise ValueError(f"Unknown splits: {sorted(unknown)}")
    for architecture in args.architectures:
        hidden_dims = ARCHITECTURES[architecture]
        for split_name in args.splits:
            split = split_map[split_name]
            parts = partitions(df, split)
            arrays = make_arrays(parts, columns)
            for method in args.methods:
                modes = args.modes
                for mode in modes:
                    if mode == "semi" and len(parts["target_labeled_train"]) + len(parts["target_val"]) == 0:
                        continue
                    x_select, _, selection_usage = selection_data(arrays, mode)
                    candidates = method_candidates(
                        method,
                        x_select is not None,
                        architecture,
                        split_name,
                        mode,
                    )
                    tuning_rows = []
                    tuning_models = {}
                    for candidate in candidates:
                        for seed in SEEDS:
                            model, info = train_candidate(arrays, hidden_dims, method, mode, candidate, seed, False, None)
                            x_train, y_train, _ = classifier_data(arrays, mode, False)
                            train_metric = metric(y_train, predictions(model, x_train))
                            row = {
                                "architecture": architecture, "split": split_name, "description": split["description"],
                                "method": method, "mode": mode, "seed": seed,
                                "alignment_strength": candidate.alignment_strength, "centroid_strength": candidate.centroid_strength,
                                "adversarial_strength": candidate.adversarial_strength,
                                "best_epoch": info["best_epoch"], "label_usage": info["label_usage"], "selection_usage": selection_usage,
                                "validation_accuracy": np.nan, "validation_balanced_accuracy": np.nan, "validation_macro_f1": np.nan,
                                **{f"train_{key}": value for key, value in train_metric.items()},
                            }
                            if x_select is not None:
                                selected_m = metric(selection_data(arrays, mode)[1], predictions(model, x_select))
                                row.update({f"validation_{key}": value for key, value in selected_m.items()})
                            tuning_rows.append(row)
                            tuning_models[(candidate, seed)] = model
                    tuning_frame = pd.DataFrame(tuning_rows)
                    chosen = selected_candidate(tuning_frame) if x_select is not None else candidates[0]
                    for seed in SEEDS:
                        selected_row = tuning_frame[
                            np.isclose(tuning_frame.alignment_strength, chosen.alignment_strength)
                            & np.isclose(tuning_frame.centroid_strength, chosen.centroid_strength)
                            & np.isclose(tuning_frame.adversarial_strength, chosen.adversarial_strength)
                            & tuning_frame.seed.eq(seed)
                        ].iloc[0]
                        final_arrays = arrays
                        if mode == "semi":
                            final_arrays = make_arrays(
                                parts,
                                columns,
                                ("source_train", "target_labeled_train", "target_val"),
                            )
                            model, info = train_candidate(final_arrays, hidden_dims, method, mode, chosen, seed, True, int(selected_row.best_epoch))
                            final_training = "Source+Target refit"
                        elif len(arrays["x"]["source_val"]):
                            final_arrays = make_arrays(parts, columns, ("source_train", "source_val"))
                            model, info = train_candidate(final_arrays, hidden_dims, method, mode, chosen, seed, True, int(selected_row.best_epoch))
                            final_training = "source train+source validation refit"
                        else:
                            model = tuning_models[(chosen, seed)]
                            info = {"best_epoch": int(selected_row.best_epoch), "label_usage": selected_row.label_usage}
                            final_training = "source-only; no refit"
                        x_final, y_final, label_usage = classifier_data(final_arrays, mode, mode == "semi" or len(final_arrays["x"]["source_val"]) > 0)
                        evaluated = evaluate_model(model, final_arrays, method, y_final, x_final)
                        if method == "source-only":
                            variant = "MLP Source-only" if mode == "uda" else "MLP Source+Target"
                        else:
                            variant = f"{method.upper()}-{mode.upper()}"
                        final_row = {
                            "architecture": architecture, "split": split_name, "description": split["description"],
                            "variant": variant, "method": method, "mode": mode, "seed": seed,
                            "alignment_strength": chosen.alignment_strength, "centroid_strength": chosen.centroid_strength,
                            "adversarial_strength": chosen.adversarial_strength,
                            "best_epoch": info["best_epoch"], "label_usage": label_usage, "selection_usage": selection_usage,
                            "final_training": final_training, "algorithm_provenance": PROVENANCE.get(method, "shared PyTorch source-only baseline"),
                            **evaluated,
                        }
                        all_rows.append(final_row)
                        test_pred = predictions(model, final_arrays["x"]["target_test"])
                        for position, (index, sample) in enumerate(parts["target_test"].iterrows()):
                            prediction_rows.append({
                                "architecture": architecture, "split": split_name, "variant": variant, "seed": seed,
                                "sample_index": int(index), "sample_id": sample.get("sample_id", index),
                                "target_file": sample.get("source_file", sample.get("file", "")),
                                "period": sample.get("period", ""), "day_label": sample.get("day_label", ""), "batch": sample.get("batch", ""),
                                "true_label": CLASSES[int(final_arrays["y"]["target_test"][position])],
                                "predicted_label": CLASSES[int(test_pred[position])],
                                "correct": bool(final_arrays["y"]["target_test"][position] == test_pred[position]),
                            })
    return pd.DataFrame(all_rows), pd.DataFrame(prediction_rows)


def write_protocol(args: argparse.Namespace) -> None:
    payload = {
        "features": 54, "classes": CLASSES, "seeds": SEEDS, "epochs": EPOCHS,
        "architectures": {key: list(value) for key, value in ARCHITECTURES.items()},
        "requested_methods": args.methods, "requested_modes": args.modes, "requested_splits": args.splits,
        "label_policy": {
            "uda": "source gas labels only; target gas labels prohibited from training and model selection",
            "semi": "source plus explicitly labeled target training data; target validation joins Source+Target refit",
            "test": "gas labels evaluation-only",
        },
        "selection": "validation macro-F1, balanced accuracy, accuracy; fixed defaults when admissible validation is absent",
        "mmd": {"kernel": "multi-kernel Gaussian", "alphas": MMD_KERNEL_ALPHAS, "estimator": "quadratic unequal-size empirical MMD"},
        "hybrid_strength_columns": {
            "alignment_strength": "lambda_mmd for MK-MMD hybrids",
            "centroid_strength": "alpha multiplying class-conditional centroid loss Lc",
            "adversarial_strength": "lambda_cdan applied through the CDAN gradient reversal branch",
        },
        "hybrid_candidate_grids": {
            "lambda_mmd": MMD_LAMBDAS,
            "alpha_lc": CENTROID_ALPHAS,
            "lambda_cdan": DANN_LAMBDAS,
        },
        "provenance": PROVENANCE,
    }
    (OUTPUT_DIR / "protocol.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURE_PATH)
    columns = feature_columns(df)
    runs, predictions_frame = run_experiment(df, columns, args)
    if runs.empty:
        raise RuntimeError("No experiment rows were produced for the requested filters.")
    summary = summarize(runs)
    suffix = "_".join(args.architectures) + "__" + "_".join(args.methods) + "__" + "_".join(args.modes)
    runs.to_csv(OUTPUT_DIR / f"runs__{suffix}.csv", index=False)
    summary.to_csv(OUTPUT_DIR / f"summary__{suffix}.csv", index=False)
    predictions_frame.to_csv(OUTPUT_DIR / f"predictions__{suffix}.csv", index=False)
    write_protocol(args)
    print(summary[["architecture", "split", "variant", "test_accuracy", "test_accuracy_std"]].to_string(index=False))
    print(f"Outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
