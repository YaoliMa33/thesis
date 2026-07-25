from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import shared_experiment_protocol as protocol


OUT_ROOT = Path(r"D:\thesis")
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures" / "dann_learning"
FEATURE_PATH = TABLE_DIR / "manifest_baseline_as_air_features.csv"

TASKS = [
    ("three_class_with_air", protocol.THREE_CLASSES),
]
REPORT_TASKS = [
    ("three_class_with_air", protocol.THREE_CLASSES),
]

LAMBDA_VALUES = [0.05, 0.2, 0.5]
SEEDS = list(protocol.MODEL_SEEDS)
DEFAULT_LAMBDA = 0.2
DEFAULT_SEED = SEEDS[0]
NO_VALIDATION_SELECTION = protocol.NO_VALIDATION_SELECTION
EPOCHS = protocol.NEURAL_EPOCHS
LEARNING_RATE = protocol.NEURAL_LEARNING_RATE
WEIGHT_DECAY = protocol.NEURAL_WEIGHT_DECAY


class GradientReverse(torch.autograd.Function):
    """Gradient reversal layer used by standard DANN/RevGrad implementations."""
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReverse.apply(x, lambd)


class FeatureExtractor(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, protocol.NEURAL_HIDDEN_DIMS[0]),
            nn.ReLU(),
            nn.Dropout(protocol.NEURAL_DROPOUT),
            nn.Linear(protocol.NEURAL_HIDDEN_DIMS[0], protocol.NEURAL_HIDDEN_DIMS[1]),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LabelClassifier(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Linear(protocol.NEURAL_HIDDEN_DIMS[1], n_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class DomainClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(protocol.NEURAL_HIDDEN_DIMS[1], protocol.NEURAL_HIDDEN_DIMS[1]),
            nn.ReLU(),
            nn.Linear(protocol.NEURAL_HIDDEN_DIMS[1], 2),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class DANN(nn.Module):
    """DANN structure: feature extractor + label classifier + domain classifier."""

    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor(n_features)
        self.label_classifier = LabelClassifier(n_classes)
        self.domain_classifier = DomainClassifier()

        # Backward-compatible aliases used by reporting scripts.
        self.feature = self.feature_extractor
        self.classifier = self.label_classifier
        self.domain = self.domain_classifier

    def forward_class(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature_extractor(x)
        return self.label_classifier(h)

    def forward_domain(self, x: torch.Tensor, lambd: float) -> torch.Tensor:
        h = self.feature_extractor(x)
        return self.domain_classifier(grad_reverse(h, lambd))

    def domain_no_grl(self, x: torch.Tensor) -> torch.Tensor:
        return self.domain_classifier(self.feature_extractor(x))


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c[:2] in {f"s{i}" for i in range(1, 7)} and c[2:3] == "_"]


def encode(labels: np.ndarray, classes: list[str]) -> np.ndarray:
    mapping = {label: i for i, label in enumerate(classes)}
    return np.array([mapping[str(v)] for v in labels], dtype=np.int64)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on an empty evaluation set.")
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    recalls, f1s = [], []
    for c in range(n_classes):
        tp = cm[c, c]
        precision = tp / max(1, cm[:, c].sum())
        recall = tp / max(1, cm[c, :].sum())
        recalls.append(recall)
        f1s.append(2 * precision * recall / max(1e-12, precision + recall))
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }


def class_counts(df: pd.DataFrame, classes: list[str]) -> str:
    if len(df) == 0:
        return ""
    counts = df["label"].value_counts().reindex(classes, fill_value=0)
    return "; ".join(f"{label}:{int(counts[label])}" for label in classes)


def days_text(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return ""
    chunks = []
    for period in ["Period_I", "Period_II", "Period_III", "Period_IV"]:
        sub = df[df["period"].eq(period)]
        if sub.empty:
            continue
        days = sorted(int(d) for d in sub["day_num"].dropna().unique() if int(d) > 0)
        chunks.append(f"{period} Day{min(days)}-Day{max(days)}" if days else period)
    return "; ".join(chunks)


natural_key = protocol.natural_key
period_i_holdout = protocol.period_i_holdout
target_domain_features = protocol.target_period_mask
SPLITS = protocol.DA_SPLITS


VARIANTS = [
    {"variant": "MLP source only", "use_domain_loss": False, "use_target_labels": False, "lambdas": [0.0]},
    {"variant": "DANN-UDA", "use_domain_loss": True, "use_target_labels": False, "lambdas": LAMBDA_VALUES},
    {"variant": "MLP source + target labels", "use_domain_loss": False, "use_target_labels": True, "lambdas": [0.0]},
    {"variant": "DANN-semi", "use_domain_loss": True, "use_target_labels": True, "lambdas": LAMBDA_VALUES},
]

SEED_MEAN_COLUMNS = [
    "best_epoch",
    "L_last",
    "Ld_last",
    "Ld_cdan_last",
    "Lc_last",
    "backward_loss_scalar_last",
    "class_loss_last",
    "domain_loss_last",
    "alignment_loss_last",
    "train_accuracy",
    "validation_accuracy",
    "test_accuracy",
    "train_balanced_accuracy",
    "validation_balanced_accuracy",
    "test_balanced_accuracy",
    "train_macro_f1",
    "validation_macro_f1",
    "test_macro_f1",
    "train_test_accuracy_gap",
    "domain_discriminator_accuracy",
]
SEED_STD_COLUMNS = [
    "best_epoch",
    "L_last",
    "Ld_last",
    "Ld_cdan_last",
    "Lc_last",
    "backward_loss_scalar_last",
    "class_loss_last",
    "domain_loss_last",
    "alignment_loss_last",
    "train_accuracy",
    "validation_accuracy",
    "test_accuracy",
    "train_balanced_accuracy",
    "validation_balanced_accuracy",
    "test_balanced_accuracy",
    "train_macro_f1",
    "validation_macro_f1",
    "test_macro_f1",
    "train_test_accuracy_gap",
    "domain_discriminator_accuracy",
]
SEED_RANGE_COLUMNS = SEED_STD_COLUMNS
SEED_POLICY = protocol.SEED_POLICY


def standardize(x_ref: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    if len(x_ref) == 0:
        raise ValueError("Standardization reference set is empty.")
    if not np.isfinite(x_ref).all():
        raise ValueError("Standardization reference contains NaN or infinite values; inspect feature extraction before training.")
    for i, arr in enumerate(arrays):
        if not np.isfinite(arr).all():
            raise ValueError(f"Input array {i} contains NaN or infinite values; inspect feature extraction before training.")
    mean = x_ref.mean(axis=0)
    std = x_ref.std(axis=0)
    if np.any(std < 1e-9):
        bad = np.where(std < 1e-9)[0].tolist()
        raise ValueError(f"Standardization reference has near-constant feature columns: {bad[:20]}. Do not silently rescale them.")
    return tuple(((arr - mean) / std).astype(np.float32) for arr in arrays)


def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32)


def to_long(y: np.ndarray) -> torch.Tensor:
    return torch.tensor(y, dtype=torch.long)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    if np.any(counts == 0):
        missing = np.where(counts == 0)[0].tolist()
        raise ValueError(f"Training labels are missing classes {missing}; class-weighted CE would hide an invalid split.")
    weights = len(y) / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def select_validation_arrays(
    split: dict,
    source_val_x: np.ndarray,
    y_source_val: np.ndarray,
    target_val_x: np.ndarray,
    y_target_val: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    selection = str(split["selection"])
    if selection == "target_validation_labels":
        if len(target_val_x) == 0:
            raise ValueError(f"{split['split']} declares target validation selection but target_val is empty.")
        return target_val_x, y_target_val, "target validation labels"
    if selection == "source_validation_labels":
        if len(source_val_x) == 0:
            raise ValueError(f"{split['split']} declares source validation selection but source_val is empty.")
        return source_val_x, y_source_val, "source validation labels"
    if selection == NO_VALIDATION_SELECTION:
        return None, None, NO_VALIDATION_SELECTION
    raise ValueError(f"Unknown split selection policy: {selection}")


def predict(model: DANN, x: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.forward_class(x).argmax(dim=1).cpu().numpy()


def predict_domain(model: DANN, x: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.domain_no_grl(x).argmax(dim=1).cpu().numpy()


def train_model(
    x_cls: np.ndarray,
    y_cls: np.ndarray,
    x_source_domain: np.ndarray,
    x_target_domain: np.ndarray,
    x_select: np.ndarray,
    y_select: np.ndarray,
    n_classes: int,
    use_domain_loss: bool,
    lambda_value: float,
    seed: int,
) -> tuple[DANN, dict[str, float]]:
    set_random_seed(seed)
    model = DANN(x_cls.shape[1], n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    x_cls_t = to_tensor(x_cls)
    y_cls_t = to_long(y_cls)
    x_select_t = to_tensor(x_select) if x_select is not None else None
    y_weight = class_weights(y_cls, n_classes)

    if use_domain_loss and len(x_target_domain) == 0:
        raise ValueError("Domain loss was requested, but x_target_domain is empty.")

    x_dom_t = torch.empty((0, x_cls.shape[1]), dtype=torch.float32)
    y_dom_t = torch.empty((0,), dtype=torch.long)
    if len(x_target_domain) > 0:
        x_dom_t = torch.cat([to_tensor(x_source_domain), to_tensor(x_target_domain)], dim=0)
        y_dom_t = torch.cat(
            [
                torch.zeros(len(x_source_domain), dtype=torch.long),
                torch.ones(len(x_target_domain), dtype=torch.long),
            ],
            dim=0,
        )

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_score = (-1.0, -1.0)
    best_epoch = 0
    last_L = 0.0
    last_Ld = 0.0
    last_backward_loss_scalar = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        class_logits = model.forward_class(x_cls_t)
        L = F.cross_entropy(class_logits, y_cls_t, weight=y_weight)
        backward_loss_scalar = L
        Ld = torch.tensor(0.0)
        if use_domain_loss:
            domain_logits = model.forward_domain(x_dom_t, lambda_value)
            Ld = F.cross_entropy(domain_logits, y_dom_t)
            backward_loss_scalar = L + Ld
        backward_loss_scalar.backward()
        optimizer.step()
        last_L = float(L.detach().cpu())
        last_Ld = float(Ld.detach().cpu())
        last_backward_loss_scalar = float(backward_loss_scalar.detach().cpu())

        if x_select_t is not None and (epoch % 10 == 0 or epoch == EPOCHS):
            select_pred = predict(model, x_select_t)
            select_m = metric_dict(y_select, select_pred, n_classes)
            score = (select_m["macro_f1"], select_m["balanced_accuracy"], select_m["accuracy"])
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif x_select_t is None and epoch == EPOCHS:
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": float(best_epoch),
        "L_last": last_L,
        "Ld_last": last_Ld,
        "backward_loss_scalar_last": last_backward_loss_scalar,
        "class_loss_last": last_L,
        "domain_loss_last": last_Ld,
    }


def evaluate_split(df: pd.DataFrame, split: dict, task_name: str, classes: list[str]) -> tuple[list[dict], list[dict]]:
    task = df[df["label"].isin(classes)].copy().reset_index(drop=True)
    masks = {
        "source_train": split["source_train"](task),
        "source_val": split["source_val"](task),
        "target_labeled_train": split["target_labeled_train"](task),
        "target_domain": split["target_domain"](task),
        "target_val": split["target_val"](task),
        "target_test": split["target_test"](task),
    }
    parts = {name: task[mask].copy() for name, mask in masks.items()}
    if len(parts["source_train"]) == 0 or len(parts["target_test"]) == 0:
        raise ValueError(f"{split['split']} has empty source_train or target_test for {task_name}.")
    if not set(classes).issubset(set(parts["source_train"]["label"])):
        raise ValueError(f"{split['split']} source_train is missing classes for {task_name}.")
    if len(parts["target_test"]) and not set(classes).issubset(set(parts["target_test"]["label"])):
        raise ValueError(f"{split['split']} target_test is missing classes for {task_name}.")

    split_rows = []
    usage_text = {
        "source_train": "labels used for classifier loss",
        "source_val": "labels used only for model selection",
        "target_labeled_train": "labels used only by semi-supervised variants",
        "target_domain": "gas labels not used; features used for UDA/domain loss",
        "target_val": "labels used only for model selection",
        "target_test": "labels used only for final evaluation",
    }
    for name, part in parts.items():
        split_rows.append(
            {
                "split": split["split"],
                "task": task_name,
                "partition": name,
                "n": len(part),
                "class_counts": class_counts(part, classes),
                "period_days": days_text(part),
                "label_usage": usage_text[name],
            }
        )

    cols = feature_columns(task)
    source_raw = parts["source_train"][cols].to_numpy(float)
    source_val_raw = parts["source_val"][cols].to_numpy(float)
    target_label_raw = parts["target_labeled_train"][cols].to_numpy(float)
    target_domain_raw = parts["target_domain"][cols].to_numpy(float)
    target_val_raw = parts["target_val"][cols].to_numpy(float)
    target_test_raw = parts["target_test"][cols].to_numpy(float)

    x_ref = source_raw
    source_x, source_val_x, target_label_x, target_domain_x, target_val_x, target_test_x = standardize(
        x_ref,
        source_raw,
        source_val_raw,
        target_label_raw,
        target_domain_raw,
        target_val_raw,
        target_test_raw,
    )

    y_source = encode(parts["source_train"]["label"].to_numpy(str), classes)
    y_source_val = encode(parts["source_val"]["label"].to_numpy(str), classes) if len(parts["source_val"]) else np.array([], dtype=np.int64)
    y_target_label = encode(parts["target_labeled_train"]["label"].to_numpy(str), classes) if len(parts["target_labeled_train"]) else np.array([], dtype=np.int64)
    y_target_val = encode(parts["target_val"]["label"].to_numpy(str), classes) if len(parts["target_val"]) else np.array([], dtype=np.int64)
    y_target_test = encode(parts["target_test"]["label"].to_numpy(str), classes)

    x_select, y_select, selection_labels = select_validation_arrays(
        split,
        source_val_x,
        y_source_val,
        target_val_x,
        y_target_val,
    )

    rows = []
    for variant in VARIANTS:
        if variant["use_target_labels"] and len(target_label_x) == 0:
            continue
        if variant["use_target_labels"]:
            x_cls = np.vstack([source_x, target_label_x])
            y_cls = np.concatenate([y_source, y_target_label])
            classifier_label_usage = "source labels + selected target labels"
        else:
            x_cls = source_x
            y_cls = y_source
            classifier_label_usage = "source labels only"

        if selection_labels == NO_VALIDATION_SELECTION:
            fixed_lambda = DEFAULT_LAMBDA if variant["use_domain_loss"] else 0.0
            candidates = [(fixed_lambda, seed) for seed in SEEDS]
        else:
            candidates = [(lambda_value, seed) for lambda_value in variant["lambdas"] for seed in SEEDS]

        for lambda_value, seed in candidates:
                model, info = train_model(
                    x_cls=x_cls,
                    y_cls=y_cls,
                    x_source_domain=source_x,
                    x_target_domain=target_domain_x,
                    x_select=x_select,
                    y_select=y_select,
                    n_classes=len(classes),
                    use_domain_loss=variant["use_domain_loss"],
                    lambda_value=lambda_value,
                    seed=seed,
                )
                train_m = metric_dict(y_cls, predict(model, to_tensor(x_cls)), len(classes))
                select_m = (
                    {"accuracy": np.nan, "balanced_accuracy": np.nan, "macro_f1": np.nan}
                    if x_select is None
                    else metric_dict(y_select, predict(model, to_tensor(x_select)), len(classes))
                )
                test_m = metric_dict(y_target_test, predict(model, to_tensor(target_test_x)), len(classes))

                domain_acc = np.nan
                if len(target_domain_x) > 0:
                    x_dom = np.vstack([source_x, target_domain_x])
                    y_dom = np.concatenate([np.zeros(len(source_x), dtype=int), np.ones(len(target_domain_x), dtype=int)])
                    domain_pred = predict_domain(model, to_tensor(x_dom))
                    domain_acc = float((domain_pred == y_dom).mean())

                rows.append(
                    {
                        "split": split["split"],
                        "description": split["description"],
                        "task": task_name,
                        "variant": variant["variant"],
                        "lambda": lambda_value,
                        "seed": seed,
                        "classifier_label_usage": classifier_label_usage,
                        "domain_label_usage": "source/target domain labels only; gas labels not used for target domain loss",
                        "model_selection_labels": selection_labels,
                        "best_epoch": int(info["best_epoch"]),
                        "L_last": info["L_last"],
                        "Ld_last": info["Ld_last"],
                        "backward_loss_scalar_last": info["backward_loss_scalar_last"],
                        "feature_extractor_objective": "min L - lambda*Ld via gradient reversal",
                        "class_loss_last": info["class_loss_last"],
                        "domain_loss_last": info["domain_loss_last"],
                        "classifier_train_n": len(y_cls),
                        "source_train_n": len(y_source),
                        "source_val_n": len(y_source_val),
                        "target_labeled_train_n": len(y_target_label),
                        "target_domain_n": len(target_domain_x),
                        "target_validation_n": len(y_target_val),
                        "target_test_n": len(y_target_test),
                        "train_accuracy": train_m["accuracy"],
                        "validation_accuracy": select_m["accuracy"],
                        "test_accuracy": test_m["accuracy"],
                        "train_balanced_accuracy": train_m["balanced_accuracy"],
                        "validation_balanced_accuracy": select_m["balanced_accuracy"],
                        "test_balanced_accuracy": test_m["balanced_accuracy"],
                        "train_macro_f1": train_m["macro_f1"],
                        "validation_macro_f1": select_m["macro_f1"],
                        "test_macro_f1": test_m["macro_f1"],
                        "train_test_accuracy_gap": train_m["accuracy"] - test_m["accuracy"],
                        "domain_discriminator_accuracy": domain_acc,
                    }
                )
    return rows, split_rows


def aggregate_seed_rows(seed_rows: pd.DataFrame, selection_status: str) -> pd.Series:
    if seed_rows.empty:
        raise ValueError("Cannot aggregate an empty seed group.")
    seed_rows = seed_rows.sort_values("seed").reset_index(drop=True)
    row = seed_rows.iloc[0].copy()
    seeds = [int(seed) for seed in sorted(seed_rows["seed"].unique())]
    row["seed"] = seeds[0]
    row["representative_seed"] = seeds[0]
    row["seed_values"] = ",".join(str(seed) for seed in seeds)
    row["seed_count"] = len(seeds)
    row["seed_policy"] = SEED_POLICY
    row["selection_status"] = selection_status
    row["judgment"] = selection_status
    for col in SEED_MEAN_COLUMNS:
        if col in seed_rows:
            row[col] = seed_rows[col].mean(skipna=True)
    for col in SEED_STD_COLUMNS:
        if col in seed_rows:
            row[f"{col}_std"] = seed_rows[col].std(skipna=True, ddof=protocol.SAMPLE_STD_DDOF)
    for col in SEED_RANGE_COLUMNS:
        if col in seed_rows:
            row[f"{col}_min"] = seed_rows[col].min(skipna=True)
            row[f"{col}_max"] = seed_rows[col].max(skipna=True)
    return row


def select_lambda_over_seed_means(sub: pd.DataFrame) -> float:
    grouped = (
        sub.groupby("lambda", sort=False)
        .agg(
            validation_macro_f1=("validation_macro_f1", "mean"),
            validation_balanced_accuracy=("validation_balanced_accuracy", "mean"),
            validation_accuracy=("validation_accuracy", "mean"),
        )
        .reset_index()
    )
    if grouped[["validation_macro_f1", "validation_balanced_accuracy", "validation_accuracy"]].isna().any().any():
        raise ValueError("Validation metrics are required for validation-based hyperparameter selection.")
    ordered = grouped.sort_values(
        ["validation_macro_f1", "validation_balanced_accuracy", "validation_accuracy", "lambda"],
        ascending=[False, False, False, True],
    )
    return float(ordered.iloc[0]["lambda"])


def choose_best(results: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for (_split, _task, _variant), sub in results.groupby(["split", "task", "variant"], sort=False):
        no_validation = sub["model_selection_labels"].eq(NO_VALIDATION_SELECTION).all()
        if no_validation:
            default_lambda = 0.0 if _variant == "MLP source only" else DEFAULT_LAMBDA
            fixed = sub[np.isclose(sub["lambda"].astype(float), default_lambda)]
            if fixed.empty:
                raise ValueError(f"Missing fixed no-validation candidate for {_split} / {_task} / {_variant}.")
            selection_status = "fixed hyperparameters because the split has no validation set; metrics averaged over seeds"
            selected.append(aggregate_seed_rows(fixed, selection_status))
        else:
            best_lambda = select_lambda_over_seed_means(sub)
            best_rows = sub[np.isclose(sub["lambda"].astype(float), best_lambda)]
            selection_status = "lambda selected by mean validation macro-F1, balanced accuracy, and accuracy over seeds"
            selected.append(aggregate_seed_rows(best_rows, selection_status))
    out = pd.DataFrame(selected).reset_index(drop=True)
    no_validation = out["model_selection_labels"].eq(NO_VALIDATION_SELECTION)
    for col in ["validation_accuracy", "validation_balanced_accuracy", "validation_macro_f1"]:
        out.loc[no_validation, col] = np.nan
    return out


def fmt(v: float) -> str:
    return "NA" if pd.isna(v) else f"{float(v):.3f}"


def fmt_mean_std(row: pd.Series, col: str) -> str:
    if col not in row or pd.isna(row[col]):
        return "NA"
    if f"{col}_std" in row and pd.notna(row[f"{col}_std"]):
        return f"{float(row[col]):.3f}+/-{float(row[f'{col}_std']):.3f}"
    return fmt(row[col])


def short_split(name: str) -> str:
    return name.split("_", 1)[0]


def write_svg(summary: pd.DataFrame, path: Path) -> None:
    cols = [
        ("Split", 54),
        ("Task", 92),
        ("Variant", 145),
        ("Lam", 48),
        ("Train", 58),
        ("Val", 58),
        ("Test", 58),
        ("BA", 58),
        ("F1", 58),
        ("Gap", 58),
        ("Dom", 58),
        ("Label usage", 210),
        ("Judgment", 245),
    ]
    rows = len(summary)
    row_h = 31
    width = 32 + sum(w for _, w in cols) + 32
    height = 116 + row_h * (rows + 1) + 160
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="32" font-family="Arial" font-size="22" font-weight="700">DANN Multi-split Results for E-nose Drift Adaptation</text>',
        '<text x="24" y="56" font-family="Arial" font-size="13" fill="#555">Target gas labels are never used in DANN-UDA training. Semi variants use selected target-day labels for classifier calibration. Domain accuracy closer to 0.5 means more domain-invariant features.</text>',
    ]
    x0, y = 24, 84
    x = x0
    for label, w in cols:
        lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{row_h}" fill="#f1f4f8" stroke="#d8dee8"/>')
        lines.append(f'<text x="{x + 5}" y="{y + 20}" font-family="Arial" font-size="10" font-weight="700">{label}</text>')
        x += w
    y += row_h
    for _, row in summary.iterrows():
        vals = [
            short_split(row["split"]),
            "3-class",
            row["variant"],
            fmt(row["lambda"]),
            fmt_mean_std(row, "train_accuracy"),
            fmt_mean_std(row, "validation_accuracy"),
            fmt_mean_std(row, "test_accuracy"),
            fmt_mean_std(row, "test_balanced_accuracy"),
            fmt_mean_std(row, "test_macro_f1"),
            fmt_mean_std(row, "train_test_accuracy_gap"),
            fmt_mean_std(row, "domain_discriminator_accuracy"),
            row["classifier_label_usage"],
            row["judgment"],
        ]
        x = x0
        for (label, w), val in zip(cols, vals):
            fill = "white"
            lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{row_h}" fill="{fill}" stroke="#d8dee8"/>')
            lines.append(f'<text x="{x + 5}" y="{y + 20}" font-family="Arial" font-size="9">{str(val)[:38]}</text>')
            x += w
        y += row_h

    legend = [
        "S1: Period I labels train; no validation; all Period II/III labels used only for final evaluation.",
        "S2: Period I train/validation holdout 25/5; all Period II/III labels used only for final evaluation.",
        "S3: Period I train/validation holdout 20/10; all Period II/III labels used only for final evaluation.",
        "S4: Period I train; Day1-2 validation; Day3-19 test.",
        "S5: Period I + Day1 target labels train; Day2-3 validation; Day4-19 test.",
        "S6: Period I + Day1-2 target labels train; Day3-5 validation; Day6-19 test.",
        "S7: Period I + Day1-3 target labels train; Day4-5 validation; Day6-19 test.",
        "UDA variants use Period II/III target features without gas labels for domain loss in every split.",
    ]
    y += 24
    for line in legend:
        lines.append(f'<text x="24" y="{y}" font-family="Arial" font-size="12" fill="#555">{line}</text>')
        y += 18
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    torch.set_num_threads(1)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURE_PATH)
    all_rows, split_rows = [], []
    for split in SPLITS:
        for task_name, classes in REPORT_TASKS:
            rows, split_info = evaluate_split(df, split, task_name, classes)
            all_rows.extend(rows)
            split_rows.extend(split_info)

    results = pd.DataFrame(all_rows)
    splits = pd.DataFrame(split_rows)
    summary = choose_best(results)

    results.to_csv(TABLE_DIR / "dann_multi_split_all_runs.csv", index=False)
    summary.to_csv(TABLE_DIR / "dann_multi_split_selected_summary.csv", index=False)
    splits.to_csv(TABLE_DIR / "dann_multi_split_partitions.csv", index=False)
    write_svg(summary, FIG_DIR / "dann_multi_split_summary.svg")

    print("PyTorch", torch.__version__, "CUDA", torch.cuda.is_available())
    print("\nPartitions")
    print(splits.to_string(index=False))
    print("\nSelected summary")
    show_cols = [
        "split",
        "task",
        "variant",
        "lambda",
        "seed",
        "classifier_label_usage",
        "model_selection_labels",
        "train_accuracy",
        "validation_accuracy",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "domain_discriminator_accuracy",
        "judgment",
    ]
    print(summary[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
