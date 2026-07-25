from __future__ import annotations

import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import run_dann_multi_splits as base


FIG_DIR = base.OUT_ROOT / "figures" / "coral"
VARIANTS = [
    {"variant": "MLP source only", "method": "source_only", "lambdas": [0.0]},
    {"variant": "Deep CORAL-UDA", "method": "deep_coral", "lambdas": base.LAMBDA_VALUES},
]

ALL_RUNS_PATH = base.TABLE_DIR / "deep_coral_multi_split_all_runs.csv"
SUMMARY_PATH = base.TABLE_DIR / "deep_coral_multi_split_selected_summary.csv"
PARTITIONS_PATH = base.TABLE_DIR / "deep_coral_multi_split_partitions.csv"
CURVES_PATH = base.TABLE_DIR / "deep_coral_multi_split_training_curves.csv"
FEATURE_SPACE_PATH = base.TABLE_DIR / "deep_coral_multi_split_feature_space.csv"
FLOW_SVG_PATH = FIG_DIR / "deep_coral_research_flow.svg"
SUMMARY_SVG_PATH = FIG_DIR / "deep_coral_multi_split_summary.svg"
HTML_PATH = FIG_DIR / "deep_coral_multi_split_report.html"
LOG_EVERY = 10


def covariance(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("CORAL covariance expects a 2D feature matrix.")
    if features.shape[0] < 2:
        raise ValueError("CORAL covariance requires at least two samples per domain.")
    centered = features - features.mean(dim=0, keepdim=True)
    return centered.t().matmul(centered) / (features.shape[0] - 1)


def coral_loss(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("Deep CORAL expects 2D source and target latent feature matrices.")
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("Source and target latent features must have the same dimensionality for CORAL.")
    d = source_features.shape[1]
    source_cov = covariance(source_features)
    target_cov = covariance(target_features)
    return (source_cov - target_cov).pow(2).sum() / (4.0 * d * d)


def train_deep_coral_model(
    x_cls: np.ndarray,
    y_cls: np.ndarray,
    x_source_domain: np.ndarray,
    x_target_domain: np.ndarray,
    x_select: np.ndarray | None,
    y_select: np.ndarray | None,
    n_classes: int,
    lambda_value: float,
    seed: int,
) -> tuple[base.DANN, dict[str, float]]:
    if len(x_target_domain) == 0:
        raise ValueError("Deep CORAL requires non-empty target-domain features.")

    base.set_random_seed(seed)
    model = base.DANN(x_cls.shape[1], n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY)

    x_cls_t = base.to_tensor(x_cls)
    y_cls_t = base.to_long(y_cls)
    x_source_t = base.to_tensor(x_source_domain)
    x_target_t = base.to_tensor(x_target_domain)
    x_select_t = base.to_tensor(x_select) if x_select is not None else None
    y_weight = base.class_weights(y_cls, n_classes)

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_score = (-1.0, -1.0, -1.0)
    best_epoch = 0
    last_class_loss = 0.0
    last_coral_loss = 0.0
    last_total_loss = 0.0

    for epoch in range(1, base.EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        class_logits = model.forward_class(x_cls_t)
        class_loss = F.cross_entropy(class_logits, y_cls_t, weight=y_weight)
        source_h = model.feature(x_source_t)
        target_h = model.feature(x_target_t)
        alignment_loss = coral_loss(source_h, target_h)
        total_loss = class_loss + lambda_value * alignment_loss
        total_loss.backward()
        optimizer.step()

        last_class_loss = float(class_loss.detach().cpu())
        last_coral_loss = float(alignment_loss.detach().cpu())
        last_total_loss = float(total_loss.detach().cpu())

        if x_select_t is not None and (epoch % 10 == 0 or epoch == base.EPOCHS):
            select_pred = base.predict(model, x_select_t)
            select_m = base.metric_dict(y_select, select_pred, n_classes)
            score = (select_m["macro_f1"], select_m["balanced_accuracy"], select_m["accuracy"])
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif x_select_t is None and epoch == base.EPOCHS:
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": float(best_epoch),
        "class_loss_last": last_class_loss,
        "coral_loss_last": last_coral_loss,
        "total_loss_last": last_total_loss,
    }


def train_source_only_model(
    x_cls: np.ndarray,
    y_cls: np.ndarray,
    x_source_domain: np.ndarray,
    x_target_domain: np.ndarray,
    x_select: np.ndarray | None,
    y_select: np.ndarray | None,
    n_classes: int,
    seed: int,
) -> tuple[base.DANN, dict[str, float]]:
    return base.train_model(
        x_cls=x_cls,
        y_cls=y_cls,
        x_source_domain=x_source_domain,
        x_target_domain=x_target_domain,
        x_select=x_select,
        y_select=y_select,
        n_classes=n_classes,
        use_domain_loss=False,
        lambda_value=0.0,
        seed=seed,
    )


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
        "target_labeled_train": "not used by Deep CORAL-UDA; retained only for split audit",
        "target_domain": "gas labels not used; features used for Deep CORAL covariance loss",
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
                "class_counts": base.class_counts(part, classes),
                "period_days": base.days_text(part),
                "label_usage": usage_text[name],
            }
        )

    cols = base.feature_columns(task)
    source_raw = parts["source_train"][cols].to_numpy(float)
    source_val_raw = parts["source_val"][cols].to_numpy(float)
    target_domain_raw = parts["target_domain"][cols].to_numpy(float)
    target_val_raw = parts["target_val"][cols].to_numpy(float)
    target_test_raw = parts["target_test"][cols].to_numpy(float)

    x_ref = source_raw
    source_x, source_val_x, target_domain_x, target_val_x, target_test_x = base.standardize(
        x_ref,
        source_raw,
        source_val_raw,
        target_domain_raw,
        target_val_raw,
        target_test_raw,
    )

    y_source = base.encode(parts["source_train"]["label"].to_numpy(str), classes)
    y_source_val = (
        base.encode(parts["source_val"]["label"].to_numpy(str), classes)
        if len(parts["source_val"])
        else np.array([], dtype=np.int64)
    )
    y_target_val = (
        base.encode(parts["target_val"]["label"].to_numpy(str), classes)
        if len(parts["target_val"])
        else np.array([], dtype=np.int64)
    )
    y_target_test = base.encode(parts["target_test"]["label"].to_numpy(str), classes)

    x_select, y_select, selection_labels = base.select_validation_arrays(
        split,
        source_val_x,
        y_source_val,
        target_val_x,
        y_target_val,
    )

    rows = []
    for variant in VARIANTS:
        x_cls = source_x
        y_cls = y_source
        if selection_labels == base.NO_VALIDATION_SELECTION:
            fixed_lambda = base.DEFAULT_LAMBDA if variant["method"] == "deep_coral" else 0.0
            candidates = [(fixed_lambda, base.DEFAULT_SEED)]
        else:
            candidates = [(lambda_value, seed) for lambda_value in variant["lambdas"] for seed in base.SEEDS]

        for lambda_value, seed in candidates:
            if variant["method"] == "source_only":
                model, info = train_source_only_model(
                    x_cls=x_cls,
                    y_cls=y_cls,
                    x_source_domain=source_x,
                    x_target_domain=target_domain_x,
                    x_select=x_select,
                    y_select=y_select,
                    n_classes=len(classes),
                    seed=seed,
                )
                coral_loss_last = np.nan
                total_loss_last = info["backward_loss_scalar_last"]
                objective = "min L using the same MLP source-only classification loss as DANN"
            elif variant["method"] == "deep_coral":
                model, info = train_deep_coral_model(
                    x_cls=x_cls,
                    y_cls=y_cls,
                    x_source_domain=source_x,
                    x_target_domain=target_domain_x,
                    x_select=x_select,
                    y_select=y_select,
                    n_classes=len(classes),
                    lambda_value=lambda_value,
                    seed=seed,
                )
                coral_loss_last = info["coral_loss_last"]
                total_loss_last = info["total_loss_last"]
                objective = "min L + lambda*Lcoral, where Lcoral = ||Cs - Ct||_F^2 / (4*d^2)"
            else:
                raise ValueError(f"Unknown Deep CORAL variant method: {variant['method']}")

            train_m = base.metric_dict(y_cls, base.predict(model, base.to_tensor(x_cls)), len(classes))
            select_m = (
                {"accuracy": np.nan, "balanced_accuracy": np.nan, "macro_f1": np.nan}
                if x_select is None
                else base.metric_dict(y_select, base.predict(model, base.to_tensor(x_select)), len(classes))
            )
            test_m = base.metric_dict(y_target_test, base.predict(model, base.to_tensor(target_test_x)), len(classes))

            rows.append(
                {
                    "split": split["split"],
                    "description": split["description"],
                    "task": task_name,
                    "variant": variant["variant"],
                    "lambda": lambda_value,
                    "seed": seed,
                    "classifier_label_usage": "source labels only",
                    "domain_label_usage": (
                        "source/target features only; gas labels not used for Deep CORAL covariance loss"
                        if variant["method"] == "deep_coral"
                        else "no target-domain adaptation loss"
                    ),
                    "model_selection_labels": selection_labels,
                    "best_epoch": int(info["best_epoch"]),
                    "class_loss_last": info["class_loss_last"],
                    "coral_loss_last": coral_loss_last,
                    "total_loss_last": total_loss_last,
                    "feature_extractor_objective": objective,
                    "classifier_train_n": len(y_cls),
                    "source_train_n": len(y_source),
                    "source_val_n": len(y_source_val),
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
                }
            )
    return rows, split_rows


def choose_best(results: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for (_split, _task, _variant), sub in results.groupby(["split", "task", "variant"], sort=False):
        no_validation = sub["model_selection_labels"].eq(base.NO_VALIDATION_SELECTION).all()
        if no_validation:
            default_lambda = 0.0 if _variant == "MLP source only" else base.DEFAULT_LAMBDA
            fixed = sub[np.isclose(sub["lambda"].astype(float), default_lambda) & sub["seed"].eq(base.DEFAULT_SEED)]
            if fixed.empty:
                raise ValueError(f"Missing fixed no-validation candidate for {_split} / {_task} / {_variant}.")
            ordered = fixed
        else:
            ordered = sub.sort_values(
                ["validation_macro_f1", "validation_balanced_accuracy", "validation_accuracy", "lambda", "seed"],
                ascending=[False, False, False, True, True],
            )
        selected.append(ordered.iloc[0])
    out = pd.DataFrame(selected).reset_index(drop=True)
    no_validation = out["model_selection_labels"].eq(base.NO_VALIDATION_SELECTION)
    out["selection_status"] = np.where(
        no_validation,
        "fixed hyperparameters because the split has no validation set",
        "hyperparameters selected by validation macro-F1, balanced accuracy, and accuracy",
    )
    for col in ["validation_accuracy", "validation_balanced_accuracy", "validation_macro_f1"]:
        out.loc[no_validation, col] = np.nan
    return out


def split_by_name(name: str) -> dict:
    for split in base.SPLITS:
        if split["split"] == name:
            return split
    raise ValueError(f"Unknown split: {name}")


def task_classes(task_name: str) -> list[str]:
    for name, classes in base.REPORT_TASKS:
        if name == task_name:
            return classes
    raise ValueError(f"Unknown task: {task_name}")


def prepare_parts(df: pd.DataFrame, split: dict, classes: list[str]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    task = df[df["label"].isin(classes)].copy().reset_index(drop=True)
    masks = {
        "source_train": split["source_train"](task),
        "source_val": split["source_val"](task),
        "target_labeled_train": split["target_labeled_train"](task),
        "target_domain": split["target_domain"](task),
        "target_val": split["target_val"](task),
        "target_test": split["target_test"](task),
    }
    return task, {name: task[mask].copy() for name, mask in masks.items()}


def make_arrays(task: pd.DataFrame, parts: dict[str, pd.DataFrame], classes: list[str]) -> dict:
    cols = base.feature_columns(task)
    raw = {
        "source_train": parts["source_train"][cols].to_numpy(float),
        "source_val": parts["source_val"][cols].to_numpy(float),
        "target_domain": parts["target_domain"][cols].to_numpy(float),
        "target_val": parts["target_val"][cols].to_numpy(float),
        "target_test": parts["target_test"][cols].to_numpy(float),
    }
    x_ref = raw["source_train"]
    standardized = base.standardize(
        x_ref,
        raw["source_train"],
        raw["source_val"],
        raw["target_domain"],
        raw["target_val"],
        raw["target_test"],
    )
    x = dict(zip(raw.keys(), standardized))
    y = {
        "source_train": base.encode(parts["source_train"]["label"].to_numpy(str), classes),
        "source_val": (
            base.encode(parts["source_val"]["label"].to_numpy(str), classes)
            if len(parts["source_val"])
            else np.array([], dtype=np.int64)
        ),
        "target_val": (
            base.encode(parts["target_val"]["label"].to_numpy(str), classes)
            if len(parts["target_val"])
            else np.array([], dtype=np.int64)
        ),
        "target_test": base.encode(parts["target_test"]["label"].to_numpy(str), classes),
    }
    return {"x": x, "y": y, "feature_columns": cols}


def selected_variant_method(variant: str) -> str:
    if variant == "MLP source only":
        return "source_only"
    if variant == "Deep CORAL-UDA":
        return "deep_coral"
    raise ValueError(f"Unknown selected variant: {variant}")


def selection_data(split: dict, arrays: dict) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    return base.select_validation_arrays(
        split,
        arrays["x"]["source_val"],
        arrays["y"]["source_val"],
        arrays["x"]["target_val"],
        arrays["y"]["target_val"],
    )


def train_with_curves(
    row: pd.Series,
    arrays: dict,
    classes: list[str],
) -> tuple[base.DANN, list[dict]]:
    seed = int(row["seed"])
    lambda_value = float(row["lambda"])
    method = selected_variant_method(str(row["variant"]))
    base.set_random_seed(seed)

    x_cls = arrays["x"]["source_train"]
    y_cls = arrays["y"]["source_train"]
    x_source = arrays["x"]["source_train"]
    x_target = arrays["x"]["target_domain"]
    x_test = arrays["x"]["target_test"]
    y_test = arrays["y"]["target_test"]
    split = split_by_name(str(row["split"]))
    x_select, y_select, _selection_label = selection_data(split, arrays)

    if method == "deep_coral" and len(x_target) == 0:
        raise ValueError("Deep CORAL selected run has empty target-domain features.")

    model = base.DANN(x_cls.shape[1], len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY)
    x_cls_t = base.to_tensor(x_cls)
    y_cls_t = base.to_long(y_cls)
    x_source_t = base.to_tensor(x_source)
    x_target_t = base.to_tensor(x_target)
    y_weight = base.class_weights(y_cls, len(classes))
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_score = (-1.0, -1.0, -1.0)
    curves = []

    for epoch in range(1, base.EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        class_logits = model.forward_class(x_cls_t)
        class_loss = F.cross_entropy(class_logits, y_cls_t, weight=y_weight)
        alignment_loss = torch.tensor(0.0)
        total_loss = class_loss
        if method == "deep_coral":
            source_h = model.feature(x_source_t)
            target_h = model.feature(x_target_t)
            alignment_loss = coral_loss(source_h, target_h)
            total_loss = class_loss + lambda_value * alignment_loss
        total_loss.backward()
        optimizer.step()

        if epoch % LOG_EVERY == 0 or epoch == 1 or epoch == base.EPOCHS:
            train_m = base.metric_dict(y_cls, base.predict(model, base.to_tensor(x_cls)), len(classes))
            select_m = (
                {"accuracy": np.nan, "balanced_accuracy": np.nan, "macro_f1": np.nan}
                if x_select is None
                else base.metric_dict(y_select, base.predict(model, base.to_tensor(x_select)), len(classes))
            )
            test_m = base.metric_dict(y_test, base.predict(model, base.to_tensor(x_test)), len(classes))
            curves.append(
                {
                    "epoch": epoch,
                    "class_loss": float(class_loss.detach().cpu()),
                    "coral_loss": float(alignment_loss.detach().cpu()),
                    "total_loss": float(total_loss.detach().cpu()),
                    "train_accuracy": train_m["accuracy"],
                    "validation_accuracy": select_m["accuracy"],
                    "test_accuracy": test_m["accuracy"],
                    "train_balanced_accuracy": train_m["balanced_accuracy"],
                    "validation_balanced_accuracy": select_m["balanced_accuracy"],
                    "test_balanced_accuracy": test_m["balanced_accuracy"],
                    "train_macro_f1": train_m["macro_f1"],
                    "validation_macro_f1": select_m["macro_f1"],
                    "test_macro_f1": test_m["macro_f1"],
                }
            )
            if x_select is not None:
                score = (select_m["macro_f1"], select_m["balanced_accuracy"], select_m["accuracy"])
                if score > best_score:
                    best_score = score
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif epoch == base.EPOCHS:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, curves


def pca_2d(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        raise ValueError("Cannot compute PCA coordinates for an empty feature matrix.")
    if not np.isfinite(x).all():
        raise ValueError("Cannot compute PCA coordinates with NaN or infinite values.")
    centered = x.astype(float) - x.astype(float).mean(axis=0, keepdims=True)
    if centered.shape[0] < 2:
        raise ValueError("PCA plot requires at least two samples.")
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    comp = vt[:2].T
    if comp.shape[1] < 2:
        comp = np.hstack([comp, np.zeros((comp.shape[0], 2 - comp.shape[1]))])
    return centered @ comp[:, :2]


def latent_features(model: base.DANN, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.feature(base.to_tensor(x)).cpu().numpy()


def feature_rows(
    split_name: str,
    task_name: str,
    variant: str,
    model: base.DANN,
    parts: dict[str, pd.DataFrame],
    arrays: dict,
) -> list[dict]:
    plot_parts = ["source_train", "source_val", "target_domain", "target_val", "target_test"]
    x_chunks = []
    meta_chunks = []
    method = selected_variant_method(variant)
    for name in plot_parts:
        if len(parts[name]) == 0:
            continue
        x_part = arrays["x"][name]
        x_chunks.append(x_part if method == "source_only" else latent_features(model, x_part))
        meta = parts[name].copy()
        meta["partition"] = name
        meta_chunks.append(meta)
    coords = pca_2d(np.vstack(x_chunks))
    meta_all = pd.concat(meta_chunks, ignore_index=True)
    rows = []
    for i, item in enumerate(meta_all.itertuples(index=False)):
        partition = str(getattr(item, "partition"))
        rows.append(
            {
                "split": split_name,
                "task": task_name,
                "variant": variant,
                "space": "standardized input PCA" if method == "source_only" else "learned 16D latent h PCA",
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "label": str(getattr(item, "label")),
                "domain": "source" if partition.startswith("source") else "target",
                "partition": partition,
                "period": str(getattr(item, "period")),
                "day_num": int(getattr(item, "day_num")),
                "target_file": str(getattr(item, "target_file")),
            }
        )
    return rows


def xml_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def fmt(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_flow_svg(feature_dim: int, latent_dim: int, n_classes: int) -> None:
    width, height = 1320, 700
    boxes = [
        (40, 90, 210, 120, "Feature table", f"x in R^{feature_dim}", "source/target split membership known"),
        (300, 90, 230, 120, "Source branch", f"x_s: n_s x {feature_dim}", "uses gas labels y_s for CE"),
        (300, 260, 230, 120, "Target branch", f"x_t: n_t x {feature_dim}", "no target gas labels used"),
        (590, 90, 230, 120, "Shared MLP feature extractor", f"G_f: R^{feature_dim} -> R^{latent_dim}", "same weights for source and target"),
        (870, 72, 220, 92, "Classifier head", f"G_y: R^{latent_dim} -> R^{n_classes}", "source labels y_s only"),
        (870, 214, 220, 112, "CORAL alignment", f"C_s,C_t in R^{latent_dim} x R^{latent_dim}", "uses domain membership: source vs target"),
        (1130, 72, 150, 92, "Class loss", "L_y = CE(G_y(h_s), y_s)", "labels used here"),
        (1130, 214, 150, 112, "CORAL loss", "Lcoral = ||Cs-Ct||^2/(4d^2)", "no gas labels"),
        (870, 410, 220, 100, "Training objective", "L = L_y + lambda Lcoral", "lambda selected only by validation policy"),
        (1130, 410, 150, 100, "Outputs", "target predictions + metrics", "test labels evaluation only"),
    ]
    arrows = [
        (250, 150, 300, 150),
        (250, 150, 300, 320),
        (530, 150, 590, 150),
        (530, 320, 590, 150),
        (820, 150, 870, 118),
        (820, 150, 870, 270),
        (1090, 118, 1130, 118),
        (1090, 270, 1130, 270),
        (1205, 164, 980, 410),
        (1205, 326, 980, 410),
        (1090, 460, 1130, 460),
    ]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#334155"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="40" y="42" font-family="Arial" font-size="26" font-weight="700" fill="#111827">Deep CORAL-UDA Scientific Workflow</text>',
        '<text x="40" y="68" font-family="Arial" font-size="14" fill="#4b5563">Independent CORAL experiment. Original MLP/DANN/CDAN/C-DANN scripts are not modified.</text>',
    ]
    for x1, y1, x2, y2 in arrows:
        lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#334155" stroke-width="1.8" marker-end="url(#arrow)"/>')
    for x, y, w, h, title, body, note in boxes:
        lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#f8fafc" stroke="#cbd5e1"/>')
        lines.append(f'<text x="{x + 14}" y="{y + 28}" font-family="Arial" font-size="15" font-weight="700" fill="#0f172a">{xml_escape(title)}</text>')
        lines.append(f'<text x="{x + 14}" y="{y + 55}" font-family="Arial" font-size="13" fill="#1f2937">{xml_escape(body)}</text>')
        lines.append(f'<text x="{x + 14}" y="{y + 82}" font-family="Arial" font-size="12" fill="#64748b">{xml_escape(note)}</text>')
    notes = [
        "Label usage: source gas labels y_s train the classifier; validation labels are used only for model selection; test labels are used only after training for final metrics.",
        "Domain information usage: source/target membership is used to form h_s and h_t covariance matrices. No target gas labels enter Lcoral.",
        "Reported dimensions are from the current feature table and model architecture.",
    ]
    y = 575
    for note in notes:
        lines.append(f'<text x="40" y="{y}" font-family="Arial" font-size="13" fill="#374151">{xml_escape(note)}</text>')
        y += 24
    lines.append("</svg>")
    FLOW_SVG_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_summary_svg(summary: pd.DataFrame) -> None:
    cols = [
        ("Split", 50),
        ("Variant", 132),
        ("Lam", 48),
        ("Seed", 42),
        ("Train", 54),
        ("Val", 54),
        ("Test", 54),
        ("BA", 54),
        ("F1", 54),
        ("Lcoral", 64),
        ("Labels", 170),
        ("Selection", 230),
    ]
    row_h = 31
    width = 44 + sum(w for _, w in cols) + 44
    height = 120 + row_h * (len(summary) + 1) + 150
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#111827">Deep CORAL Multi-split Results for E-nose Drift Adaptation</text>',
        '<text x="24" y="58" font-family="Arial" font-size="13" fill="#4b5563">Independent report: source-only MLP vs Deep CORAL-UDA. Target gas labels are not used by CORAL loss.</text>',
    ]
    x0, y = 24, 84
    x = x0
    for label, w in cols:
        lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{row_h}" fill="#eef2f7" stroke="#d8dee8"/>')
        lines.append(f'<text x="{x + 5}" y="{y + 20}" font-family="Arial" font-size="10" font-weight="700" fill="#1f2937">{xml_escape(label)}</text>')
        x += w
    y += row_h
    for _, row in summary.iterrows():
        vals = [
            base.short_split(row["split"]) if hasattr(base, "short_split") else str(row["split"]).split("_", 1)[0],
            row["variant"],
            fmt(row["lambda"]),
            int(row["seed"]),
            fmt(row["train_accuracy"]),
            fmt(row["validation_accuracy"]),
            fmt(row["test_accuracy"]),
            fmt(row["test_balanced_accuracy"]),
            fmt(row["test_macro_f1"]),
            fmt(row["coral_loss_last"]),
            row["classifier_label_usage"],
            row["selection_status"],
        ]
        x = x0
        for (_label, w), val in zip(cols, vals):
            lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{row_h}" fill="#ffffff" stroke="#d8dee8"/>')
            lines.append(f'<text x="{x + 5}" y="{y + 20}" font-family="Arial" font-size="9" fill="#111827">{xml_escape(str(val)[:36])}</text>')
            x += w
        y += row_h
    y += 24
    notes = [
        "Deep CORAL objective: L = CE(G_y(h_s), y_s) + lambda * ||C_s - C_t||_F^2 / (4*d^2).",
        "Input dimension: current feature columns from manifest_baseline_as_air_features.csv. Latent dimension: 16.",
        "All outputs are written under deep_coral_* paths and D:/thesis/figures/coral.",
    ]
    for note in notes:
        lines.append(f'<text x="24" y="{y}" font-family="Arial" font-size="12" fill="#4b5563">{xml_escape(note)}</text>')
        y += 18
    lines.append("</svg>")
    SUMMARY_SVG_PATH.write_text("\n".join(lines), encoding="utf-8")


def json_records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.replace({np.nan: None}).to_json(orient="records"))


def write_html(summary: pd.DataFrame, partitions: pd.DataFrame, curves: pd.DataFrame, features: pd.DataFrame, feature_dim: int) -> None:
    data = {
        "summary": json_records(summary),
        "partitions": json_records(partitions),
        "curves": json_records(curves),
        "features": json_records(features),
        "settings": {
            "inputDim": feature_dim,
            "latentDim": 16,
            "classes": ["air", "alcohol", "acetone"],
            "objective": "L = CE(G_y(h_s), y_s) + lambda * ||C_s - C_t||_F^2 / (4*d^2)",
            "labelUse": "source labels for classification; validation labels only for model selection; test labels only for final evaluation",
            "domainUse": "source/target membership partitions h_s and h_t for covariance alignment; no target gas labels are used",
            "epochs": base.EPOCHS,
            "learningRate": base.LEARNING_RATE,
            "weightDecay": base.WEIGHT_DECAY,
            "seeds": base.SEEDS,
            "lambda": base.LAMBDA_VALUES,
        },
    }
    data_json = json.dumps(data, ensure_ascii=False)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Deep CORAL Multi-split Report</title>
<style>
:root {{ --line:#d8dee8; --muted:#5f6b7a; --source:#1f77b4; --target:#d62728; }}
body {{ margin:0; font-family: Arial, Helvetica, sans-serif; background:#f6f8fb; color:#111827; }}
header {{ padding:24px 30px; background:#ffffff; border-bottom:1px solid var(--line); }}
h1 {{ margin:0 0 6px; font-size:26px; }}
h2 {{ margin:0 0 12px; font-size:18px; }}
.sub, .note {{ color:var(--muted); font-size:13px; line-height:1.45; }}
.layout {{ display:grid; grid-template-columns:260px 1fr; gap:18px; padding:18px; }}
aside, section {{ background:#ffffff; border:1px solid var(--line); border-radius:7px; padding:14px; }}
aside {{ align-self:start; position:sticky; top:14px; }}
label {{ display:block; font-size:12px; color:var(--muted); margin:10px 0 4px; }}
select {{ width:100%; padding:7px; border:1px solid var(--line); border-radius:6px; background:#fff; }}
.content {{ display:grid; gap:14px; }}
.metrics {{ display:grid; grid-template-columns:repeat(5,minmax(120px,1fr)); gap:10px; }}
.metric {{ border:1px solid var(--line); border-radius:7px; padding:10px; background:#fbfcfe; }}
.metric .k {{ color:var(--muted); font-size:11px; }}
.metric .v {{ font-size:20px; font-weight:700; margin-top:4px; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; }}
th, td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:left; vertical-align:top; }}
th {{ background:#f2f5f8; font-weight:700; }}
.pill {{ display:inline-block; padding:4px 7px; border-radius:999px; background:#eef3f8; font-size:11px; margin:2px 4px 2px 0; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.chart {{ border:1px solid var(--line); border-radius:7px; background:#fff; min-height:280px; }}
.axis {{ stroke:#9aa8b5; stroke-width:1; }}
.gridline {{ stroke:#edf1f5; stroke-width:1; }}
@media (max-width: 1000px) {{ .layout, .grid2 {{ grid-template-columns:1fr; }} aside {{ position:static; }} .metrics {{ grid-template-columns:repeat(2,minmax(120px,1fr)); }} }}
</style>
</head>
<body>
<header>
  <h1>Deep CORAL-UDA Multi-split Report</h1>
  <div class="sub">Independent CORAL report. Original MLP/DANN/CDAN/C-DANN outputs are not modified.</div>
</header>
<div class="layout">
  <aside>
    <label>Split</label><select id="splitSelect"></select>
    <label>Variant</label><select id="variantSelect"></select>
    <p class="note">Input R^<span id="inputDim"></span> -> shared MLP h in R^16 -> classifier logits R^3. CORAL uses source/target membership only.</p>
  </aside>
  <main class="content">
    <section><h2>Selected Model Summary</h2><div id="metrics" class="metrics"></div><div id="usage"></div></section>
    <section><h2>Variant Comparison</h2><div id="variantTable"></div></section>
    <section><h2>Partition and Label Usage</h2><div id="partitionTable"></div></section>
    <section><h2>Training Curves</h2><div class="grid2"><div id="lossChart" class="chart"></div><div id="accChart" class="chart"></div></div></section>
    <section><h2>Feature Space</h2><p id="spaceNote" class="note"></p><div id="featureChart" class="chart"></div></section>
    <section><h2>Method Definition</h2><div id="settings"></div></section>
  </main>
</div>
<script>
const DATA = {data_json};
function fmt(v,d=3) {{ if (v === null || v === undefined || Number.isNaN(Number(v))) return "NA"; return Number(v).toFixed(d); }}
function esc(s) {{ return String(s).replace(/[&<>"']/g, m => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[m])); }}
function uniq(a) {{ return [...new Set(a)]; }}
function shortSplit(s) {{ return s.split("_")[0]; }}
function fill(el, values, labelFn) {{ const old=el.value; el.innerHTML=values.map(v=>`<option value="${{esc(v)}}">${{esc(labelFn?labelFn(v):v)}}</option>`).join(""); if(values.includes(old)) el.value=old; }}
function selected() {{ const s=document.getElementById("splitSelect").value; const v=document.getElementById("variantSelect").value; return DATA.summary.find(r=>r.split===s && r.variant===v); }}
function metric(k,v) {{ return `<div class="metric"><div class="k">${{esc(k)}}</div><div class="v">${{esc(v)}}</div></div>`; }}
function table(rows, cols) {{ if(!rows.length) return "<p class='note'>No rows.</p>"; return `<table><thead><tr>${{cols.map(c=>`<th>${{esc(c.label)}}</th>`).join("")}}</tr></thead><tbody>` + rows.map(r=>`<tr>${{cols.map(c=>`<td>${{esc(c.f(r))}}</td>`).join("")}}</tr>`).join("") + "</tbody></table>"; }}
function setup() {{
  document.getElementById("inputDim").textContent = DATA.settings.inputDim;
  const splitSel=document.getElementById("splitSelect"), variantSel=document.getElementById("variantSelect");
  fill(splitSel, uniq(DATA.summary.map(r=>r.split)), s=>`${{shortSplit(s)}} - ${{s.replace(shortSplit(s)+"_","")}}`);
  function refreshVariants() {{ fill(variantSel, DATA.summary.filter(r=>r.split===splitSel.value).map(r=>r.variant)); }}
  splitSel.addEventListener("change",()=>{{ refreshVariants(); render(); }});
  variantSel.addEventListener("change",render);
  refreshVariants();
}}
function renderSummary(r) {{
  document.getElementById("metrics").innerHTML = [
    metric("Train acc", fmt(r.train_accuracy)), metric("Val acc", fmt(r.validation_accuracy)),
    metric("Test acc", fmt(r.test_accuracy)), metric("Test BA", fmt(r.test_balanced_accuracy)),
    metric("Test macro F1", fmt(r.test_macro_f1))
  ].join("");
  document.getElementById("usage").innerHTML = `
    <span class="pill">lambda=${{fmt(r.lambda,2)}}</span><span class="pill">seed=${{r.seed}}</span>
    <span class="pill">best epoch=${{r.best_epoch}}</span><span class="pill">${{esc(r.classifier_label_usage)}}</span>
    <span class="pill">${{esc(r.domain_label_usage)}}</span><p class="note">${{esc(r.description)}}</p>`;
}}
function renderTables(r) {{
  const rows=DATA.summary.filter(x=>x.split===r.split);
  document.getElementById("variantTable").innerHTML = table(rows, [
    {{label:"Variant", f:x=>x.variant}}, {{label:"Lambda", f:x=>fmt(x.lambda,2)}}, {{label:"Seed", f:x=>x.seed}},
    {{label:"Train", f:x=>fmt(x.train_accuracy)}}, {{label:"Val", f:x=>fmt(x.validation_accuracy)}},
    {{label:"Test", f:x=>fmt(x.test_accuracy)}}, {{label:"BA", f:x=>fmt(x.test_balanced_accuracy)}},
    {{label:"F1", f:x=>fmt(x.test_macro_f1)}}, {{label:"Lcoral", f:x=>fmt(x.coral_loss_last)}}
  ]);
  const parts=DATA.partitions.filter(x=>x.split===r.split);
  document.getElementById("partitionTable").innerHTML = table(parts, [
    {{label:"Partition", f:x=>x.partition}}, {{label:"N", f:x=>x.n}}, {{label:"Class counts", f:x=>x.class_counts||""}},
    {{label:"Period / days", f:x=>x.period_days||""}}, {{label:"Label usage", f:x=>x.label_usage}}
  ]);
}}
function extent(vals) {{ let min=Math.min(...vals), max=Math.max(...vals); if(!Number.isFinite(min)||!Number.isFinite(max)) return [-1,1]; if(min===max) {{ min-=1; max+=1; }} const pad=(max-min)*0.08; return [min-pad,max+pad]; }}
function scale(v,min,max,a,b) {{ return a+(v-min)/(max-min)*(b-a); }}
function lineChart(rows, series, title) {{
  const w=560,h=300,m={{l:48,r:18,t:32,b:36}}; if(!rows.length) return `<svg viewBox="0 0 ${{w}} ${{h}}"><text x="20" y="30">No curves</text></svg>`;
  const xs=rows.map(r=>Number(r.epoch)); const ys=[]; series.forEach(s=>rows.forEach(r=>{{ if(Number.isFinite(Number(r[s.key]))) ys.push(Number(r[s.key])); }}));
  const [xmin,xmax]=extent(xs), [ymin,ymax]=extent(ys);
  const lines=series.map(s=>`<polyline points="${{rows.filter(r=>Number.isFinite(Number(r[s.key]))).map(r=>`${{scale(Number(r.epoch),xmin,xmax,m.l,w-m.r)}},${{scale(Number(r[s.key]),ymin,ymax,h-m.b,m.t)}}`).join(" ")}}" fill="none" stroke="${{s.color}}" stroke-width="2"/>`).join("");
  const legend=series.map((s,i)=>`<text x="${{m.l+i*150}}" y="${{h-10}}" font-size="11" fill="${{s.color}}">${{s.label}}</text>`).join("");
  return `<svg viewBox="0 0 ${{w}} ${{h}}" width="100%" height="${{h}}"><text x="${{m.l}}" y="22" font-size="12" font-weight="700">${{esc(title)}}</text><line class="axis" x1="${{m.l}}" y1="${{h-m.b}}" x2="${{w-m.r}}" y2="${{h-m.b}}"/><line class="axis" x1="${{m.l}}" y1="${{m.t}}" x2="${{m.l}}" y2="${{h-m.b}}"/>${{lines}}${{legend}}</svg>`;
}}
function scatter(points) {{
  const w=760,h=360,m={{l:45,r:18,t:32,b:36}}; if(!points.length) return "";
  const xs=points.map(p=>Number(p.x)), ys=points.map(p=>Number(p.y)); const [xmin,xmax]=extent(xs), [ymin,ymax]=extent(ys);
  const colors={{air:"#2ca02c", alcohol:"#1f77b4", acetone:"#d62728"}};
  const circles=points.map(p=>{{ const cx=scale(Number(p.x),xmin,xmax,m.l,w-m.r), cy=scale(Number(p.y),ymin,ymax,h-m.b,m.t); const opacity=p.domain==="source"?1:0.45; return `<circle cx="${{cx}}" cy="${{cy}}" r="4.5" fill="${{colors[p.label]||"#555"}}" opacity="${{opacity}}"><title>${{esc(p.label+" | "+p.partition+" | "+p.target_file)}}</title></circle>`; }}).join("");
  return `<svg viewBox="0 0 ${{w}} ${{h}}" width="100%" height="${{h}}"><text x="${{m.l}}" y="22" font-size="12" font-weight="700">PCA projection of selected feature space</text><line class="axis" x1="${{m.l}}" y1="${{h-m.b}}" x2="${{w-m.r}}" y2="${{h-m.b}}"/><line class="axis" x1="${{m.l}}" y1="${{m.t}}" x2="${{m.l}}" y2="${{h-m.b}}"/>${{circles}}<text x="52" y="338" font-size="11" fill="#5f6b7a">darker = source, lighter = target; color = gas class</text></svg>`;
}}
function renderCurvesAndSpace(r) {{
  const rows=DATA.curves.filter(x=>x.split===r.split && x.variant===r.variant);
  document.getElementById("lossChart").innerHTML = lineChart(rows, [
    {{key:"class_loss", label:"class loss", color:"#2166ac"}}, {{key:"coral_loss", label:"coral loss", color:"#e08214"}}, {{key:"total_loss", label:"total", color:"#4d9221"}}
  ], "Loss curves");
  document.getElementById("accChart").innerHTML = lineChart(rows, [
    {{key:"train_accuracy", label:"train", color:"#2166ac"}}, {{key:"validation_accuracy", label:"val", color:"#b2182b"}}, {{key:"test_accuracy", label:"test", color:"#4d9221"}}
  ], "Accuracy curves");
  const points=DATA.features.filter(x=>x.split===r.split && x.variant===r.variant);
  document.getElementById("spaceNote").textContent = points[0] ? points[0].space : "";
  document.getElementById("featureChart").innerHTML = scatter(points);
}}
function renderSettings() {{
  const s=DATA.settings;
  document.getElementById("settings").innerHTML = `<span class="pill">input dim=${{s.inputDim}}</span><span class="pill">latent dim=${{s.latentDim}}</span><span class="pill">classes=${{s.classes.join(", ")}}</span><span class="pill">epochs=${{s.epochs}}</span><span class="pill">lr=${{s.learningRate}}</span><span class="pill">weight decay=${{s.weightDecay}}</span><p class="note">${{esc(s.objective)}}</p><p class="note">${{esc(s.labelUse)}}</p><p class="note">${{esc(s.domainUse)}}</p>`;
}}
function render() {{ const r=selected(); if(!r) return; renderSummary(r); renderTables(r); renderCurvesAndSpace(r); renderSettings(); }}
setup(); render();
</script>
</body>
</html>
"""
    HTML_PATH.write_text(html_text, encoding="utf-8")


def main() -> None:
    torch.set_num_threads(1)
    df = pd.read_csv(base.FEATURE_PATH)
    all_rows = []
    all_splits = []
    for split in base.SPLITS:
        for task_name, classes in base.REPORT_TASKS:
            rows, split_rows = evaluate_split(df, split, task_name, classes)
            all_rows.extend(rows)
            all_splits.extend(split_rows)
            print(f"{split['split']} | {task_name}: {len(rows)} runs")

    results = pd.DataFrame(all_rows)
    summary = choose_best(results)
    partitions = pd.DataFrame(all_splits)

    base.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(ALL_RUNS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    partitions.to_csv(PARTITIONS_PATH, index=False)

    feature_dim = len(base.feature_columns(df[df["label"].isin(base.REPORT_TASKS[0][1])].copy()))
    all_curves = []
    all_features = []
    for i, row in summary.iterrows():
        split = split_by_name(str(row["split"]))
        classes = task_classes(str(row["task"]))
        task, parts = prepare_parts(df, split, classes)
        arrays = make_arrays(task, parts, classes)
        model, curves = train_with_curves(row, arrays, classes)
        for curve in curves:
            curve.update(
                {
                    "split": row["split"],
                    "task": row["task"],
                    "variant": row["variant"],
                    "lambda": float(row["lambda"]),
                    "seed": int(row["seed"]),
                }
            )
        all_curves.extend(curves)
        all_features.extend(feature_rows(str(row["split"]), str(row["task"]), str(row["variant"]), model, parts, arrays))
        print(f"selected {i + 1:02d}/{len(summary)} {row['split']} | {row['variant']}")

    curves = pd.DataFrame(all_curves)
    features = pd.DataFrame(all_features)
    curves.to_csv(CURVES_PATH, index=False)
    features.to_csv(FEATURE_SPACE_PATH, index=False)
    write_flow_svg(feature_dim=feature_dim, latent_dim=16, n_classes=3)
    write_summary_svg(summary)
    write_html(summary, partitions, curves, features, feature_dim)

    print(f"Wrote {ALL_RUNS_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {PARTITIONS_PATH}")
    print(f"Wrote {CURVES_PATH}")
    print(f"Wrote {FEATURE_SPACE_PATH}")
    print(f"Wrote {FLOW_SVG_PATH}")
    print(f"Wrote {SUMMARY_SVG_PATH}")
    print(f"Wrote {HTML_PATH}")


if __name__ == "__main__":
    main()
