from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import run_dann_multi_splits as base


OUT_ROOT = Path(r"D:\thesis")
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures" / "dann_learning"
FEATURE_PATH = TABLE_DIR / "manifest_baseline_as_air_features.csv"

# Official CDAN-style conditional adversarial variants.
# This script is prepared for the thesis data but is not run automatically.
VARIANTS = [
    {"variant": "CDAN-UDA", "use_target_labels": False, "lambdas": base.LAMBDA_VALUES},
    {"variant": "CDAN-semi", "use_target_labels": True, "lambdas": base.LAMBDA_VALUES},
]


class CDANDomainDiscriminator(nn.Module):
    """Domain discriminator for official CDAN-style conditional features.

    Instead of receiving h directly as in DANN, this discriminator receives
    T(h, g) = g outer h, flattened to C*H dimensions.
    """

    def __init__(self, feature_dim: int, n_classes: int) -> None:
        super().__init__()
        cdan_dim = feature_dim * n_classes
        self.net = nn.Sequential(
            nn.Linear(cdan_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, conditioned: torch.Tensor) -> torch.Tensor:
        return self.net(conditioned)


def multilinear_condition(h: torch.Tensor, logits: torch.Tensor, detach_predictions: bool = True) -> torch.Tensor:
    """Build the CDAN multilinear map T(h, g).

    Args:
        h: latent feature h, shape (N, H), here H=16.
        logits: gas classifier logits before softmax, shape (N, C).
        detach_predictions: if True, stop the adversarial loss from updating
            the classifier through g. This follows the common official CDAN
            PyTorch implementation pattern.

    Returns:
        Flattened outer product g x h, shape (N, C*H).
    """

    probabilities = F.softmax(logits, dim=1)
    if detach_predictions:
        probabilities = probabilities.detach()
    outer = torch.bmm(probabilities.unsqueeze(2), h.unsqueeze(1))
    return outer.reshape(h.size(0), -1)


class CDAN(nn.Module):
    """CDAN structure for this e-nose dataset.

    Shared G_f: 54 -> 32 -> 16.
    Gas head G_y: 16 -> C.
    Conditional domain head G_cd: (16*C) -> 32 -> 2.
    """

    def __init__(self, n_features: int, n_classes: int, detach_predictions: bool = True) -> None:
        super().__init__()
        self.feature_extractor = base.FeatureExtractor(n_features)
        self.label_classifier = base.LabelClassifier(n_classes)
        self.domain_discriminator = CDANDomainDiscriminator(feature_dim=16, n_classes=n_classes)
        self.detach_predictions = detach_predictions
        self.feature = self.feature_extractor
        self.classifier = self.label_classifier

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def forward_class(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature_extractor(x)
        return self.label_classifier(h)

    def forward_domain(self, x: torch.Tensor, lambd: float) -> torch.Tensor:
        h = self.feature_extractor(x)
        logits = self.label_classifier(h)
        conditioned = multilinear_condition(h, logits, self.detach_predictions)
        return self.domain_discriminator(base.grad_reverse(conditioned, lambd))

    def domain_no_grl(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature_extractor(x)
        logits = self.label_classifier(h)
        conditioned = multilinear_condition(h, logits, self.detach_predictions)
        return self.domain_discriminator(conditioned)


def predict(model: CDAN, x: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.forward_class(x).argmax(dim=1).cpu().numpy()


def predict_domain(model: CDAN, x: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.domain_no_grl(x).argmax(dim=1).cpu().numpy()


def train_cdan_model(
    x_cls: np.ndarray,
    y_cls: np.ndarray,
    x_source_domain: np.ndarray,
    x_target_domain: np.ndarray,
    x_select: np.ndarray | None,
    y_select: np.ndarray | None,
    n_classes: int,
    lambda_value: float,
    seed: int,
) -> tuple[CDAN, dict[str, float]]:
    base.set_random_seed(seed)
    model = CDAN(x_cls.shape[1], n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY)

    x_cls_t = base.to_tensor(x_cls)
    y_cls_t = base.to_long(y_cls)
    x_select_t = base.to_tensor(x_select) if x_select is not None else None
    y_weight = base.class_weights(y_cls, n_classes)
    if len(x_target_domain) == 0:
        raise ValueError("CDAN domain loss was requested, but x_target_domain is empty.")

    x_dom_t = torch.cat([base.to_tensor(x_source_domain), base.to_tensor(x_target_domain)], dim=0)
    y_dom_t = torch.cat(
        [
            torch.zeros(len(x_source_domain), dtype=torch.long),
            torch.ones(len(x_target_domain), dtype=torch.long),
        ],
        dim=0,
    )

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_score = (-1.0, -1.0, -1.0)
    best_epoch = 0
    last_L = 0.0
    last_Ld = 0.0
    last_backward_loss_scalar = 0.0

    for epoch in range(1, base.EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        class_logits = model.forward_class(x_cls_t)
        L = F.cross_entropy(class_logits, y_cls_t, weight=y_weight)

        domain_logits = model.forward_domain(x_dom_t, lambda_value)
        Ld = F.cross_entropy(domain_logits, y_dom_t)

        backward_loss_scalar = L + Ld
        backward_loss_scalar.backward()
        optimizer.step()

        last_L = float(L.detach().cpu())
        last_Ld = float(Ld.detach().cpu())
        last_backward_loss_scalar = float(backward_loss_scalar.detach().cpu())

        if x_select_t is not None and (epoch % 10 == 0 or epoch == base.EPOCHS):
            select_pred = predict(model, x_select_t)
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
        "L_last": last_L,
        "Ld_cdan_last": last_Ld,
        "backward_loss_scalar_last": last_backward_loss_scalar,
        "feature_extractor_objective": "min L - lambda*Ld_cdan via GRL on T(h,g)",
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
    if len(parts["source_train"]) == 0 or len(parts["target_domain"]) == 0 or len(parts["target_test"]) == 0:
        raise ValueError(f"{split['split']} / {task_name} has empty source_train, target_domain, or target_test.")
    if not set(classes).issubset(set(parts["source_train"]["label"])):
        raise ValueError(f"{split['split']} / {task_name} source_train is missing required classes.")
    if len(parts["target_test"]) and not set(classes).issubset(set(parts["target_test"]["label"])):
        raise ValueError(f"{split['split']} / {task_name} target_test is missing required classes.")

    split_rows = []
    usage_text = {
        "source_train": "gas labels used for classifier loss L",
        "source_val": "labels used only for model selection",
        "target_labeled_train": "gas labels used only by CDAN-semi",
        "target_domain": "gas labels not used; features used for CDAN domain loss Ld_cdan",
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
    target_label_raw = parts["target_labeled_train"][cols].to_numpy(float)
    target_domain_raw = parts["target_domain"][cols].to_numpy(float)
    target_val_raw = parts["target_val"][cols].to_numpy(float)
    target_test_raw = parts["target_test"][cols].to_numpy(float)

    x_ref = source_raw
    source_x, source_val_x, target_label_x, target_domain_x, target_val_x, target_test_x = base.standardize(
        x_ref,
        source_raw,
        source_val_raw,
        target_label_raw,
        target_domain_raw,
        target_val_raw,
        target_test_raw,
    )

    y_source = base.encode(parts["source_train"]["label"].to_numpy(str), classes)
    y_source_val = base.encode(parts["source_val"]["label"].to_numpy(str), classes) if len(parts["source_val"]) else np.array([], dtype=np.int64)
    y_target_label = base.encode(parts["target_labeled_train"]["label"].to_numpy(str), classes) if len(parts["target_labeled_train"]) else np.array([], dtype=np.int64)
    y_target_val = base.encode(parts["target_val"]["label"].to_numpy(str), classes) if len(parts["target_val"]) else np.array([], dtype=np.int64)
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

        if selection_labels == base.NO_VALIDATION_SELECTION:
            candidates = [(base.DEFAULT_LAMBDA, seed) for seed in base.SEEDS]
        else:
            candidates = [(lambda_value, seed) for lambda_value in variant["lambdas"] for seed in base.SEEDS]

        for lambda_value, seed in candidates:
                model, info = train_cdan_model(
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
                train_m = base.metric_dict(y_cls, predict(model, base.to_tensor(x_cls)), len(classes))
                select_m = (
                    {"accuracy": np.nan, "balanced_accuracy": np.nan, "macro_f1": np.nan}
                    if x_select is None
                    else base.metric_dict(y_select, predict(model, base.to_tensor(x_select)), len(classes))
                )
                test_m = base.metric_dict(y_target_test, predict(model, base.to_tensor(target_test_x)), len(classes))

                x_dom = np.vstack([source_x, target_domain_x])
                y_dom = np.concatenate([np.zeros(len(source_x), dtype=int), np.ones(len(target_domain_x), dtype=int)])
                domain_pred = predict_domain(model, base.to_tensor(x_dom))
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
                        "domain_label_usage": "source/target domain labels only; gas labels not used for CDAN domain loss",
                        "model_selection_labels": selection_labels,
                        "best_epoch": int(info["best_epoch"]),
                        "L_last": info["L_last"],
                        "Ld_cdan_last": info["Ld_cdan_last"],
                        "backward_loss_scalar_last": info["backward_loss_scalar_last"],
                        "feature_extractor_objective": info["feature_extractor_objective"],
                        "condition_map": "T(h,g)=softmax(G_y(h)) outer h; shape N x (C*16)",
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


def choose_best(results: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for (_split, _task, _variant), sub in results.groupby(["split", "task", "variant"], sort=False):
        no_validation = sub["model_selection_labels"].eq(base.NO_VALIDATION_SELECTION).all()
        if no_validation:
            fixed = sub[np.isclose(sub["lambda"].astype(float), base.DEFAULT_LAMBDA)]
            if fixed.empty:
                raise ValueError(f"Missing fixed no-validation candidate for {_split} / {_task} / {_variant}.")
            selection_status = "fixed hyperparameters because the split has no validation set; metrics averaged over seeds"
            selected.append(base.aggregate_seed_rows(fixed, selection_status))
        else:
            best_lambda = base.select_lambda_over_seed_means(sub)
            best_rows = sub[np.isclose(sub["lambda"].astype(float), best_lambda)]
            selection_status = "lambda selected by mean validation macro-F1, balanced accuracy, and accuracy over seeds"
            selected.append(base.aggregate_seed_rows(best_rows, selection_status))
    out = pd.DataFrame(selected).reset_index(drop=True)
    no_validation = out["model_selection_labels"].eq(base.NO_VALIDATION_SELECTION)
    out.loc[no_validation, ["validation_accuracy", "validation_balanced_accuracy", "validation_macro_f1"]] = np.nan
    best = out.groupby(["split", "task"])["test_accuracy"].transform("max")
    out["judgment"] = out["selection_status"]
    out.loc[np.isclose(out["test_accuracy"], best), "judgment"] += "; highest selected CDAN test accuracy in split"
    out.loc[out["train_test_accuracy_gap"] > 0.25, "judgment"] += "; high train-test gap"
    out.loc[out["domain_discriminator_accuracy"].between(0.45, 0.60, inclusive="both"), "judgment"] += "; domain near-invariant"
    out.loc[out["domain_discriminator_accuracy"] > 0.75, "judgment"] += "; domain still separable"
    return out


def main() -> None:
    """Train CDAN variants and write CSV/SVG reports.

    This function is intentionally not run during code generation. Execute this
    script manually only when the machine has enough power/time for training.
    """

    torch.set_num_threads(1)
    df = pd.read_csv(FEATURE_PATH)
    all_rows: list[dict] = []
    split_rows: list[dict] = []
    for split in base.SPLITS:
        for task_name, classes in base.TASKS:
            rows, parts = evaluate_split(df, split, task_name, classes)
            all_rows.extend(rows)
            split_rows.extend(parts)

    results = pd.DataFrame(all_rows)
    splits = pd.DataFrame(split_rows)
    summary = choose_best(results)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(TABLE_DIR / "cdan_multi_split_all_runs.csv", index=False)
    summary.to_csv(TABLE_DIR / "cdan_multi_split_selected_summary.csv", index=False)
    splits.to_csv(TABLE_DIR / "cdan_multi_split_partitions.csv", index=False)
    base.write_svg(summary, FIG_DIR / "cdan_multi_split_summary.svg")


if __name__ == "__main__":
    main()
