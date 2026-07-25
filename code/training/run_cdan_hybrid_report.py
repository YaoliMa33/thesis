from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_dann_interactive_report as report  # noqa: E402
import run_cdan_multi_splits as cdan  # noqa: E402
import run_dann_multi_splits as base  # noqa: E402


OUT_ROOT = Path(r"D:\thesis")
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures" / "dann_learning"
FEATURE_PATH = TABLE_DIR / "manifest_baseline_as_air_features.csv"
HTML_PATH = FIG_DIR / "dann_9feature_cdann_report.html"

SUMMARY_PATH = TABLE_DIR / "cdan_hybrid_selected_summary.csv"
ALL_RUNS_PATH = TABLE_DIR / "cdan_hybrid_all_runs.csv"
CURVES_PATH = TABLE_DIR / "cdan_hybrid_training_curves.csv"
FEATURE_SPACE_PATH = TABLE_DIR / "cdan_hybrid_feature_space.csv"

ALIGN_STRENGTHS = [0.05, 0.2, 0.5]
DEFAULT_LAMBDA = 0.2
DEFAULT_ALPHA = 0.2
DEFAULT_SEED = 7
LOG_EVERY = 5


VARIANTS = [
    {
        "variant": "CDAN-UDA",
        "use_target_labels": False,
        "use_dann_domain": False,
        "use_cdan_domain": True,
        "use_centroid": False,
        "lambdas": base.LAMBDA_VALUES,
        "alphas": [np.nan],
    },
    {
        "variant": "CDAN-semi",
        "use_target_labels": True,
        "use_dann_domain": False,
        "use_cdan_domain": True,
        "use_centroid": False,
        "lambdas": base.LAMBDA_VALUES,
        "alphas": [np.nan],
    },
    {
        "variant": "CDAN+C-DANN-UDA",
        "use_target_labels": False,
        "use_dann_domain": True,
        "use_cdan_domain": True,
        "use_centroid": True,
        "lambdas": base.LAMBDA_VALUES,
        "alphas": ALIGN_STRENGTHS,
    },
    {
        "variant": "CDAN+C-DANN-semi",
        "use_target_labels": True,
        "use_dann_domain": True,
        "use_cdan_domain": True,
        "use_centroid": True,
        "lambdas": base.LAMBDA_VALUES,
        "alphas": ALIGN_STRENGTHS,
    },
]


def load_reference_hyperparameters() -> pd.DataFrame:
    frames = []
    for path in [TABLE_DIR / "dann_multi_split_selected_summary.csv", TABLE_DIR / "dann_conditional_selected_summary.csv"]:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def reference_variant_name(variant_name: str) -> str:
    if variant_name == "CDAN-UDA":
        return "DANN-UDA"
    if variant_name == "CDAN-semi":
        return "DANN-semi"
    if variant_name == "CDAN+C-DANN-UDA":
        return "C-DANN-UDA"
    if variant_name == "CDAN+C-DANN-semi":
        return "C-DANN-semi"
    raise KeyError(variant_name)


def candidate_hyperparameters(variant: dict, has_validation: bool) -> list[tuple[float, float, int]]:
    if not has_validation:
        alpha_value = DEFAULT_ALPHA if variant["use_centroid"] else np.nan
        return [(DEFAULT_LAMBDA, alpha_value, int(seed)) for seed in base.SEEDS]
    return [
        (float(lambda_value), float(alpha_value) if pd.notna(alpha_value) else np.nan, int(seed))
        for lambda_value in variant["lambdas"]
        for alpha_value in variant["alphas"]
        for seed in base.SEEDS
    ]


class CDANHybrid(nn.Module):
    """CDAN plus optional ordinary DANN domain head and centroid alignment."""

    def __init__(self, n_features: int, n_classes: int, detach_predictions: bool = True) -> None:
        super().__init__()
        self.feature_extractor = base.FeatureExtractor(n_features)
        self.label_classifier = base.LabelClassifier(n_classes)
        self.dann_domain_classifier = base.DomainClassifier()
        self.cdan_domain_discriminator = cdan.CDANDomainDiscriminator(feature_dim=16, n_classes=n_classes)
        self.detach_predictions = detach_predictions

        # Compatibility aliases for shared report helpers.
        self.feature = self.feature_extractor
        self.classifier = self.label_classifier

    def forward_class(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature_extractor(x)
        return self.label_classifier(h)

    def forward_dann_domain(self, x: torch.Tensor, lambd: float) -> torch.Tensor:
        h = self.feature_extractor(x)
        return self.dann_domain_classifier(base.grad_reverse(h, lambd))

    def forward_cdan_domain(self, x: torch.Tensor, lambd: float) -> torch.Tensor:
        h = self.feature_extractor(x)
        logits = self.label_classifier(h)
        conditioned = cdan.multilinear_condition(h, logits, self.detach_predictions)
        return self.cdan_domain_discriminator(base.grad_reverse(conditioned, lambd))

    def cdan_domain_no_grl(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature_extractor(x)
        logits = self.label_classifier(h)
        conditioned = cdan.multilinear_condition(h, logits, self.detach_predictions)
        return self.cdan_domain_discriminator(conditioned)


def one_hot(y: np.ndarray, n_classes: int) -> torch.Tensor:
    return F.one_hot(torch.tensor(y, dtype=torch.long), num_classes=n_classes).float()


def weighted_mean(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denom = weights.sum(dim=0)
    if bool((denom <= 0).any().detach().cpu()):
        raise ValueError("Cannot compute class-wise centroid with zero class weight.")
    return weights.T @ features / denom[:, None]


def conditional_alignment_loss(
    model: CDANHybrid,
    x_source: torch.Tensor,
    y_source: np.ndarray,
    x_target: torch.Tensor,
    target_label_probs: torch.Tensor | None,
    n_classes: int,
) -> torch.Tensor:
    if len(x_target) == 0:
        raise ValueError("CDAN+C-DANN centroid alignment was requested, but x_target is empty.")
    source_h = model.feature_extractor(x_source)
    target_h = model.feature_extractor(x_target)
    source_w = one_hot(y_source, n_classes).to(source_h.device)
    if target_label_probs is None:
        with torch.no_grad():
            target_label_probs = F.softmax(model.label_classifier(target_h), dim=1)
    target_w = target_label_probs.to(target_h.device)
    source_mean = weighted_mean(source_h, source_w)
    target_mean = weighted_mean(target_h, target_w)
    return ((source_mean - target_mean) ** 2).mean()


def predict(model: CDANHybrid, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.forward_class(base.to_tensor(x)).argmax(dim=1).cpu().numpy()


def domain_accuracy(model: CDANHybrid, x_source: np.ndarray, x_target: np.ndarray) -> float:
    if len(x_target) == 0:
        return float("nan")
    x_dom = np.vstack([x_source, x_target])
    y_dom = np.concatenate([np.zeros(len(x_source), dtype=int), np.ones(len(x_target), dtype=int)])
    model.eval()
    with torch.no_grad():
        pred = model.cdan_domain_no_grl(base.to_tensor(x_dom)).argmax(dim=1).cpu().numpy()
    return float((pred == y_dom).mean())


def selection_data(arrays: dict) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    selection = str(arrays["selection"])
    if selection == "target_validation_labels":
        if len(arrays["x"]["target_val"]) == 0:
            raise ValueError("Split declares target validation selection but target_val is empty.")
        return arrays["x"]["target_val"], arrays["y"]["target_val"], "target validation labels"
    if selection == "source_validation_labels":
        if len(arrays["x"]["source_val"]) == 0:
            raise ValueError("Split declares source validation selection but source_val is empty.")
        return arrays["x"]["source_val"], arrays["y"]["source_val"], "source validation labels"
    if selection == base.NO_VALIDATION_SELECTION:
        return None, None, base.NO_VALIDATION_SELECTION
    raise ValueError(f"Unknown split selection policy: {selection}")


def classifier_data(arrays: dict, use_target_labels: bool) -> tuple[np.ndarray, np.ndarray, str]:
    x_source = arrays["x"]["source_train"]
    y_source = arrays["y"]["source_train"]
    x_target = arrays["x"]["target_labeled_train"]
    y_target = arrays["y"]["target_labeled_train"]
    if use_target_labels:
        if len(x_target) == 0:
            raise ValueError("Target-label training was requested, but target_labeled_train is empty.")
        return np.vstack([x_source, x_target]), np.concatenate([y_source, y_target]), "source labels + selected target labels"
    return x_source, y_source, "source labels only"


def train_candidate(
    arrays: dict,
    classes: list[str],
    variant: dict,
    lambda_value: float,
    alpha_value: float,
    seed: int,
) -> tuple[CDANHybrid, list[dict], dict]:
    base.set_random_seed(seed)

    x_cls, y_cls, classifier_label_usage = classifier_data(arrays, bool(variant["use_target_labels"]))
    x_select, y_select, selection_labels = selection_data(arrays)
    x_source = arrays["x"]["source_train"]
    y_source = arrays["y"]["source_train"]
    x_target = arrays["x"]["target_domain"]
    x_test = arrays["x"]["target_test"]
    y_test = arrays["y"]["target_test"]
    if len(x_target) == 0:
        raise ValueError("CDAN requires non-empty target_domain features for domain loss.")

    x_centroid_target = x_target
    target_probs = None
    if variant["use_centroid"] and variant["use_target_labels"]:
        if len(arrays["y"]["target_labeled_train"]) == 0:
            raise ValueError("Semi CDAN+C-DANN requested target labels, but target_labeled_train is empty.")
        x_centroid_target = arrays["x"]["target_labeled_train"]
        target_probs = one_hot(arrays["y"]["target_labeled_train"], len(classes))

    model = CDANHybrid(x_cls.shape[1], len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY)

    x_cls_t = base.to_tensor(x_cls)
    y_cls_t = base.to_long(y_cls)
    y_weight = base.class_weights(y_cls, len(classes))
    x_source_t = base.to_tensor(x_source)
    x_target_t = base.to_tensor(x_target)
    x_centroid_target_t = base.to_tensor(x_centroid_target)
    x_dom_t = torch.cat([x_source_t, x_target_t], dim=0)
    y_dom_t = torch.cat([torch.zeros(len(x_source), dtype=torch.long), torch.ones(len(x_target), dtype=torch.long)], dim=0)

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_score = (-1.0, -1.0)
    best_epoch = 0
    curves: list[dict] = []
    last_info: dict = {}

    for epoch in range(1, base.EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        class_logits = model.forward_class(x_cls_t)
        L = F.cross_entropy(class_logits, y_cls_t, weight=y_weight)
        Ld = torch.tensor(0.0)
        Ld_cdan = torch.tensor(0.0)
        Lc = torch.tensor(0.0)
        backward_loss_scalar = L

        if variant["use_dann_domain"]:
            dann_logits = model.forward_dann_domain(x_dom_t, lambda_value)
            Ld = F.cross_entropy(dann_logits, y_dom_t)
            backward_loss_scalar = backward_loss_scalar + Ld
        if variant["use_cdan_domain"]:
            cdan_logits = model.forward_cdan_domain(x_dom_t, lambda_value)
            Ld_cdan = F.cross_entropy(cdan_logits, y_dom_t)
            backward_loss_scalar = backward_loss_scalar + Ld_cdan
        if variant["use_centroid"]:
            Lc = conditional_alignment_loss(model, x_source_t, y_source, x_centroid_target_t, target_probs, len(classes))
            backward_loss_scalar = backward_loss_scalar + float(alpha_value) * Lc

        backward_loss_scalar.backward()
        optimizer.step()

        if epoch % LOG_EVERY == 0 or epoch == 1 or epoch == base.EPOCHS:
            train_m = base.metric_dict(y_cls, predict(model, x_cls), len(classes))
            select_m = (
                {"accuracy": np.nan, "balanced_accuracy": np.nan, "macro_f1": np.nan}
                if x_select is None
                else base.metric_dict(y_select, predict(model, x_select), len(classes))
            )
            test_m = base.metric_dict(y_test, predict(model, x_test), len(classes))
            dom_acc = domain_accuracy(model, x_source, x_target)
            curve = {
                "epoch": epoch,
                "L": float(L.detach().cpu()),
                "Ld": float(Ld.detach().cpu()),
                "Ld_cdan": float(Ld_cdan.detach().cpu()),
                "Lc": float(Lc.detach().cpu()),
                "backward_loss_scalar": float(backward_loss_scalar.detach().cpu()),
                "train_accuracy": train_m["accuracy"],
                "validation_accuracy": select_m["accuracy"],
                "test_accuracy": test_m["accuracy"],
                "domain_discriminator_accuracy": dom_acc,
            }
            curves.append(curve)
            score = (select_m["macro_f1"], select_m["balanced_accuracy"], select_m["accuracy"])
            if x_select is not None and score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                last_info = {
                    "train_accuracy": train_m["accuracy"],
                    "validation_accuracy": select_m["accuracy"],
                    "test_accuracy": test_m["accuracy"],
                    "train_balanced_accuracy": train_m["balanced_accuracy"],
                    "validation_balanced_accuracy": select_m["balanced_accuracy"],
                    "test_balanced_accuracy": test_m["balanced_accuracy"],
                    "train_macro_f1": train_m["macro_f1"],
                    "validation_macro_f1": select_m["macro_f1"],
                    "test_macro_f1": test_m["macro_f1"],
                    "domain_discriminator_accuracy": dom_acc,
                    "L_last": curve["L"],
                    "Ld_last": curve["Ld"],
                    "Ld_cdan_last": curve["Ld_cdan"],
                    "Lc_last": curve["Lc"],
                    "backward_loss_scalar_last": curve["backward_loss_scalar"],
                }
            elif x_select is None and epoch == base.EPOCHS:
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                last_info = {
                    "train_accuracy": train_m["accuracy"],
                    "validation_accuracy": np.nan,
                    "test_accuracy": test_m["accuracy"],
                    "train_balanced_accuracy": train_m["balanced_accuracy"],
                    "validation_balanced_accuracy": np.nan,
                    "test_balanced_accuracy": test_m["balanced_accuracy"],
                    "train_macro_f1": train_m["macro_f1"],
                    "validation_macro_f1": np.nan,
                    "test_macro_f1": test_m["macro_f1"],
                    "domain_discriminator_accuracy": dom_acc,
                    "L_last": curve["L"],
                    "Ld_last": curve["Ld"],
                    "Ld_cdan_last": curve["Ld_cdan"],
                    "Lc_last": curve["Lc"],
                    "backward_loss_scalar_last": curve["backward_loss_scalar"],
                }

    model.load_state_dict(best_state)
    last_info["best_epoch"] = best_epoch
    last_info["classifier_label_usage"] = classifier_label_usage
    last_info["model_selection_labels"] = selection_labels
    return model, curves, last_info


def aggregate_seed_rows(seed_rows: list[dict], selection_status: str) -> dict:
    if not seed_rows:
        raise ValueError("Cannot aggregate an empty seed group.")
    frame = pd.DataFrame(seed_rows).sort_values("seed").reset_index(drop=True)
    row = frame.iloc[0].copy()
    seeds = [int(seed) for seed in sorted(frame["seed"].unique())]
    row["seed"] = seeds[0]
    row["representative_seed"] = seeds[0]
    row["seed_values"] = ",".join(str(seed) for seed in seeds)
    row["seed_count"] = len(seeds)
    row["seed_policy"] = base.SEED_POLICY
    row["selection_status"] = selection_status
    row["judgment"] = selection_status
    for col in base.SEED_MEAN_COLUMNS:
        if col in frame:
            row[col] = frame[col].mean(skipna=True)
    for col in base.SEED_STD_COLUMNS:
        if col in frame:
            row[f"{col}_std"] = frame[col].std(skipna=True, ddof=base.protocol.SAMPLE_STD_DDOF)
    for col in base.SEED_RANGE_COLUMNS:
        if col in frame:
            row[f"{col}_min"] = frame[col].min(skipna=True)
            row[f"{col}_max"] = frame[col].max(skipna=True)
    return row.to_dict()


def selected_candidate_indices(candidates: list[tuple[dict, list[dict], CDANHybrid]], has_validation: bool, variant: dict) -> list[int]:
    frame = pd.DataFrame([candidate[0] for candidate in candidates])
    alpha_key = frame["conditional_alignment_strength"].where(frame["conditional_alignment_strength"].notna(), -1.0).astype(float)
    frame = frame.assign(_alpha_key=alpha_key)
    default_alpha_key = DEFAULT_ALPHA if variant["use_centroid"] else -1.0
    if has_validation:
        grouped = (
            frame.groupby(["lambda", "_alpha_key"], sort=False)
            .agg(
                validation_macro_f1=("validation_macro_f1", "mean"),
                validation_balanced_accuracy=("validation_balanced_accuracy", "mean"),
                validation_accuracy=("validation_accuracy", "mean"),
            )
            .reset_index()
        )
        if grouped[["validation_macro_f1", "validation_balanced_accuracy", "validation_accuracy"]].isna().any().any():
            raise ValueError("Validation metrics are required for CDAN hyperparameter selection.")
        best = grouped.sort_values(
            ["validation_macro_f1", "validation_balanced_accuracy", "validation_accuracy", "lambda", "_alpha_key"],
            ascending=[False, False, False, True, True],
        ).iloc[0]
        mask = np.isclose(frame["lambda"].astype(float), float(best["lambda"])) & np.isclose(
            frame["_alpha_key"].astype(float),
            float(best["_alpha_key"]),
        )
        return frame.index[mask].tolist()
    mask = np.isclose(frame["lambda"].astype(float), DEFAULT_LAMBDA) & np.isclose(
        frame["_alpha_key"].astype(float),
        default_alpha_key,
    )
    return frame.index[mask].tolist()


def validate_task_parts(split_name: str, task_name: str, parts: dict[str, pd.DataFrame], classes: list[str]) -> None:
    if len(parts["source_train"]) == 0 or len(parts["target_domain"]) == 0 or len(parts["target_test"]) == 0:
        raise ValueError(f"{split_name} / {task_name} has empty source_train, target_domain, or target_test.")
    if not set(classes).issubset(set(parts["source_train"]["label"])):
        raise ValueError(f"{split_name} / {task_name} source_train is missing required classes.")
    if len(parts["target_test"]) and not set(classes).issubset(set(parts["target_test"]["label"])):
        raise ValueError(f"{split_name} / {task_name} target_test is missing required classes.")


def run_training() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(FEATURE_PATH)
    summary_rows = []
    all_run_rows = []
    all_curves = []
    all_features = []

    for split in base.SPLITS:
        for task_name, classes in base.REPORT_TASKS:
            task, parts = report.prepare_parts(df, split, classes)
            validate_task_parts(split["split"], task_name, parts, classes)
            arrays = report.make_arrays(task, parts, classes)
            arrays["selection"] = split["selection"]
            has_validation = selection_data(arrays)[2] != base.NO_VALIDATION_SELECTION
            for variant in VARIANTS:
                if variant["use_target_labels"] and len(arrays["x"]["target_labeled_train"]) == 0:
                    continue
                print(f"{split['split']} | {task_name} | {variant['variant']}")
                candidates = []
                for lambda_value, alpha_value, seed in candidate_hyperparameters(variant, has_validation):
                    model, curves, info = train_candidate(arrays, classes, variant, float(lambda_value), float(alpha_value) if pd.notna(alpha_value) else 0.0, seed)
                    row = {
                        "split": split["split"],
                        "description": split["description"],
                        "task": task_name,
                        "variant": variant["variant"],
                        "lambda": float(lambda_value),
                        "conditional_alignment_strength": np.nan if pd.isna(alpha_value) else float(alpha_value),
                        "seed": int(seed),
                        "classifier_label_usage": info["classifier_label_usage"],
                        "domain_label_usage": "source/target domain labels only; gas labels not used for domain losses",
                        "model_selection_labels": info["model_selection_labels"],
                        "selection_status": "validation-selected best hyperparameters" if has_validation else "no validation; fixed default hyperparameters",
                        "best_epoch": int(info["best_epoch"]),
                        "feature_extractor_objective": (
                            "min L - lambda*Ld_cdan via GRL on T(h,g)"
                            if not variant["use_centroid"]
                            else "min L - lambda*Ld - lambda*Ld_cdan + alpha*Lc via GRL and centroid alignment"
                        ),
                        "condition_map": "T(h,g)=softmax(G_y(h)) outer h; shape N x (C*16)",
                        "classifier_train_n": len(classifier_data(arrays, bool(variant["use_target_labels"]))[1]),
                        "source_train_n": len(arrays["y"]["source_train"]),
                        "source_val_n": len(arrays["y"]["source_val"]),
                        "target_labeled_train_n": len(arrays["y"]["target_labeled_train"]),
                        "target_domain_n": len(arrays["x"]["target_domain"]),
                        "target_validation_n": len(arrays["y"]["target_val"]),
                        "target_test_n": len(arrays["y"]["target_test"]),
                        "train_test_accuracy_gap": info["train_accuracy"] - info["test_accuracy"],
                    }
                    row.update(info)
                    all_run_rows.append(row.copy())
                    candidates.append((row, curves, model))

                selected_indices = selected_candidate_indices(candidates, has_validation, variant)
                if not selected_indices:
                    raise ValueError(f"Missing selected CDAN candidates for {split['split']} / {task_name} / {variant['variant']}.")
                selection_status = (
                    "lambda and alpha selected by mean validation macro-F1, balanced accuracy, and accuracy over seeds"
                    if has_validation and variant["use_centroid"]
                    else "lambda selected by mean validation macro-F1, balanced accuracy, and accuracy over seeds"
                    if has_validation
                    else "fixed hyperparameters because the split has no validation set; metrics averaged over seeds"
                )
                best_row = aggregate_seed_rows([candidates[index][0] for index in selected_indices], selection_status)
                representative_index = min(selected_indices, key=lambda index: int(candidates[index][0]["seed"]))
                _representative_row, best_curves, best_model = candidates[representative_index]
                summary_rows.append(best_row)
                for curve in best_curves:
                    curve.update(
                        {
                            "split": best_row["split"],
                            "task": best_row["task"],
                            "variant": best_row["variant"],
                            "lambda": best_row["lambda"],
                            "conditional_alignment_strength": best_row["conditional_alignment_strength"],
                            "seed": best_row["seed"],
                        }
                    )
                all_curves.extend(best_curves)
                all_features.extend(report.build_feature_rows(best_row["split"], best_row["task"], best_row["variant"], best_model, parts, arrays))

    summary = pd.DataFrame(summary_rows)
    all_runs = pd.DataFrame(all_run_rows)
    curves = pd.DataFrame(all_curves)
    features = pd.DataFrame(all_features)

    if not summary.empty:
        summary["judgment"] = summary["selection_status"]

    return all_runs, summary, curves, features


def hide_summary_pseudo_validation(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    no_validation = summary["model_selection_labels"].eq(base.NO_VALIDATION_SELECTION)
    for col in ["validation_accuracy", "validation_balanced_accuracy", "validation_macro_f1"]:
        if col in summary:
            summary.loc[no_validation, col] = np.nan
    return summary


def update_html(summary: pd.DataFrame, curves: pd.DataFrame, features: pd.DataFrame) -> None:
    old_summary_paths = [
        TABLE_DIR / "dann_multi_split_selected_summary.csv",
        TABLE_DIR / "dann_conditional_selected_summary.csv",
    ]
    old_curve_paths = [
        TABLE_DIR / "dann_interactive_training_curves.csv",
        TABLE_DIR / "dann_conditional_training_curves.csv",
    ]
    old_feature_paths = [
        TABLE_DIR / "dann_interactive_feature_space.csv",
        TABLE_DIR / "dann_conditional_feature_space.csv",
    ]
    combined_summary = pd.concat([*(pd.read_csv(p) for p in old_summary_paths if p.exists()), summary], ignore_index=True, sort=False)
    combined_curves = pd.concat([*(pd.read_csv(p) for p in old_curve_paths if p.exists()), curves], ignore_index=True, sort=False)
    combined_features = pd.concat([*(pd.read_csv(p) for p in old_feature_paths if p.exists()), features], ignore_index=True, sort=False)
    partitions = pd.read_csv(TABLE_DIR / "dann_multi_split_partitions.csv")
    combined_summary = combined_summary[combined_summary["task"].eq("three_class_with_air")].reset_index(drop=True)
    combined_curves = combined_curves[combined_curves["task"].eq("three_class_with_air")].reset_index(drop=True)
    combined_features = combined_features[combined_features["task"].eq("three_class_with_air")].reset_index(drop=True)
    partitions = partitions[partitions["task"].eq("three_class_with_air")].reset_index(drop=True)
    report.HTML_PATH = HTML_PATH
    report.write_html(combined_summary, partitions, combined_curves, combined_features)


def main() -> None:
    torch.set_num_threads(1)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_runs, summary, curves, features = run_training()
    summary = hide_summary_pseudo_validation(summary)
    all_runs.to_csv(ALL_RUNS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    curves.to_csv(CURVES_PATH, index=False)
    features.to_csv(FEATURE_SPACE_PATH, index=False)
    update_html(summary, curves, features)
    print(ALL_RUNS_PATH)
    print(SUMMARY_PATH)
    print(CURVES_PATH)
    print(FEATURE_SPACE_PATH)
    print(HTML_PATH)


if __name__ == "__main__":
    main()
