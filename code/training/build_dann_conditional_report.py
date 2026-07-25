from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_dann_interactive_report as base  # noqa: E402
import run_dann_multi_splits as dann  # noqa: E402


OUT_ROOT = Path(r"D:\thesis")
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures" / "dann_learning"
FEATURE_PATH = TABLE_DIR / "manifest_baseline_as_air_features.csv"
OLD_SUMMARY_PATH = TABLE_DIR / "dann_multi_split_selected_summary.csv"
OLD_CURVES_PATH = TABLE_DIR / "dann_interactive_training_curves.csv"
OLD_FEATURE_SPACE_PATH = TABLE_DIR / "dann_interactive_feature_space.csv"
PARTITION_PATH = TABLE_DIR / "dann_multi_split_partitions.csv"

NEW_SUMMARY_PATH = TABLE_DIR / "dann_conditional_selected_summary.csv"
NEW_CURVES_PATH = TABLE_DIR / "dann_conditional_training_curves.csv"
NEW_FEATURE_SPACE_PATH = TABLE_DIR / "dann_conditional_feature_space.csv"
COMPARE_PATH = TABLE_DIR / "dann_conditional_old_vs_new_comparison.csv"
MIXING_PATH = TABLE_DIR / "dann_conditional_mixing_diagnostics.csv"
HTML_PATH = FIG_DIR / "dann_9feature_cdann_report.html"
SVG_PATH = FIG_DIR / "dann_9feature_cdann_summary.svg"

ALIGN_STRENGTHS = [0.05, 0.2, 0.5]
DEFAULT_LAMBDA = 0.2
DEFAULT_ALPHA = 0.2
DEFAULT_SEED = 7
LOG_EVERY = 5


def one_hot(y: np.ndarray, n_classes: int) -> torch.Tensor:
    return F.one_hot(torch.tensor(y, dtype=torch.long), num_classes=n_classes).float()


def weighted_mean(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denom = weights.sum(dim=0)
    if bool((denom <= 0).any().detach().cpu()):
        raise ValueError("Cannot compute class-wise centroid with zero class weight.")
    return weights.T @ features / denom[:, None]


def conditional_alignment_loss(
    model: dann.DANN,
    x_source: torch.Tensor,
    y_source: np.ndarray,
    x_target: torch.Tensor,
    target_label_probs: torch.Tensor | None,
    n_classes: int,
) -> torch.Tensor:
    if len(x_target) == 0:
        raise ValueError("C-DANN centroid alignment was requested, but x_target is empty.")
    source_h = model.feature(x_source)
    target_h = model.feature(x_target)
    source_w = one_hot(y_source, n_classes).to(source_h.device)
    if target_label_probs is None:
        with torch.no_grad():
            target_label_probs = F.softmax(model.classifier(target_h), dim=1)
    source_mean = weighted_mean(source_h, source_w)
    target_mean = weighted_mean(target_h, target_label_probs.to(target_h.device))
    return ((source_mean - target_mean) ** 2).mean()


def train_conditional(row: pd.Series, arrays: dict, classes: list[str], align_strength: float) -> tuple[dann.DANN, list[dict], dict]:
    seed = int(row["seed"])
    lambda_value = float(row["lambda"])
    dann.set_random_seed(seed)

    x_cls, y_cls, classifier_label_usage = base.classifier_data(arrays, row)
    x_select, y_select, selection_labels = base.selection_data(arrays, row)
    x_source = arrays["x"]["source_train"]
    y_source = arrays["y"]["source_train"]
    x_target = arrays["x"]["target_domain"]
    x_test = arrays["x"]["target_test"]
    y_test = arrays["y"]["target_test"]
    if len(x_target) == 0:
        raise ValueError("C-DANN requires non-empty target_domain features for domain loss and centroid alignment.")

    x_centroid_target = x_target
    target_probs = None
    if row["classifier_label_usage"] == "source labels + selected target labels":
        if len(arrays["y"]["target_labeled_train"]) == 0:
            raise ValueError("Semi C-DANN requested target labels, but target_labeled_train is empty.")
        x_centroid_target = arrays["x"]["target_labeled_train"]
        target_probs = one_hot(arrays["y"]["target_labeled_train"], len(classes))

    model = dann.DANN(x_cls.shape[1], len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=dann.LEARNING_RATE, weight_decay=dann.WEIGHT_DECAY)

    x_cls_t = dann.to_tensor(x_cls)
    y_cls_t = dann.to_long(y_cls)
    y_weight = dann.class_weights(y_cls, len(classes))
    x_source_t = dann.to_tensor(x_source)
    x_target_t = dann.to_tensor(x_target)
    x_centroid_target_t = dann.to_tensor(x_centroid_target)
    x_dom_t = torch.cat([x_source_t, x_target_t], dim=0)
    y_dom_t = torch.cat([torch.zeros(len(x_source), dtype=torch.long), torch.ones(len(x_target), dtype=torch.long)], dim=0)

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_score = (-1.0, -1.0, -1e9)
    best_epoch = 0
    curves = []
    last_info = {}

    for epoch in range(1, dann.EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        class_logits = model.forward_class(x_cls_t)
        L = F.cross_entropy(class_logits, y_cls_t, weight=y_weight)
        domain_logits = model.forward_domain(x_dom_t, lambda_value)
        Ld = F.cross_entropy(domain_logits, y_dom_t)
        Lc = conditional_alignment_loss(model, x_source_t, y_source, x_centroid_target_t, target_probs, len(classes))
        backward_loss_scalar = L + Ld + align_strength * Lc
        backward_loss_scalar.backward()
        optimizer.step()

        if epoch % LOG_EVERY == 0 or epoch == 1 or epoch == dann.EPOCHS:
            train_m = dann.metric_dict(y_cls, base.predict(model, x_cls), len(classes))
            select_m = (
                {"accuracy": np.nan, "balanced_accuracy": np.nan, "macro_f1": np.nan}
                if x_select is None
                else dann.metric_dict(y_select, base.predict(model, x_select), len(classes))
            )
            test_m = dann.metric_dict(y_test, base.predict(model, x_test), len(classes))
            dom_acc = base.domain_accuracy(model, x_source, x_target)
            curve = {
                "epoch": epoch,
                "L": float(L.detach().cpu()),
                "Ld": float(Ld.detach().cpu()),
                "Lc": float(Lc.detach().cpu()),
                "backward_loss_scalar": float(backward_loss_scalar.detach().cpu()),
                "class_loss": float(L.detach().cpu()),
                "domain_loss": float(Ld.detach().cpu()),
                "alignment_loss": float(Lc.detach().cpu()),
                "total_loss": float(backward_loss_scalar.detach().cpu()),
                "train_accuracy": train_m["accuracy"],
                "validation_accuracy": select_m["accuracy"],
                "test_accuracy": test_m["accuracy"],
                "domain_discriminator_accuracy": dom_acc,
            }
            curves.append(curve)
            mixing_proxy = -abs(dom_acc - 0.5) if np.isfinite(dom_acc) else -1.0
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
                    "Lc_last": curve["Lc"],
                    "backward_loss_scalar_last": curve["backward_loss_scalar"],
                    "class_loss_last": curve["class_loss"],
                    "domain_loss_last": curve["domain_loss"],
                    "alignment_loss_last": curve["alignment_loss"],
                }
            elif x_select is None and epoch == dann.EPOCHS:
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
                    "Lc_last": curve["Lc"],
                    "backward_loss_scalar_last": curve["backward_loss_scalar"],
                    "class_loss_last": curve["class_loss"],
                    "domain_loss_last": curve["domain_loss"],
                    "alignment_loss_last": curve["alignment_loss"],
                }

    model.load_state_dict(best_state)
    last_info["best_epoch"] = best_epoch
    last_info["classifier_label_usage"] = classifier_label_usage
    last_info["model_selection_labels"] = selection_labels
    return model, curves, last_info


def mixing_metrics(feature_space: pd.DataFrame, method_group: str) -> pd.DataFrame:
    rows = []
    keys = ["split", "task", "variant", "label"]
    for key, sub in feature_space.groupby(keys, dropna=False):
        domains = set(sub["domain"].dropna().astype(str))
        if not {"source", "target"}.issubset(domains):
            continue
        pts = sub[["x", "y"]].to_numpy(float)
        src = sub[sub["domain"].eq("source")][["x", "y"]].to_numpy(float)
        tgt = sub[sub["domain"].eq("target")][["x", "y"]].to_numpy(float)
        pooled = float(np.nanstd(pts, axis=0).mean())
        if pooled <= 1e-9:
            raise ValueError("Feature-space diagnostic has near-zero pooled spread; inspect latent coordinates.")
        centroid_distance = float(np.linalg.norm(src.mean(axis=0) - tgt.mean(axis=0)) / pooled)
        k = min(5, len(sub) - 1)
        mixed = []
        labels = sub["domain"].to_numpy(str)
        if k > 0:
            d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
            np.fill_diagonal(d2, np.inf)
            nn = np.argsort(d2, axis=1)[:, :k]
            for i in range(len(sub)):
                mixed.append(float(np.mean(labels[nn[i]] != labels[i])))
        neighbor_mixing = float(np.mean(mixed)) if mixed else np.nan
        rows.append(
            {
                "method_group": method_group,
                "split": key[0],
                "task": key[1],
                "variant": key[2],
                "label": key[3],
                "centroid_distance_norm": centroid_distance,
                "neighbor_opposite_domain_rate": neighbor_mixing,
                "n_source": len(src),
                "n_target": len(tgt),
            }
        )
    return pd.DataFrame(rows)


def selection_label(split: dict, arrays: dict) -> str:
    selection = str(split["selection"])
    if selection == "target_validation_labels":
        if len(arrays["x"]["target_val"]) == 0:
            raise ValueError(f"{split['split']} declares target validation selection but target_val is empty.")
        return "target validation labels"
    if selection == "source_validation_labels":
        if len(arrays["x"]["source_val"]) == 0:
            raise ValueError(f"{split['split']} declares source validation selection but source_val is empty.")
        return "source validation labels"
    if selection == dann.NO_VALIDATION_SELECTION:
        return dann.NO_VALIDATION_SELECTION
    raise ValueError(f"Unknown split selection policy: {selection}")


def candidate_grid(has_validation: bool) -> list[tuple[float, float, int]]:
    if not has_validation:
        return [(DEFAULT_LAMBDA, DEFAULT_ALPHA, int(seed)) for seed in dann.SEEDS]
    return [
        (float(lambda_value), float(align_strength), int(seed))
        for lambda_value in dann.LAMBDA_VALUES
        for align_strength in ALIGN_STRENGTHS
        for seed in dann.SEEDS
    ]


def pseudo_dann_row(use_target_labels: bool, lambda_value: float, seed: int, select_label: str) -> pd.Series:
    return pd.Series(
        {
            "lambda": float(lambda_value),
            "seed": int(seed),
            "classifier_label_usage": "source labels + selected target labels" if use_target_labels else "source labels only",
            "model_selection_labels": select_label,
        }
    )


def aggregate_seed_rows(seed_rows: list[pd.Series], selection_status: str) -> pd.Series:
    if not seed_rows:
        raise ValueError("Cannot aggregate an empty seed group.")
    frame = pd.DataFrame([row.to_dict() for row in seed_rows]).sort_values("seed").reset_index(drop=True)
    row = frame.iloc[0].copy()
    seeds = [int(seed) for seed in sorted(frame["seed"].unique())]
    row["seed"] = seeds[0]
    row["representative_seed"] = seeds[0]
    row["seed_values"] = ",".join(str(seed) for seed in seeds)
    row["seed_count"] = len(seeds)
    row["seed_policy"] = dann.SEED_POLICY
    row["selection_status"] = selection_status
    row["judgment"] = selection_status
    for col in dann.SEED_MEAN_COLUMNS:
        if col in frame:
            row[col] = frame[col].mean(skipna=True)
    for col in dann.SEED_STD_COLUMNS:
        if col in frame:
            row[f"{col}_std"] = frame[col].std(skipna=True, ddof=dann.protocol.SAMPLE_STD_DDOF)
    for col in dann.SEED_RANGE_COLUMNS:
        if col in frame:
            row[f"{col}_min"] = frame[col].min(skipna=True)
            row[f"{col}_max"] = frame[col].max(skipna=True)
    return row


def selected_candidate_indices(candidates: list[tuple[pd.Series, list[dict], dann.DANN, float]], has_validation: bool) -> list[int]:
    frame = pd.DataFrame([candidate[0].to_dict() for candidate in candidates])
    if has_validation:
        grouped = (
            frame.groupby(["lambda", "conditional_alignment_strength"], sort=False)
            .agg(
                validation_macro_f1=("validation_macro_f1", "mean"),
                validation_balanced_accuracy=("validation_balanced_accuracy", "mean"),
                validation_accuracy=("validation_accuracy", "mean"),
            )
            .reset_index()
        )
        if grouped[["validation_macro_f1", "validation_balanced_accuracy", "validation_accuracy"]].isna().any().any():
            raise ValueError("Validation metrics are required for C-DANN hyperparameter selection.")
        best = grouped.sort_values(
            ["validation_macro_f1", "validation_balanced_accuracy", "validation_accuracy", "lambda", "conditional_alignment_strength"],
            ascending=[False, False, False, True, True],
        ).iloc[0]
        mask = np.isclose(frame["lambda"].astype(float), float(best["lambda"])) & np.isclose(
            frame["conditional_alignment_strength"].astype(float),
            float(best["conditional_alignment_strength"]),
        )
        return frame.index[mask].tolist()
    mask = np.isclose(frame["lambda"].astype(float), DEFAULT_LAMBDA) & np.isclose(
        frame["conditional_alignment_strength"].astype(float),
        DEFAULT_ALPHA,
    )
    return frame.index[mask].tolist()


def train_improved() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(FEATURE_PATH)
    summary_rows = []
    all_curves = []
    all_features = []

    variants = [
        ("DANN-UDA", "C-DANN-UDA", False),
        ("DANN-semi", "C-DANN-semi", True),
    ]
    jobs = []
    for split in dann.SPLITS:
        for task_name, classes in dann.REPORT_TASKS:
            for old_variant, new_variant, use_target_labels in variants:
                jobs.append((split, task_name, classes, old_variant, new_variant, use_target_labels))

    for i, (split, task_name, classes, old_variant, new_variant, use_target_labels) in enumerate(jobs):
        task, parts = base.prepare_parts(df, split, classes)
        arrays = base.make_arrays(task, parts, classes)
        if use_target_labels and len(arrays["x"]["target_labeled_train"]) == 0:
            continue
        candidates = []
        select_label = selection_label(split, arrays)
        has_validation = select_label != dann.NO_VALIDATION_SELECTION
        print(f"{i + 1:02d}/{len(jobs)} {split['split']} | {task_name} | {new_variant}")
        for lambda_value, align_strength, seed in candidate_grid(has_validation):
            row_for_training = pseudo_dann_row(use_target_labels, lambda_value, seed, select_label)
            model, curves, info = train_conditional(row_for_training, arrays, classes, align_strength)
            x_cls, y_cls, classifier_label_usage = base.classifier_data(arrays, row_for_training)
            summary_row = pd.Series(
                {
                    "split": split["split"],
                    "description": split["description"],
                    "task": task_name,
                    "variant": new_variant,
                    "lambda": lambda_value,
                    "seed": seed,
                    "classifier_train_n": len(y_cls),
                    "source_train_n": len(arrays["y"]["source_train"]),
                    "source_val_n": len(arrays["y"]["source_val"]),
                    "target_labeled_train_n": len(arrays["y"]["target_labeled_train"]),
                    "target_domain_n": len(arrays["x"]["target_domain"]),
                    "target_validation_n": len(arrays["y"]["target_val"]),
                    "target_test_n": len(arrays["y"]["target_test"]),
                    "domain_label_usage": "source/target domain labels only; gas labels not used for domain loss",
                    "feature_extractor_objective": "min L - lambda*Ld + alpha*Lc via GRL and class-wise centroid alignment",
                }
            )
            summary_row["classifier_label_usage"] = info["classifier_label_usage"]
            summary_row["model_selection_labels"] = info["model_selection_labels"]
            summary_row["conditional_alignment_strength"] = align_strength
            summary_row["selection_status"] = "validation-selected best hyperparameters" if has_validation else "no validation; fixed default hyperparameters"
            for metric, value in info.items():
                if metric not in {"classifier_label_usage", "model_selection_labels"}:
                    summary_row[metric] = value
            summary_row["train_test_accuracy_gap"] = info["train_accuracy"] - info["test_accuracy"]
            summary_row["judgment"] = summary_row["selection_status"]
            candidate = (summary_row, curves, model, align_strength)
            candidates.append(candidate)

        selected_indices = selected_candidate_indices(candidates, has_validation)
        if not selected_indices:
            raise ValueError(f"Missing selected C-DANN candidates for {split['split']} / {task_name} / {new_variant}.")
        selection_status = (
            "lambda and alpha selected by mean validation macro-F1, balanced accuracy, and accuracy over seeds"
            if has_validation
            else "fixed hyperparameters because the split has no validation set; metrics averaged over seeds"
        )
        best_row = aggregate_seed_rows([candidates[index][0] for index in selected_indices], selection_status)
        representative_index = min(selected_indices, key=lambda index: int(candidates[index][0]["seed"]))
        _representative_row, best_curves, best_model, best_align = candidates[representative_index]
        summary_rows.append(best_row.to_dict())
        for curve in best_curves:
            curve.update(
                {
                    "split": best_row["split"],
                    "task": best_row["task"],
                    "variant": best_row["variant"],
                    "lambda": float(best_row["lambda"]),
                    "seed": int(best_row["seed"]),
                    "conditional_alignment_strength": best_align,
                }
            )
        all_curves.extend(best_curves)
        all_features.extend(
            base.build_feature_rows(str(best_row["split"]), str(best_row["task"]), str(best_row["variant"]), best_model, parts, arrays)
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(all_curves), pd.DataFrame(all_features)


def hide_summary_pseudo_validation(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    no_validation = summary["model_selection_labels"].eq(dann.NO_VALIDATION_SELECTION)
    for col in ["validation_accuracy", "validation_balanced_accuracy", "validation_macro_f1"]:
        if col in summary:
            summary.loc[no_validation, col] = np.nan
    return summary


def compare_old_new(old_summary: pd.DataFrame, new_summary: pd.DataFrame, old_mix: pd.DataFrame, new_mix: pd.DataFrame) -> pd.DataFrame:
    pairs = {"C-DANN-UDA": "DANN-UDA", "C-DANN-semi": "DANN-semi"}
    rows = []
    old_mix_avg = old_mix.groupby(["split", "task", "variant"]).agg(
        old_centroid_distance=("centroid_distance_norm", "mean"),
        old_neighbor_mixing=("neighbor_opposite_domain_rate", "mean"),
    )
    new_mix_avg = new_mix.groupby(["split", "task", "variant"]).agg(
        new_centroid_distance=("centroid_distance_norm", "mean"),
        new_neighbor_mixing=("neighbor_opposite_domain_rate", "mean"),
    )
    for _, new in new_summary.iterrows():
        old_variant = pairs[str(new["variant"])]
        old = old_summary[
            old_summary["split"].eq(new["split"])
            & old_summary["task"].eq(new["task"])
            & old_summary["variant"].eq(old_variant)
        ].iloc[0]
        old_m = old_mix_avg.loc[(old["split"], old["task"], old["variant"])]
        new_m = new_mix_avg.loc[(new["split"], new["task"], new["variant"])]
        test_delta = float(new["test_accuracy"]) - float(old["test_accuracy"])
        centroid_delta = float(new_m["new_centroid_distance"]) - float(old_m["old_centroid_distance"])
        mixing_delta = float(new_m["new_neighbor_mixing"]) - float(old_m["old_neighbor_mixing"])
        rows.append(
            {
                "split": new["split"],
                "task": new["task"],
                "old_variant": old_variant,
                "new_variant": new["variant"],
                "old_test_accuracy": old["test_accuracy"],
                "new_test_accuracy": new["test_accuracy"],
                "test_accuracy_delta": test_delta,
                "old_domain_accuracy": old["domain_discriminator_accuracy"],
                "new_domain_accuracy": new["domain_discriminator_accuracy"],
                "old_centroid_distance": old_m["old_centroid_distance"],
                "new_centroid_distance": new_m["new_centroid_distance"],
                "centroid_distance_delta": centroid_delta,
                "old_neighbor_mixing": old_m["old_neighbor_mixing"],
                "new_neighbor_mixing": new_m["new_neighbor_mixing"],
                "neighbor_mixing_delta": mixing_delta,
            }
        )
    return pd.DataFrame(rows)


def svg_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def fmt_pct(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{100 * float(value):.1f}%"


def fmt_delta(value: object, signed: bool = True) -> str:
    if pd.isna(value):
        return "NA"
    sign = "+" if signed and float(value) > 0 else ""
    return f"{sign}{float(value):.3f}"


def write_comparison_svg(comparison: pd.DataFrame) -> None:
    rows = comparison.copy().sort_values(["split", "task", "old_variant"]).reset_index(drop=True)
    row_h = 29
    header_h = 170
    width = 1800
    height = header_h + 44 + max(len(rows), 1) * row_h + 110
    col_x = [35, 300, 610, 815, 990, 1165, 1335, 1505, 1665]
    headers = [
        "Split",
        "Task",
        "Variant",
        "Test old",
        "Test C-DANN",
        "Delta",
        "Domain old/new",
        "Centroid delta",
        "Mixing delta",
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="35" y="44" font-family="Arial, Helvetica, sans-serif" font-size="28" font-weight="700" fill="#111827">9-feature DANN vs C-DANN Summary</text>',
        '<text x="35" y="78" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="#4b5563">Features: 9 feature families x 6 sensors = 54-dimensional input vector per trial. C-DANN adds class-conditional latent centroid alignment.</text>',
        '<text x="35" y="105" font-family="Arial, Helvetica, sans-serif" font-size="15" fill="#4b5563">Positive mixing delta and negative centroid delta indicate better source-target mixing in the plotted latent space.</text>',
        '<rect x="25" y="135" width="1750" height="38" rx="6" fill="#f3f4f6" stroke="#d1d5db"/>',
    ]
    for x, header in zip(col_x, headers):
        parts.append(
            f'<text x="{x}" y="160" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="700" fill="#111827">{svg_escape(header)}</text>'
        )

    y = header_h + 30
    for idx, row in rows.iterrows():
        fill = "#ffffff" if idx % 2 == 0 else "#f9fafb"
        parts.append(f'<rect x="25" y="{y - 19}" width="1750" height="{row_h}" fill="{fill}" stroke="#eef2f7"/>')
        values = [
            row["split"],
            row["task"],
            f'{row["old_variant"]} -> {row["new_variant"]}',
            fmt_pct(row["old_test_accuracy"]),
            fmt_pct(row["new_test_accuracy"]),
            fmt_delta(row["test_accuracy_delta"]),
            f'{fmt_pct(row["old_domain_accuracy"])} / {fmt_pct(row["new_domain_accuracy"])}',
            fmt_delta(row["centroid_distance_delta"]),
            fmt_delta(row["neighbor_mixing_delta"]),
        ]
        for x, value in zip(col_x, values):
            parts.append(
                f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#111827">{svg_escape(value)}</text>'
            )
        y += row_h

    parts += [
        '<text x="35" y="{}" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#374151">This table reports raw deltas only; it does not apply threshold-based pass/fail labels.</text>'.format(height - 48),
        "</svg>",
    ]
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    torch.set_num_threads(1)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    old_summary = pd.read_csv(OLD_SUMMARY_PATH)
    old_curves = pd.read_csv(OLD_CURVES_PATH)
    old_features = pd.read_csv(OLD_FEATURE_SPACE_PATH)
    partitions = pd.read_csv(PARTITION_PATH)

    new_summary, new_curves, new_features = train_improved()
    new_summary = hide_summary_pseudo_validation(new_summary)
    old_mix = mixing_metrics(old_features, "old")
    new_mix = mixing_metrics(new_features, "conditional")
    comparison = compare_old_new(old_summary, new_summary, old_mix, new_mix)

    new_summary.to_csv(NEW_SUMMARY_PATH, index=False)
    new_curves.to_csv(NEW_CURVES_PATH, index=False)
    new_features.to_csv(NEW_FEATURE_SPACE_PATH, index=False)
    pd.concat([old_mix, new_mix], ignore_index=True).to_csv(MIXING_PATH, index=False)
    comparison.to_csv(COMPARE_PATH, index=False)
    write_comparison_svg(comparison)

    combined_summary = pd.concat([old_summary, new_summary], ignore_index=True, sort=False)
    combined_curves = pd.concat([old_curves, new_curves], ignore_index=True, sort=False)
    combined_features = pd.concat([old_features, new_features], ignore_index=True, sort=False)
    combined_summary = combined_summary[combined_summary["task"].eq("three_class_with_air")].reset_index(drop=True)
    combined_curves = combined_curves[combined_curves["task"].eq("three_class_with_air")].reset_index(drop=True)
    combined_features = combined_features[combined_features["task"].eq("three_class_with_air")].reset_index(drop=True)
    partitions = partitions[partitions["task"].eq("three_class_with_air")].reset_index(drop=True)
    base.HTML_PATH = HTML_PATH
    base.write_html(combined_summary, partitions, combined_curves, combined_features)

    print("\nComparison")
    print(
        comparison[
            [
                "split",
                "task",
                "old_variant",
                "new_variant",
                "old_test_accuracy",
                "new_test_accuracy",
                "test_accuracy_delta",
                "old_centroid_distance",
                "new_centroid_distance",
                "centroid_distance_delta",
                "old_neighbor_mixing",
                "new_neighbor_mixing",
                "neighbor_mixing_delta",
            ]
        ].to_string(index=False)
    )
    print("\nWrote")
    print(HTML_PATH)
    print(SVG_PATH)
    print(COMPARE_PATH)
    print(MIXING_PATH)


if __name__ == "__main__":
    main()
