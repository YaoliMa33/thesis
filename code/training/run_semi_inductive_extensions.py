"""Strict inductive Semi feature-fusion extensions.

This runner evaluates three exploratory extensions without exposing target test
features or labels during training/model selection:

1. probability-ensemble: validation-selected soft voting between raw-54D LR
   and CDAN+C-DANN feature-fusion LR;
2. target-weighted: select beta in Ls + beta*Lt for the neural gas loss, then
   classify [x, h] with fixed-C L2 Logistic Regression;
3. elastic-net: validation-select C and l1_ratio for multinomial Logistic
   Regression on [x, h].

The old transductive artifacts are not modified. These experiments remain
exploratory because their design followed inspection of earlier test results;
the strict split removes code-level test exposure but cannot make the reused
dataset a historically pristine test set.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_cdan_hybrid_report as existing_hybrid
import run_dann_multi_splits as existing_dann
import run_unified_da_source_target_suite as suite
import shared_experiment_protocol as protocol


ROOT = Path(r"D:\thesis")
FEATURE_PATH = ROOT / "tables" / "manifest_baseline_as_air_features.csv"
OUTPUT_DIR = ROOT / "tables" / "semi_inductive_extensions"
EXPERIMENTS = ("probability-ensemble", "target-weighted", "elastic-net")
SEEDS = tuple(protocol.MODEL_SEEDS)
EPOCHS = int(protocol.NEURAL_EPOCHS)
EVAL_EVERY = 10
LR_C = 1.0
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
BETA_GRID = (0.5, 1.0, 2.0)
ELASTIC_C_GRID = (0.1, 1.0, 10.0)
L1_RATIO_GRID = (0.25, 0.5, 0.75)
ALIGNMENT_STRENGTH = 0.2
CENTROID_STRENGTH = 0.2


def target_range(start: int, end: int) -> callable:
    return lambda df: protocol.target_days(df, start, end)


def target_from(start: int) -> callable:
    return lambda df: protocol.target_days(df, start)


STRICT_SPLITS = [
    {
        "split": "SI1_train_I_D1_val_D2_test_D3plus",
        "description": "Period I + labeled target Day1 train; target Day2 validation; Day3+ test.",
        "target_train": target_range(1, 1),
        "target_val": target_range(2, 2),
        "target_test": target_from(3),
    },
    {
        "split": "SI2_train_I_D1_val_D2_3_test_D4plus",
        "description": "Period I + labeled target Day1 train; target Day2-3 validation; Day4+ test.",
        "target_train": target_range(1, 1),
        "target_val": target_range(2, 3),
        "target_test": target_from(4),
    },
    {
        "split": "SI3_train_I_D1_2_val_D3_5_test_D6plus",
        "description": "Period I + labeled target Day1-2 train; target Day3-5 validation; Day6+ test.",
        "target_train": target_range(1, 2),
        "target_val": target_range(3, 5),
        "target_test": target_from(6),
    },
    {
        "split": "SI4_train_I_D1_3_val_D4_5_test_D6plus",
        "description": "Period I + labeled target Day1-3 train; target Day4-5 validation; Day6+ test.",
        "target_train": target_range(1, 3),
        "target_val": target_range(4, 5),
        "target_test": target_from(6),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--splits", nargs="+", default=[item["split"] for item in STRICT_SPLITS])
    return parser.parse_args()


def output_paths() -> dict[str, Path]:
    return {
        "tuning": OUTPUT_DIR / "tuning.csv",
        "runs": OUTPUT_DIR / "runs.csv",
        "predictions": OUTPUT_DIR / "predictions.csv",
        "summary": OUTPUT_DIR / "summary.csv",
        "protocol": OUTPUT_DIR / "protocol.json",
    }


def save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def read_records(path: Path) -> list[dict]:
    return pd.read_csv(path).to_dict("records") if path.exists() else []


def strict_parts(df: pd.DataFrame, split: dict) -> dict[str, pd.DataFrame]:
    parts = {
        "source_train": df.loc[protocol.source_period_mask(df)].copy(),
        "target_train": df.loc[split["target_train"](df)].copy(),
        "target_val": df.loc[split["target_val"](df)].copy(),
        "target_test": df.loc[split["target_test"](df)].copy(),
    }
    if any(frame.empty for frame in parts.values()):
        raise ValueError(f"{split['split']} contains an empty partition.")
    names = list(parts)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = set(parts[left].index) & set(parts[right].index)
            if overlap:
                raise ValueError(f"{split['split']} overlaps {left} and {right} by {len(overlap)} rows.")
    expected = set(suite.CLASSES)
    for key in ("source_train", "target_train", "target_val", "target_test"):
        present = set(parts[key]["label"].astype(str))
        if present != expected:
            raise ValueError(f"{split['split']} {key} classes are {sorted(present)}, expected {sorted(expected)}.")
    return parts


def encode(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return suite.encode(frame, columns)


def arrays_from_parts(parts: dict[str, pd.DataFrame], columns: list[str], final_refit: bool) -> dict:
    source_x, source_y = encode(parts["source_train"], columns)
    target_train = parts["target_train"]
    if final_refit:
        target_train = pd.concat([target_train, parts["target_val"]], axis=0)
    target_x, target_y = encode(target_train, columns)
    val_x, val_y = encode(parts["target_val"], columns)
    test_x, test_y = encode(parts["target_test"], columns)
    reference = np.concatenate([source_x, target_x], axis=0)
    raw_arrays = {
        "source": source_x,
        "target_train": target_x,
        "validation": val_x,
        "test": test_x,
    }
    standardized = suite.fit_standardizer(reference, raw_arrays)
    if set(parts["target_test"].index) & set(target_train.index):
        raise ValueError("Target test rows entered the final training frame.")
    return {
        "x": standardized,
        "y": {
            "source": source_y,
            "target_train": target_y,
            "validation": val_y,
            "test": test_y,
        },
    }


def l2_lr() -> Pipeline:
    return Pipeline([
        ("standardize", StandardScaler()),
        ("classifier", LogisticRegression(
            C=LR_C, class_weight="balanced", solver="lbfgs", max_iter=3000,
        )),
    ])


def elastic_lr(c_value: float, l1_ratio: float, seed: int) -> Pipeline:
    return Pipeline([
        ("standardize", StandardScaler()),
        ("classifier", LogisticRegression(
            C=c_value,
            class_weight="balanced",
            l1_ratio=l1_ratio,
            solver="saga",
            max_iter=10000,
            random_state=seed,
            tol=1e-4,
        )),
    ])


def latent(model: suite.UnifiedModel, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.features(existing_dann.to_tensor(x)).cpu().numpy()


def fusion_features(model: suite.UnifiedModel, x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, latent(model, x)], axis=1)


def class_weights(y: np.ndarray) -> torch.Tensor:
    return existing_dann.class_weights(y, len(suite.CLASSES))


def train_hybrid(
    arrays: dict,
    seed: int,
    target_beta: float | None,
    select_on_validation: bool,
    fixed_epochs: int | None = None,
) -> tuple[suite.UnifiedModel, dict]:
    existing_dann.set_random_seed(seed)
    model = suite.UnifiedModel(54, len(suite.CLASSES), suite.ARCHITECTURES["one-layer"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=protocol.NEURAL_LEARNING_RATE,
        weight_decay=protocol.NEURAL_WEIGHT_DECAY,
    )
    xs = existing_dann.to_tensor(arrays["x"]["source"])
    ys = existing_dann.to_long(arrays["y"]["source"])
    xt = existing_dann.to_tensor(arrays["x"]["target_train"])
    yt = existing_dann.to_long(arrays["y"]["target_train"])
    xv, yv = arrays["x"]["validation"], arrays["y"]["validation"]
    source_weight, target_weight = class_weights(arrays["y"]["source"]), class_weights(arrays["y"]["target_train"])
    domain_y = torch.cat([torch.zeros(len(xs), dtype=torch.long), torch.ones(len(xt), dtype=torch.long)])
    epochs_to_run = int(fixed_epochs or EPOCHS)
    best_state = copy.deepcopy(model.state_dict())
    best_score = (-1.0, -1.0, -1.0)
    best_epoch = 0
    last = {}
    for epoch in range(1, epochs_to_run + 1):
        model.train()
        optimizer.zero_grad()
        source_logits, target_logits = model.class_logits(xs), model.class_logits(xt)
        if target_beta is None:
            combined_logits = torch.cat([source_logits, target_logits], dim=0)
            combined_y = torch.cat([ys, yt], dim=0)
            combined_weight = class_weights(np.concatenate([arrays["y"]["source"], arrays["y"]["target_train"]]))
            class_loss = F.cross_entropy(combined_logits, combined_y, weight=combined_weight)
            source_loss = F.cross_entropy(source_logits, ys, weight=source_weight)
            target_loss = F.cross_entropy(target_logits, yt, weight=target_weight)
        else:
            source_loss = F.cross_entropy(source_logits, ys, weight=source_weight)
            target_loss = F.cross_entropy(target_logits, yt, weight=target_weight)
            class_loss = source_loss + target_beta * target_loss
        source_h, target_h = model.features(xs), model.features(xt)
        domain_h = torch.cat([source_h, target_h], dim=0)
        dann_loss = F.cross_entropy(model.dann_domain_logits(domain_h, ALIGNMENT_STRENGTH), domain_y)
        cdan_loss = F.cross_entropy(model.cdan_domain_logits(domain_h, ALIGNMENT_STRENGTH), domain_y)
        centroid = suite.centroid_loss(
            model,
            source_h,
            arrays["y"]["source"],
            target_h,
            arrays["y"]["target_train"],
        )
        total = class_loss + dann_loss + cdan_loss + CENTROID_STRENGTH * centroid
        total.backward()
        optimizer.step()
        last = {
            "classification_loss": float(class_loss.detach()),
            "source_loss": float(source_loss.detach()),
            "target_loss": float(target_loss.detach()),
            "dann_loss": float(dann_loss.detach()),
            "cdan_loss": float(cdan_loss.detach()),
            "centroid_loss": float(centroid.detach()),
            "total_loss": float(total.detach()),
        }
        if select_on_validation and (epoch % EVAL_EVERY == 0 or epoch == epochs_to_run):
            model.eval()
            with torch.no_grad():
                pred = model.class_logits(existing_dann.to_tensor(xv)).argmax(dim=1).cpu().numpy()
            selected = suite.metric(yv, pred)
            score = (selected["macro_f1"], selected["balanced_accuracy"], selected["accuracy"])
            if score > best_score:
                best_score, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        elif not select_on_validation and epoch == epochs_to_run:
            best_epoch, best_state = epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, {**last, "best_epoch": best_epoch}


def metric_from_probabilities(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return suite.metric(y, probabilities.argmax(axis=1))


def score_key(frame: pd.DataFrame, parameter_columns: list[str]) -> pd.Series:
    grouped = frame.groupby(parameter_columns, as_index=False)[
        ["validation_macro_f1", "validation_balanced_accuracy", "validation_accuracy"]
    ].mean()
    return grouped.sort_values(
        ["validation_macro_f1", "validation_balanced_accuracy", "validation_accuracy", *parameter_columns],
        ascending=[False, False, False, *([True] * len(parameter_columns))],
        kind="mergesort",
    ).iloc[0]


def append_predictions(
    prediction_records: list[dict],
    experiment: str,
    split_name: str,
    seed: int,
    test_frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    for position, (index, sample) in enumerate(test_frame.iterrows()):
        prediction_records.append({
            "experiment": experiment,
            "split": split_name,
            "seed": seed,
            "sample_index": int(index),
            "sample_id": sample.get("sample_id", index),
            "period": sample.get("period", ""),
            "day_label": sample.get("day_label", ""),
            "batch": sample.get("batch", ""),
            "true_label": suite.CLASSES[int(y_true[position])],
            "predicted_label": suite.CLASSES[int(y_pred[position])],
            "correct": bool(y_true[position] == y_pred[position]),
        })


def run_base_models(
    arrays: dict,
    tuning_records: list[dict],
    split_name: str,
    paths: dict[str, Path],
) -> dict[int, tuple[suite.UnifiedModel, int]]:
    models = {}
    for seed in SEEDS:
        model, info = train_hybrid(arrays, seed, None, True)
        models[seed] = (model, int(info["best_epoch"]))
        tuning_records.append({
            "experiment": "base-feature-fusion",
            "split": split_name,
            "seed": seed,
            "parameter": "fixed CDAN+C-DANN strengths",
            "parameter_value": f"alignment={ALIGNMENT_STRENGTH};centroid={CENTROID_STRENGTH}",
            "best_epoch": int(info["best_epoch"]),
            "validation_accuracy": np.nan,
            "validation_balanced_accuracy": np.nan,
            "validation_macro_f1": np.nan,
        })
        save(pd.DataFrame(tuning_records), paths["tuning"])
    return models


def run_probability_ensemble(
    split: dict,
    parts: dict,
    train_arrays: dict,
    final_arrays: dict,
    base_models: dict,
    tuning_records: list[dict],
    run_records: list[dict],
    prediction_records: list[dict],
    paths: dict[str, Path],
) -> None:
    split_name = split["split"]
    tuning = []
    for seed, (model, _) in base_models.items():
        x_train = np.concatenate([train_arrays["x"]["source"], train_arrays["x"]["target_train"]])
        y_train = np.concatenate([train_arrays["y"]["source"], train_arrays["y"]["target_train"]])
        raw_lr, fusion_lr = l2_lr(), l2_lr()
        raw_lr.fit(x_train, y_train)
        fusion_lr.fit(fusion_features(model, x_train), y_train)
        raw_prob = raw_lr.predict_proba(train_arrays["x"]["validation"])
        fusion_prob = fusion_lr.predict_proba(fusion_features(model, train_arrays["x"]["validation"]))
        for alpha in ALPHA_GRID:
            metrics = metric_from_probabilities(
                train_arrays["y"]["validation"], alpha * raw_prob + (1.0 - alpha) * fusion_prob
            )
            row = {
                "experiment": "probability-ensemble", "split": split_name, "seed": seed,
                "alpha": alpha, "validation_accuracy": metrics["accuracy"],
                "validation_balanced_accuracy": metrics["balanced_accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
            }
            tuning.append(row); tuning_records.append(row)
    save(pd.DataFrame(tuning_records), paths["tuning"])
    selected_alpha = float(score_key(pd.DataFrame(tuning), ["alpha"])["alpha"])
    for seed, (_, best_epoch) in base_models.items():
        model, _ = train_hybrid(final_arrays, seed, None, False, best_epoch)
        x_train = np.concatenate([final_arrays["x"]["source"], final_arrays["x"]["target_train"]])
        y_train = np.concatenate([final_arrays["y"]["source"], final_arrays["y"]["target_train"]])
        raw_lr, fusion_lr = l2_lr(), l2_lr()
        raw_lr.fit(x_train, y_train); fusion_lr.fit(fusion_features(model, x_train), y_train)
        raw_prob = raw_lr.predict_proba(final_arrays["x"]["test"])
        fusion_prob = fusion_lr.predict_proba(fusion_features(model, final_arrays["x"]["test"]))
        combined = selected_alpha * raw_prob + (1.0 - selected_alpha) * fusion_prob
        pred = combined.argmax(axis=1); metrics = suite.metric(final_arrays["y"]["test"], pred)
        run_records.append({
            "experiment": "probability-ensemble", "split": split_name, "seed": seed,
            "selected_alpha": selected_alpha, "selected_beta": np.nan, "selected_C": LR_C,
            "selected_l1_ratio": 0.0, "best_epoch": best_epoch,
            "test_accuracy": metrics["accuracy"], "test_balanced_accuracy": metrics["balanced_accuracy"],
            "test_macro_f1": metrics["macro_f1"],
        })
        append_predictions(prediction_records, "probability-ensemble", split_name, seed, parts["target_test"], final_arrays["y"]["test"], pred)
        save(pd.DataFrame(run_records), paths["runs"]); save(pd.DataFrame(prediction_records), paths["predictions"])


def run_target_weighted(
    split: dict,
    parts: dict,
    train_arrays: dict,
    final_arrays: dict,
    tuning_records: list[dict],
    run_records: list[dict],
    prediction_records: list[dict],
    paths: dict[str, Path],
) -> None:
    split_name = split["split"]
    tuning, selected_models = [], {}
    x_train = np.concatenate([train_arrays["x"]["source"], train_arrays["x"]["target_train"]])
    y_train = np.concatenate([train_arrays["y"]["source"], train_arrays["y"]["target_train"]])
    for beta in BETA_GRID:
        for seed in SEEDS:
            model, info = train_hybrid(train_arrays, seed, beta, True)
            classifier = l2_lr().fit(fusion_features(model, x_train), y_train)
            pred = classifier.predict(fusion_features(model, train_arrays["x"]["validation"]))
            metrics = suite.metric(train_arrays["y"]["validation"], pred)
            row = {
                "experiment": "target-weighted", "split": split_name, "seed": seed,
                "beta": beta, "best_epoch": int(info["best_epoch"]),
                "validation_accuracy": metrics["accuracy"],
                "validation_balanced_accuracy": metrics["balanced_accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
            }
            tuning.append(row); tuning_records.append(row)
            selected_models[(beta, seed)] = int(info["best_epoch"])
            save(pd.DataFrame(tuning_records), paths["tuning"])
    selected_beta = float(score_key(pd.DataFrame(tuning), ["beta"])["beta"])
    x_final = np.concatenate([final_arrays["x"]["source"], final_arrays["x"]["target_train"]])
    y_final = np.concatenate([final_arrays["y"]["source"], final_arrays["y"]["target_train"]])
    for seed in SEEDS:
        best_epoch = selected_models[(selected_beta, seed)]
        model, _ = train_hybrid(final_arrays, seed, selected_beta, False, best_epoch)
        classifier = l2_lr().fit(fusion_features(model, x_final), y_final)
        pred = classifier.predict(fusion_features(model, final_arrays["x"]["test"]))
        metrics = suite.metric(final_arrays["y"]["test"], pred)
        run_records.append({
            "experiment": "target-weighted", "split": split_name, "seed": seed,
            "selected_alpha": np.nan, "selected_beta": selected_beta, "selected_C": LR_C,
            "selected_l1_ratio": 0.0, "best_epoch": best_epoch,
            "test_accuracy": metrics["accuracy"], "test_balanced_accuracy": metrics["balanced_accuracy"],
            "test_macro_f1": metrics["macro_f1"],
        })
        append_predictions(prediction_records, "target-weighted", split_name, seed, parts["target_test"], final_arrays["y"]["test"], pred)
        save(pd.DataFrame(run_records), paths["runs"]); save(pd.DataFrame(prediction_records), paths["predictions"])


def run_elastic_net(
    split: dict,
    parts: dict,
    train_arrays: dict,
    final_arrays: dict,
    base_models: dict,
    tuning_records: list[dict],
    run_records: list[dict],
    prediction_records: list[dict],
    paths: dict[str, Path],
) -> None:
    split_name = split["split"]
    tuning = []
    x_train = np.concatenate([train_arrays["x"]["source"], train_arrays["x"]["target_train"]])
    y_train = np.concatenate([train_arrays["y"]["source"], train_arrays["y"]["target_train"]])
    for seed, (model, _) in base_models.items():
        train_features = fusion_features(model, x_train)
        val_features = fusion_features(model, train_arrays["x"]["validation"])
        for c_value in ELASTIC_C_GRID:
            for l1_ratio in L1_RATIO_GRID:
                classifier = elastic_lr(c_value, l1_ratio, seed).fit(train_features, y_train)
                pred = classifier.predict(val_features); metrics = suite.metric(train_arrays["y"]["validation"], pred)
                row = {
                    "experiment": "elastic-net", "split": split_name, "seed": seed,
                    "C": c_value, "l1_ratio": l1_ratio,
                    "validation_accuracy": metrics["accuracy"],
                    "validation_balanced_accuracy": metrics["balanced_accuracy"],
                    "validation_macro_f1": metrics["macro_f1"],
                }
                tuning.append(row); tuning_records.append(row)
    save(pd.DataFrame(tuning_records), paths["tuning"])
    selected = score_key(pd.DataFrame(tuning), ["C", "l1_ratio"])
    selected_c, selected_ratio = float(selected["C"]), float(selected["l1_ratio"])
    x_final = np.concatenate([final_arrays["x"]["source"], final_arrays["x"]["target_train"]])
    y_final = np.concatenate([final_arrays["y"]["source"], final_arrays["y"]["target_train"]])
    for seed, (_, best_epoch) in base_models.items():
        model, _ = train_hybrid(final_arrays, seed, None, False, best_epoch)
        classifier = elastic_lr(selected_c, selected_ratio, seed).fit(fusion_features(model, x_final), y_final)
        pred = classifier.predict(fusion_features(model, final_arrays["x"]["test"]))
        metrics = suite.metric(final_arrays["y"]["test"], pred)
        run_records.append({
            "experiment": "elastic-net", "split": split_name, "seed": seed,
            "selected_alpha": np.nan, "selected_beta": np.nan, "selected_C": selected_c,
            "selected_l1_ratio": selected_ratio, "best_epoch": best_epoch,
            "test_accuracy": metrics["accuracy"], "test_balanced_accuracy": metrics["balanced_accuracy"],
            "test_macro_f1": metrics["macro_f1"],
        })
        append_predictions(prediction_records, "elastic-net", split_name, seed, parts["target_test"], final_arrays["y"]["test"], pred)
        save(pd.DataFrame(run_records), paths["runs"]); save(pd.DataFrame(prediction_records), paths["predictions"])


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, split_name), group in runs.groupby(["experiment", "split"], sort=False):
        first = group.iloc[0]
        row = {
            "experiment": experiment, "split": split_name,
            "selected_alpha": first["selected_alpha"], "selected_beta": first["selected_beta"],
            "selected_C": first["selected_C"], "selected_l1_ratio": first["selected_l1_ratio"],
            "seeds": ",".join(map(str, sorted(group.seed.astype(int).unique()))),
        }
        for metric in ("test_accuracy", "test_balanced_accuracy", "test_macro_f1", "best_epoch"):
            values = group[metric].astype(float)
            row[metric] = float(values.mean()); row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_min"] = float(values.min()); row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def write_protocol(paths: dict[str, Path], args: argparse.Namespace) -> None:
    payload = {
        "status": "exploratory strict-inductive reanalysis of previously inspected data",
        "features": 54, "latent_features": 16, "fusion_features": 70,
        "classes": suite.CLASSES, "seeds": list(SEEDS), "epochs": EPOCHS,
        "experiments": list(args.experiments), "splits": list(args.splits),
        "test_boundary": "target test features and labels excluded from training, alignment, validation, and model selection",
        "selection_metric_order": ["validation macro-F1", "validation balanced accuracy", "validation accuracy"],
        "probability_ensemble": {"alpha_grid": list(ALPHA_GRID), "formula": "alpha*p_raw + (1-alpha)*p_fusion"},
        "target_weighted": {"beta_grid": list(BETA_GRID), "formula": "Ls + beta*Lt + domain losses"},
        "elastic_net": {"C_grid": list(ELASTIC_C_GRID), "l1_ratio_grid": list(L1_RATIO_GRID), "solver": "saga"},
        "CDAN_C_DANN_fixed": {"alignment_strength": ALIGNMENT_STRENGTH, "centroid_strength": CENTROID_STRENGTH},
        "sources": {
            "soft_voting": "scikit-learn VotingClassifier weighted predict_proba mechanism",
            "elastic_net": "scikit-learn LogisticRegression penalty=elasticnet, solver=saga",
            "target_weighting": "PyTorch cross-entropy domain-specific losses",
        },
    }
    paths["protocol"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args(); paths = output_paths(); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(FEATURE_PATH); columns = suite.feature_columns(raw)
    split_map = {item["split"]: item for item in STRICT_SPLITS}
    unknown = set(args.splits) - set(split_map)
    if unknown:
        raise ValueError(f"Unknown strict splits: {sorted(unknown)}")
    tuning_records, run_records, prediction_records = (
        read_records(paths["tuning"]), read_records(paths["runs"]), read_records(paths["predictions"])
    )
    done = set()
    if run_records:
        frame = pd.DataFrame(run_records)
        counts = frame.groupby(["experiment", "split"]).size()
        done = {key for key, count in counts.items() if int(count) == len(SEEDS)}
    for split_name in args.splits:
        split = split_map[split_name]; parts = strict_parts(raw, split)
        train_arrays = arrays_from_parts(parts, columns, False)
        final_data = arrays_from_parts(parts, columns, True)
        needed = [experiment for experiment in args.experiments if (experiment, split_name) not in done]
        if not needed:
            continue
        # Incomplete final groups are restarted as complete five-seed groups.
        run_records = [row for row in run_records if not (row["split"] == split_name and row["experiment"] in needed)]
        prediction_records = [row for row in prediction_records if not (row["split"] == split_name and row["experiment"] in needed)]
        tuning_records = [
            row for row in tuning_records
            if not (
                row.get("split") == split_name
                and row.get("experiment") in set(needed) | {"base-feature-fusion"}
            )
        ]
        base_models = {}
        if {"probability-ensemble", "elastic-net"} & set(needed):
            print(f"[{split_name}] fitting shared strict-inductive feature-fusion backbone", flush=True)
            base_models = run_base_models(train_arrays, tuning_records, split_name, paths)
        if "probability-ensemble" in needed:
            print(f"[{split_name}] probability-ensemble", flush=True)
            run_probability_ensemble(split, parts, train_arrays, final_data, base_models, tuning_records, run_records, prediction_records, paths)
        if "target-weighted" in needed:
            print(f"[{split_name}] target-weighted", flush=True)
            run_target_weighted(split, parts, train_arrays, final_data, tuning_records, run_records, prediction_records, paths)
        if "elastic-net" in needed:
            print(f"[{split_name}] elastic-net", flush=True)
            run_elastic_net(split, parts, train_arrays, final_data, base_models, tuning_records, run_records, prediction_records, paths)
    runs = pd.DataFrame(run_records)
    expected = len(args.experiments) * len(args.splits) * len(SEEDS)
    relevant = runs[runs.experiment.isin(args.experiments) & runs.split.isin(args.splits)]
    if len(relevant) != expected:
        raise RuntimeError(f"Expected {expected} requested final rows, found {len(relevant)}.")
    save(summarize(runs), paths["summary"]); write_protocol(paths, args)
    print(f"Outputs written to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
