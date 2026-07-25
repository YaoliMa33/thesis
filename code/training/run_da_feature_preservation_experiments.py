"""Feature-preserving domain-adaptation experiments for the e-nose data.

This runner is intentionally independent from the established DA result files.
It evaluates three pre-specified follow-up methods with Logistic Regression C=1:

1. feature-fusion: concatenate standardized raw x (54D) with the learned
   CDAN+C-DANN latent h (16D), then fit LR on the resulting 70D vector;
2. residual-mkmmd: learn h = x + F(x), where F is 54->64->54, and align the
   54D residual representation with global MK-MMD;
3. local-mkmmd: align class-wise distributions in a 16D latent space. UDA uses
   detached target class probabilities; Semi uses target labels assigned to
   training by the split when they are available.

The design follows the general mechanisms of feature concatenation for DA,
residual learning (He et al., CVPR 2016), DAN MK-MMD (Long et al., ICML 2015),
and local/class-conditional MMD (DSAN). It is an exploratory combination for
this thesis, not a claim of exact reproduction of any one paper.

Methodological boundaries
-------------------------
* UDA gas loss uses source labels only. Target gas labels are prohibited from
  training and selection.
* Semi uses only target labels explicitly assigned to training/validation by
  the established split protocol.
* Test labels are evaluation-only.
* LR uses a pre-specified C=1 for every split, method, mode, and seed.
* Hyperparameter selection concerns only the DA alignment strength and uses
  admissible validation labels. Without admissible validation, lambda=0.1.
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_dann_multi_splits as existing_dann
import run_unified_da_source_target_suite as suite
import shared_experiment_protocol as protocol


ROOT = Path(r"D:\thesis")
FEATURE_PATH = ROOT / "tables" / "manifest_baseline_as_air_features.csv"
OUTPUT_DIR = ROOT / "tables" / "da_feature_preservation"
EXPERIMENTS = ("feature-fusion", "residual-mkmmd", "local-mkmmd")
MODES = ("uda", "semi")
SEEDS = tuple(protocol.MODEL_SEEDS)
EPOCHS = int(protocol.NEURAL_EPOCHS)
EVAL_EVERY = 10
LAMBDA_GRID = (0.01, 0.1, 1.0)
FIXED_LAMBDA = 0.1
LR_C = 1.0

HYBRID_RUN_PATH = (
    ROOT
    / "tables"
    / "unified_da_source_target"
    / "runs__one-layer__c-dann_cdan_cdan+c-dann__uda_semi.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--splits", nargs="+", default=[item["split"] for item in protocol.DA_SPLITS])
    return parser.parse_args()


def output_paths(args: argparse.Namespace) -> dict[str, Path]:
    tag = "_".join(args.experiments) + "__" + "_".join(args.modes)
    default_splits = [item["split"] for item in protocol.DA_SPLITS]
    if list(args.splits) != default_splits:
        tag += "__" + "_".join(args.splits)
    return {
        "tuning": OUTPUT_DIR / f"tuning__{tag}.csv",
        "runs": OUTPUT_DIR / f"runs__{tag}.csv",
        "predictions": OUTPUT_DIR / f"predictions__{tag}.csv",
        "summary": OUTPUT_DIR / f"summary__{tag}.csv",
        "protocol": OUTPUT_DIR / f"protocol__{tag}.json",
    }


def save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def read_records(path: Path) -> list[dict]:
    return pd.read_csv(path).to_dict("records") if path.exists() else []


def lr_pipeline() -> Pipeline:
    return Pipeline([
        ("standardize", StandardScaler()),
        ("classifier", LogisticRegression(C=LR_C, class_weight="balanced", solver="lbfgs", max_iter=2000)),
    ])


class AdaptationModel(nn.Module):
    def __init__(self, n_features: int, n_classes: int, experiment: str) -> None:
        super().__init__()
        self.experiment = experiment
        if experiment == "residual-mkmmd":
            self.transform = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.ReLU(),
                nn.Dropout(protocol.NEURAL_DROPOUT),
                nn.Linear(64, n_features),
            )
            latent_dim = n_features
        elif experiment == "local-mkmmd":
            self.transform = suite.FeatureExtractor(n_features, (16,))
            latent_dim = 16
        else:
            raise ValueError(f"Unsupported trainable experiment: {experiment}")
        self.label_classifier = nn.Linear(latent_dim, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        transformed = self.transform(x)
        return x + transformed if self.experiment == "residual-mkmmd" else transformed

    def class_logits_from_h(self, h: torch.Tensor) -> torch.Tensor:
        return self.label_classifier(h)

    def class_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.class_logits_from_h(self.features(x))


def model_predictions(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.class_logits(existing_dann.to_tensor(x)).argmax(dim=1).cpu().numpy()


def latent(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.features(existing_dann.to_tensor(x)).cpu().numpy()


def normalized_class_weights(probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mass = probabilities.sum(dim=0)
    active = mass > torch.finfo(probabilities.dtype).eps
    weights = probabilities / mass.clamp_min(torch.finfo(probabilities.dtype).eps)
    return weights, active


def local_mkmmd_loss(
    model: AdaptationModel,
    source_h: torch.Tensor,
    source_y: np.ndarray,
    target_h: torch.Tensor,
    target_y: np.ndarray | None,
) -> torch.Tensor:
    source_prob = F.one_hot(existing_dann.to_long(source_y), num_classes=len(suite.CLASSES)).float()
    if target_y is None:
        target_prob = F.softmax(model.class_logits_from_h(target_h), dim=1).detach()
    else:
        target_prob = F.one_hot(existing_dann.to_long(target_y), num_classes=len(suite.CLASSES)).float()
    source_w, source_active = normalized_class_weights(source_prob)
    target_w, target_active = normalized_class_weights(target_prob)
    active = source_active & target_active
    if not bool(active.any()):
        raise ValueError("Local MK-MMD found no class with mass in both domains.")

    combined = torch.cat([source_h, target_h], dim=0)
    kernel = sum(suite.gaussian_kernel(combined, alpha) for alpha in suite.MMD_KERNEL_ALPHAS)
    ns = source_h.shape[0]
    k_ss, k_tt, k_st = kernel[:ns, :ns], kernel[ns:, ns:], kernel[:ns, ns:]
    class_losses = []
    for class_index in torch.where(active)[0]:
        ws = source_w[:, class_index]
        wt = target_w[:, class_index]
        class_losses.append(ws @ k_ss @ ws + wt @ k_tt @ wt - 2.0 * (ws @ k_st @ wt))
    return torch.stack(class_losses).mean()


def local_target_data(arrays: dict, mode: str, include_validation: bool) -> tuple[np.ndarray, np.ndarray | None, str]:
    if mode == "semi":
        keys = ["target_labeled_train"] + (["target_val"] if include_validation else [])
        x = suite.concatenate([arrays["x"][key] for key in keys], arrays["x"]["source_train"].shape[1])
        y = suite.concatenate([arrays["y"][key] for key in keys])
        if len(x):
            return x, y, "true labels from target partitions assigned to training"
    return arrays["x"]["target_domain"], None, "detached target class probabilities"


def train_adaptation_model(
    arrays: dict,
    experiment: str,
    mode: str,
    alignment_strength: float,
    seed: int,
    include_validation: bool,
    fixed_epochs: int | None = None,
) -> tuple[AdaptationModel, dict]:
    x_cls, y_cls, label_usage = suite.classifier_data(arrays, mode, include_validation)
    x_select, y_select, selection_usage = suite.selection_data(arrays, mode)
    if include_validation:
        x_select = y_select = None

    existing_dann.set_random_seed(seed)
    model = AdaptationModel(x_cls.shape[1], len(suite.CLASSES), experiment)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=protocol.NEURAL_LEARNING_RATE,
        weight_decay=protocol.NEURAL_WEIGHT_DECAY,
    )
    class_weight = existing_dann.class_weights(y_cls, len(suite.CLASSES))
    x_cls_t, y_cls_t = existing_dann.to_tensor(x_cls), existing_dann.to_long(y_cls)
    x_source, y_source = arrays["x"]["source_train"], arrays["y"]["source_train"]
    x_source_t = existing_dann.to_tensor(x_source)
    target_x, target_y, alignment_label_usage = local_target_data(arrays, mode, include_validation)
    if experiment == "residual-mkmmd":
        target_x, target_y = arrays["x"]["target_domain"], None
        alignment_label_usage = "target features only"
    target_x_t = existing_dann.to_tensor(target_x)

    epochs_to_run = int(fixed_epochs or EPOCHS)
    best_state = copy.deepcopy(model.state_dict())
    best_score = (-1.0, -1.0, -1.0)
    best_epoch = 0
    last = {}
    for epoch in range(1, epochs_to_run + 1):
        model.train()
        optimizer.zero_grad()
        class_loss = F.cross_entropy(model.class_logits(x_cls_t), y_cls_t, weight=class_weight)
        source_h, target_h = model.features(x_source_t), model.features(target_x_t)
        if experiment == "residual-mkmmd":
            alignment_loss = suite.mkmmd_loss(source_h, target_h)
        else:
            alignment_loss = local_mkmmd_loss(model, source_h, y_source, target_h, target_y)
        total_loss = class_loss + alignment_strength * alignment_loss
        total_loss.backward()
        optimizer.step()
        last = {
            "classification_loss": float(class_loss.detach()),
            "alignment_loss": float(alignment_loss.detach()),
            "total_loss": float(total_loss.detach()),
        }
        if x_select is not None and (epoch % EVAL_EVERY == 0 or epoch == epochs_to_run):
            selected = suite.metric(y_select, model_predictions(model, x_select))
            score = (selected["macro_f1"], selected["balanced_accuracy"], selected["accuracy"])
            if score > best_score:
                best_score, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        elif x_select is None and epoch == epochs_to_run:
            best_epoch, best_state = epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, {
        **last,
        "best_epoch": best_epoch,
        "label_usage": label_usage,
        "alignment_label_usage": alignment_label_usage,
        "selection_usage": selection_usage,
    }


def final_arrays(parts: dict, columns: list[str], mode: str, has_source_val: bool) -> tuple[dict, bool, str]:
    if mode == "semi":
        return (
            suite.make_arrays(parts, columns, ("source_train", "target_labeled_train", "target_val")),
            True,
            "Source+Target refit",
        )
    if has_source_val:
        return suite.make_arrays(parts, columns, ("source_train", "source_val")), True, "source train+source validation refit"
    return suite.make_arrays(parts, columns), False, "source-only; no refit"


def select_alignment_strength(tuning: pd.DataFrame) -> float:
    means = tuning.groupby("alignment_strength", as_index=False)[
        ["validation_macro_f1", "validation_balanced_accuracy", "validation_accuracy"]
    ].mean()
    chosen = means.sort_values(
        ["validation_macro_f1", "validation_balanced_accuracy", "validation_accuracy", "alignment_strength"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).iloc[0]
    return float(chosen["alignment_strength"])


def probe_features(experiment: str, model: nn.Module, x: np.ndarray) -> np.ndarray:
    h = latent(model, x)
    return np.concatenate([x, h], axis=1) if experiment == "feature-fusion" else h


def append_evaluation(
    experiment: str,
    model: nn.Module,
    final_data: dict,
    parts: dict,
    mode: str,
    split_name: str,
    seed: int,
    alignment_strength: float,
    best_epoch: int,
    final_training: str,
    label_usage: str,
    alignment_label_usage: str,
    selection_usage: str,
    run_records: list[dict],
    prediction_records: list[dict],
) -> None:
    include_validation = mode == "semi" or len(final_data["x"]["source_val"]) > 0
    x_train, y_train, final_label_usage = suite.classifier_data(final_data, mode, include_validation)
    x_test, y_test = final_data["x"]["target_test"], final_data["y"]["target_test"]
    train_probe, test_probe = probe_features(experiment, model, x_train), probe_features(experiment, model, x_test)
    classifier = lr_pipeline().fit(train_probe, y_train)
    train_pred, test_pred = classifier.predict(train_probe), classifier.predict(test_probe)
    neural_test = suite.metric(y_test, model_predictions(model, x_test))
    train_metric, test_metric = suite.metric(y_train, train_pred), suite.metric(y_test, test_pred)
    variant = f"{experiment.upper()}-{mode.upper()}"
    run_records.append({
        "experiment": experiment,
        "split": split_name,
        "variant": variant,
        "mode": mode,
        "seed": seed,
        "alignment_strength": alignment_strength,
        "best_epoch": best_epoch,
        "lr_C": LR_C,
        "probe_dimensions": int(train_probe.shape[1]),
        "label_usage": final_label_usage or label_usage,
        "alignment_label_usage": alignment_label_usage,
        "selection_usage": selection_usage,
        "final_training": final_training,
        "neural_test_accuracy": neural_test["accuracy"],
        "lr_train_accuracy": train_metric["accuracy"],
        "lr_train_balanced_accuracy": train_metric["balanced_accuracy"],
        "lr_train_macro_f1": train_metric["macro_f1"],
        "lr_test_accuracy": test_metric["accuracy"],
        "lr_test_balanced_accuracy": test_metric["balanced_accuracy"],
        "lr_test_macro_f1": test_metric["macro_f1"],
    })
    test_frame = parts["target_test"]
    for position, (index, sample) in enumerate(test_frame.iterrows()):
        prediction_records.append({
            "experiment": experiment,
            "split": split_name,
            "variant": variant,
            "mode": mode,
            "seed": seed,
            "sample_index": int(index),
            "sample_id": sample.get("sample_id", index),
            "target_file": sample.get("source_file", sample.get("file", "")),
            "period": sample.get("period", ""),
            "day_label": sample.get("day_label", ""),
            "batch": sample.get("batch", ""),
            "true_label": suite.CLASSES[int(y_test[position])],
            "predicted_label": suite.CLASSES[int(test_pred[position])],
            "correct": bool(y_test[position] == test_pred[position]),
        })


def run_feature_fusion(
    raw: pd.DataFrame,
    columns: list[str],
    split: dict,
    mode: str,
    hybrid_rows: pd.DataFrame,
    done: set,
    run_records: list[dict],
    prediction_records: list[dict],
    paths: dict[str, Path],
) -> None:
    split_name = split["split"]
    selected = hybrid_rows[
        hybrid_rows["split"].eq(split_name)
        & hybrid_rows["method"].eq("cdan+c-dann")
        & hybrid_rows["mode"].eq(mode)
    ]
    if selected.empty:
        if mode == "semi":
            return
        raise ValueError(f"Missing existing CDAN+C-DANN rows for {split_name} {mode}.")
    parts = suite.partitions(raw, split)
    initial = suite.make_arrays(parts, columns)
    has_source_val = len(initial["x"]["source_val"]) > 0
    data, include_validation, training_status = final_arrays(parts, columns, mode, has_source_val)
    for row in selected.itertuples(index=False):
        key = ("feature-fusion", split_name, mode, int(row.seed))
        if key in done:
            continue
        adversarial_value = getattr(row, "adversarial_strength", np.nan)
        # The original CDAN+C-DANN rows predate the separate adversarial column.
        # Their implementation explicitly reused alignment_strength for CDAN.
        if pd.isna(adversarial_value):
            adversarial_value = float(row.alignment_strength)
        candidate = suite.Candidate(
            float(row.alignment_strength),
            float(row.centroid_strength),
            float(adversarial_value),
        )
        model, info = suite.train_candidate(
            data,
            suite.ARCHITECTURES["one-layer"],
            "cdan+c-dann",
            mode,
            candidate,
            int(row.seed),
            include_validation,
            int(row.best_epoch),
        )
        append_evaluation(
            "feature-fusion", model, data, parts, mode, split_name, int(row.seed),
            candidate.alignment_strength, int(info["best_epoch"]), training_status,
            info["label_usage"], "CDAN domain labels plus class-conditional centroid weights",
            str(row.selection_usage), run_records, prediction_records,
        )
        save(pd.DataFrame(run_records), paths["runs"])
        save(pd.DataFrame(prediction_records), paths["predictions"])
        done.add(key)


def run_trainable_experiment(
    experiment: str,
    raw: pd.DataFrame,
    columns: list[str],
    split: dict,
    mode: str,
    done: set,
    tuning_records: list[dict],
    run_records: list[dict],
    prediction_records: list[dict],
    paths: dict[str, Path],
) -> None:
    split_name = split["split"]
    parts = suite.partitions(raw, split)
    if mode == "semi" and len(parts["target_labeled_train"]) + len(parts["target_val"]) == 0:
        return
    initial = suite.make_arrays(parts, columns)
    x_select, y_select, selection_usage = suite.selection_data(initial, mode)
    candidates = LAMBDA_GRID if x_select is not None else (FIXED_LAMBDA,)
    tuning_frame = pd.DataFrame(tuning_records)
    tuning_done = set()
    trained_models: dict[tuple[float, int], tuple[AdaptationModel, dict]] = {}
    if not tuning_frame.empty:
        subset = tuning_frame[
            tuning_frame["experiment"].eq(experiment)
            & tuning_frame["split"].eq(split_name)
            & tuning_frame["mode"].eq(mode)
        ]
        tuning_done = set(zip(subset.alignment_strength.astype(float), subset.seed.astype(int)))
    for strength in candidates:
        for seed in SEEDS:
            if (float(strength), int(seed)) in tuning_done:
                continue
            model, info = train_adaptation_model(initial, experiment, mode, strength, seed, False)
            trained_models[(float(strength), int(seed))] = (model, info)
            validation = {"accuracy": np.nan, "balanced_accuracy": np.nan, "macro_f1": np.nan}
            if x_select is not None:
                validation = suite.metric(y_select, model_predictions(model, x_select))
            tuning_records.append({
                "experiment": experiment,
                "split": split_name,
                "mode": mode,
                "alignment_strength": strength,
                "seed": seed,
                "best_epoch": info["best_epoch"],
                "validation_accuracy": validation["accuracy"],
                "validation_balanced_accuracy": validation["balanced_accuracy"],
                "validation_macro_f1": validation["macro_f1"],
                "selection_usage": selection_usage,
            })
            save(pd.DataFrame(tuning_records), paths["tuning"])
    tuning_frame = pd.DataFrame(tuning_records)
    group_tuning = tuning_frame[
        tuning_frame["experiment"].eq(experiment)
        & tuning_frame["split"].eq(split_name)
        & tuning_frame["mode"].eq(mode)
    ]
    selected_strength = select_alignment_strength(group_tuning) if x_select is not None else FIXED_LAMBDA
    selected_rows = group_tuning[np.isclose(group_tuning.alignment_strength, selected_strength)]
    has_source_val = len(initial["x"]["source_val"]) > 0
    data, include_validation, training_status = final_arrays(parts, columns, mode, has_source_val)
    for seed in SEEDS:
        key = (experiment, split_name, mode, int(seed))
        if key in done:
            continue
        selected_row = selected_rows[selected_rows.seed.eq(seed)].iloc[0]
        cached = trained_models.get((float(selected_strength), int(seed)))
        if x_select is None and not include_validation and cached is not None:
            model, info = cached
        else:
            model, info = train_adaptation_model(
                data, experiment, mode, selected_strength, seed, include_validation, int(selected_row.best_epoch)
            )
        append_evaluation(
            experiment, model, data, parts, mode, split_name, seed, selected_strength,
            int(info["best_epoch"]), training_status, info["label_usage"],
            info["alignment_label_usage"], selection_usage, run_records, prediction_records,
        )
        save(pd.DataFrame(run_records), paths["runs"])
        save(pd.DataFrame(prediction_records), paths["predictions"])
        done.add(key)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "neural_test_accuracy",
        "lr_train_accuracy",
        "lr_train_balanced_accuracy",
        "lr_train_macro_f1",
        "lr_test_accuracy",
        "lr_test_balanced_accuracy",
        "lr_test_macro_f1",
        "best_epoch",
    ]
    rows = []
    for key, group in runs.groupby(["experiment", "split", "mode"], sort=False):
        first = group.iloc[0]
        row = {
            "experiment": key[0],
            "split": key[1],
            "mode": key[2],
            "alignment_strength": first["alignment_strength"],
            "lr_C": LR_C,
            "probe_dimensions": first["probe_dimensions"],
            "label_usage": first["label_usage"],
            "alignment_label_usage": first["alignment_label_usage"],
            "selection_usage": first["selection_usage"],
            "final_training": first["final_training"],
            "seeds": ",".join(map(str, sorted(group.seed.astype(int).unique()))),
        }
        for metric_name in metrics:
            values = group[metric_name].astype(float)
            row[metric_name] = float(values.mean())
            row[f"{metric_name}_std"] = float(values.std(ddof=1))
            row[f"{metric_name}_min"] = float(values.min())
            row[f"{metric_name}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def write_protocol(args: argparse.Namespace, path: Path) -> None:
    payload = {
        "experiments": list(args.experiments),
        "modes": list(args.modes),
        "splits": list(args.splits),
        "features": 54,
        "classes": suite.CLASSES,
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "LR": {"C": LR_C, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 2000},
        "DA_alignment_grid": list(LAMBDA_GRID),
        "DA_fixed_default_without_validation": FIXED_LAMBDA,
        "selection": "validation macro-F1, balanced accuracy, accuracy; test labels prohibited",
        "label_policy": {
            "uda": "source gas labels only; target labels prohibited from training and selection",
            "semi": "source plus target labels explicitly assigned by the split",
            "test": "evaluation-only",
        },
        "provenance": {
            "feature-fusion": "raw standardized x concatenated with existing CDAN+C-DANN h",
            "residual-mkmmd": "residual representation plus DAN-style global MK-MMD",
            "local-mkmmd": "DSAN-style class-wise MMD; detached probabilities in UDA and known training labels in Semi",
        },
        "exploratory_status": "These are follow-up experiments motivated by prior observed test results; independent confirmation is required.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = output_paths(args)
    raw = pd.read_csv(FEATURE_PATH)
    columns = suite.feature_columns(raw)
    split_map = {item["split"]: item for item in protocol.DA_SPLITS}
    unknown = set(args.splits) - set(split_map)
    if unknown:
        raise ValueError(f"Unknown splits: {sorted(unknown)}")
    hybrid_rows = pd.read_csv(HYBRID_RUN_PATH)
    tuning_records = read_records(paths["tuning"])
    run_records = read_records(paths["runs"])
    prediction_records = read_records(paths["predictions"])
    done = {
        (str(row["experiment"]), str(row["split"]), str(row["mode"]), int(row["seed"]))
        for row in run_records
    }
    total = len(args.experiments) * len(args.splits) * len(args.modes)
    counter = 0
    for experiment in args.experiments:
        for split_name in args.splits:
            for mode in args.modes:
                counter += 1
                print(f"[{counter}/{total}] {experiment} | {split_name} | {mode}", flush=True)
                split = split_map[split_name]
                if experiment == "feature-fusion":
                    run_feature_fusion(
                        raw, columns, split, mode, hybrid_rows, done, run_records,
                        prediction_records, paths,
                    )
                else:
                    run_trainable_experiment(
                        experiment, raw, columns, split, mode, done, tuning_records,
                        run_records, prediction_records, paths,
                    )
    runs = pd.DataFrame(run_records)
    if runs.empty:
        raise RuntimeError("No experiment rows were produced.")
    save(summarize(runs), paths["summary"])
    write_protocol(args, paths["protocol"])
    print(f"Outputs written to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
