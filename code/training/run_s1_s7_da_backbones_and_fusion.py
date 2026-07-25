"""Original S1-S7 transductive DA with 54D backbones and feature fusion.

Protocol requested by the thesis author:
* S1-S3 run UDA; gas loss uses source labels only.
* S4-S7 run Semi; gas loss uses source plus assigned target labels.
* target_domain is the complete later-period feature set and therefore includes
  unlabeled target-test features. This is transductive DA, not inductive DA.
* Target-test gas labels are evaluation-only.

The model definitions and fixed strengths are shared with
run_strict_inductive_da_backbones_and_fusion.py, but outputs are independent.
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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_dann_multi_splits as existing_dann
import run_strict_inductive_da_backbones_and_fusion as models
import run_unified_da_source_target_suite as suite
import shared_experiment_protocol as protocol


ROOT = Path(r"D:\thesis")
FEATURE_PATH = ROOT / "tables" / "manifest_baseline_as_air_features.csv"
OUTPUT_DIR = ROOT / "tables" / "s1_s7_da_backbones_fusion"
SEEDS = tuple(protocol.MODEL_SEEDS)
EPOCHS = int(protocol.NEURAL_EPOCHS)
EVAL_EVERY = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", choices=list(models.SPECS), default=list(models.SPECS))
    parser.add_argument("--splits", nargs="+", default=[item["split"] for item in protocol.DA_SPLITS])
    return parser.parse_args()


def paths() -> dict[str, Path]:
    return {
        "tuning": OUTPUT_DIR / "tuning.csv",
        "runs": OUTPUT_DIR / "runs.csv",
        "predictions": OUTPUT_DIR / "predictions.csv",
        "curves": OUTPUT_DIR / "curves.csv",
        "summary": OUTPUT_DIR / "summary.csv",
        "protocol": OUTPUT_DIR / "protocol.json",
    }


def save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def read_records(path: Path) -> list[dict]:
    return pd.read_csv(path).to_dict("records") if path.exists() else []


def split_number(split_name: str) -> int:
    return int(split_name[1])


def mode_for(split_name: str) -> str:
    return "uda" if split_number(split_name) <= 3 else "semi"


def final_arrays(parts: dict, columns: list[str], mode: str) -> tuple[dict, bool, str]:
    initial = suite.make_arrays(parts, columns)
    if mode == "semi":
        return (
            suite.make_arrays(parts, columns, ("source_train", "target_labeled_train", "target_val")),
            True,
            "Source+Target refit",
        )
    if len(initial["x"]["source_val"]):
        return suite.make_arrays(parts, columns, ("source_train", "source_val")), True, "source train+validation refit"
    return initial, False, "source-only; no refit"


def centroid_target(arrays: dict, mode: str, include_validation: bool) -> tuple[np.ndarray, np.ndarray | None]:
    if mode == "semi":
        keys = ["target_labeled_train"] + (["target_val"] if include_validation else [])
        x = suite.concatenate([arrays["x"][key] for key in keys], 54)
        y = suite.concatenate([arrays["y"][key] for key in keys])
        if len(x):
            return x, y
    return arrays["x"]["target_domain"], None


def train_model(
    arrays: dict,
    spec: models.ExperimentSpec,
    mode: str,
    seed: int,
    include_validation: bool,
    fixed_epochs: int | None = None,
) -> tuple[models.DAModel, dict, list[dict]]:
    x_cls, y_cls, label_usage = suite.classifier_data(arrays, mode, include_validation)
    x_select, y_select, selection_usage = suite.selection_data(arrays, mode)
    if include_validation:
        x_select = y_select = None
    existing_dann.set_random_seed(seed)
    model = models.DAModel(spec.backbone)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=protocol.NEURAL_LEARNING_RATE,
        weight_decay=protocol.NEURAL_WEIGHT_DECAY,
    )
    x_cls_t, y_cls_t = existing_dann.to_tensor(x_cls), existing_dann.to_long(y_cls)
    class_weight = existing_dann.class_weights(y_cls, len(suite.CLASSES))
    xs = existing_dann.to_tensor(arrays["x"]["source_train"])
    xt = existing_dann.to_tensor(arrays["x"]["target_domain"])
    domain_y = torch.cat([torch.zeros(len(xs), dtype=torch.long), torch.ones(len(xt), dtype=torch.long)])
    strengths = models.FIXED_STRENGTHS[spec.method]
    target_centroid_x, target_centroid_y = centroid_target(arrays, mode, include_validation)

    epochs_to_run = int(fixed_epochs or EPOCHS)
    best_state = copy.deepcopy(model.state_dict())
    best_score = (-1.0, -1.0, -1.0)
    best_epoch = 0
    history = []
    for epoch in range(1, epochs_to_run + 1):
        model.train(); optimizer.zero_grad()
        class_loss = F.cross_entropy(model.class_logits(x_cls_t), y_cls_t, weight=class_weight)
        source_h, target_h = model.features(xs), model.features(xt)
        alignment_loss = torch.tensor(0.0)
        centroid_loss = torch.tensor(0.0)
        total = class_loss
        if spec.method in {"dann", "c-dann"}:
            alignment_loss = F.cross_entropy(
                model.dann_domain_logits(torch.cat([source_h, target_h]), strengths["alignment"]), domain_y
            )
            total = total + alignment_loss
        elif spec.method == "cdan":
            alignment_loss = F.cross_entropy(
                model.cdan_domain_logits(torch.cat([source_h, target_h]), strengths["alignment"]), domain_y
            )
            total = total + alignment_loss
        elif spec.method == "mk-mmd":
            alignment_loss = suite.mkmmd_loss(source_h, target_h)
            total = total + strengths["alignment"] * alignment_loss
        elif spec.method == "deep-coral":
            alignment_loss = suite.coral_loss(source_h, target_h)
            total = total + strengths["alignment"] * alignment_loss
        else:
            raise ValueError(f"Unsupported method: {spec.method}")
        if spec.method == "c-dann":
            centroid_h = model.features(existing_dann.to_tensor(target_centroid_x))
            centroid_loss = suite.centroid_loss(
                model, source_h, arrays["y"]["source_train"], centroid_h, target_centroid_y
            )
            total = total + strengths["centroid"] * centroid_loss
        total.backward(); optimizer.step()

        evaluate = epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == epochs_to_run
        if evaluate:
            val_metric = (
                suite.metric(y_select, models.neural_predictions(model, x_select))
                if x_select is not None else {"accuracy": np.nan, "balanced_accuracy": np.nan, "macro_f1": np.nan}
            )
            history.append({
                "epoch": epoch, "classification_loss": float(class_loss.detach()),
                "alignment_loss": float(alignment_loss.detach()),
                "centroid_loss": float(centroid_loss.detach()), "total_loss": float(total.detach()),
                "validation_accuracy": val_metric["accuracy"],
                "validation_balanced_accuracy": val_metric["balanced_accuracy"],
                "validation_macro_f1": val_metric["macro_f1"],
            })
            if x_select is not None:
                score = (val_metric["macro_f1"], val_metric["balanced_accuracy"], val_metric["accuracy"])
                if score > best_score:
                    best_score, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        if x_select is None and epoch == epochs_to_run:
            best_epoch, best_state = epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch, "label_usage": label_usage,
        "selection_usage": selection_usage,
    }, history


def run_group(
    spec: models.ExperimentSpec, split: dict, parts: dict, initial: dict,
    tuning_records: list[dict], run_records: list[dict], prediction_records: list[dict],
    curve_records: list[dict], output_paths: dict[str, Path],
) -> None:
    split_name = split["split"]; mode = mode_for(split_name)
    selected_epochs = {}
    for seed in SEEDS:
        model, info, _ = train_model(initial, spec, mode, seed, False)
        x_select, y_select, _ = suite.selection_data(initial, mode)
        val = (
            suite.metric(y_select, models.neural_predictions(model, x_select))
            if x_select is not None else {"accuracy": np.nan, "balanced_accuracy": np.nan, "macro_f1": np.nan}
        )
        selected_epochs[seed] = int(info["best_epoch"])
        tuning_records.append({
            "experiment": spec.name, "family": spec.family, "split": split_name,
            "mode": mode, "seed": seed, "best_epoch": int(info["best_epoch"]),
            "alignment_strength": models.FIXED_STRENGTHS[spec.method]["alignment"],
            "centroid_strength": models.FIXED_STRENGTHS[spec.method]["centroid"],
            "validation_accuracy": val["accuracy"],
            "validation_balanced_accuracy": val["balanced_accuracy"],
            "validation_macro_f1": val["macro_f1"],
            "selection_usage": info["selection_usage"],
        })
        save(pd.DataFrame(tuning_records), output_paths["tuning"])

    final, include_validation, training_status = final_arrays(parts, suite.feature_columns(pd.read_csv(FEATURE_PATH)), mode)
    x_final, y_final, label_usage = suite.classifier_data(final, mode, include_validation)
    for seed in SEEDS:
        model, info, history = train_model(final, spec, mode, seed, include_validation, selected_epochs[seed])
        classifier = strict_lr().fit(models.probe_features(spec, model, x_final), y_final)
        train_pred = classifier.predict(models.probe_features(spec, model, x_final))
        test_pred = classifier.predict(models.probe_features(spec, model, final["x"]["target_test"]))
        train_metric = suite.metric(y_final, train_pred)
        test_metric = suite.metric(final["y"]["target_test"], test_pred)
        neural_test = suite.metric(
            final["y"]["target_test"], models.neural_predictions(model, final["x"]["target_test"])
        )
        run_records.append({
            "experiment": spec.name, "family": spec.family, "backbone": spec.backbone,
            "method": spec.method, "probe": spec.probe, "split": split_name, "mode": mode,
            "seed": seed, "latent_dimensions": model.latent_dim,
            "probe_dimensions": model.latent_dim + (54 if spec.probe == "fusion" else 0),
            "best_epoch": selected_epochs[seed], "final_training": training_status,
            "label_usage": label_usage, "lr_C": models.LR_C,
            "train_accuracy": train_metric["accuracy"],
            "train_balanced_accuracy": train_metric["balanced_accuracy"],
            "train_macro_f1": train_metric["macro_f1"],
            "neural_test_accuracy": neural_test["accuracy"],
            "neural_test_balanced_accuracy": neural_test["balanced_accuracy"],
            "neural_test_macro_f1": neural_test["macro_f1"],
            "test_accuracy": test_metric["accuracy"],
            "test_balanced_accuracy": test_metric["balanced_accuracy"],
            "test_macro_f1": test_metric["macro_f1"],
        })
        for row in history:
            curve_records.append({"experiment": spec.name, "split": split_name, "mode": mode, "seed": seed, **row})
        models.append_predictions(
            prediction_records, spec, split_name, seed, parts["target_test"],
            final["y"]["target_test"], test_pred,
        )
        save(pd.DataFrame(run_records), output_paths["runs"])
        save(pd.DataFrame(prediction_records), output_paths["predictions"])
        save(pd.DataFrame(curve_records), output_paths["curves"])


def strict_lr():
    import run_semi_inductive_extensions as strict_helpers
    return strict_helpers.l2_lr()


def write_protocol(args: argparse.Namespace, output_paths: dict[str, Path]) -> None:
    payload = {
        "protocol": "original S1-S7 transductive DA",
        "classes": suite.CLASSES, "seeds": list(SEEDS), "epochs": EPOCHS,
        "requested_experiments": list(args.experiments), "requested_splits": list(args.splits),
        "modes": "S1-S3 UDA; S4-S7 Semi",
        "target_test_labels": "evaluation-only",
        "target_test_features": "included unlabeled in target_domain alignment; transductive setting",
        "fixed_strengths": models.FIXED_STRENGTHS, "LR_C": models.LR_C,
        "experiments": {name: spec.__dict__ for name, spec in models.SPECS.items()},
    }
    output_paths["protocol"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args(); output_paths = paths(); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(FEATURE_PATH); columns = suite.feature_columns(raw)
    split_map = {item["split"]: item for item in protocol.DA_SPLITS}
    unknown = set(args.splits) - set(split_map)
    if unknown:
        raise ValueError(f"Unknown S1-S7 splits: {sorted(unknown)}")
    tuning_records = read_records(output_paths["tuning"])
    run_records = read_records(output_paths["runs"])
    prediction_records = read_records(output_paths["predictions"])
    curve_records = read_records(output_paths["curves"])
    complete = set()
    if run_records:
        counts = pd.DataFrame(run_records).groupby(["experiment", "split"]).size()
        complete = {key for key, count in counts.items() if int(count) == len(SEEDS)}
    total = len(args.experiments) * len(args.splits); number = 0
    for experiment in args.experiments:
        spec = models.SPECS[experiment]
        for split_name in args.splits:
            number += 1
            if (experiment, split_name) in complete:
                continue
            print(f"[{number}/{total}] {experiment} | {split_name} | {mode_for(split_name)}", flush=True)
            run_records = [row for row in run_records if not (row["experiment"] == experiment and row["split"] == split_name)]
            prediction_records = [row for row in prediction_records if not (row["experiment"] == experiment and row["split"] == split_name)]
            curve_records = [row for row in curve_records if not (row["experiment"] == experiment and row["split"] == split_name)]
            tuning_records = [row for row in tuning_records if not (row["experiment"] == experiment and row["split"] == split_name)]
            split = split_map[split_name]; parts = suite.partitions(raw, split)
            initial = suite.make_arrays(parts, columns)
            run_group(
                spec, split, parts, initial, tuning_records, run_records,
                prediction_records, curve_records, output_paths,
            )
    runs = pd.DataFrame(run_records)
    requested = runs[runs.experiment.isin(args.experiments) & runs.split.isin(args.splits)]
    expected = len(args.experiments) * len(args.splits) * len(SEEDS)
    if len(requested) != expected:
        raise RuntimeError(f"Expected {expected} requested rows, found {len(requested)}.")
    save(models.summarize(runs), output_paths["summary"]); write_protocol(args, output_paths)
    print(f"Outputs written to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
