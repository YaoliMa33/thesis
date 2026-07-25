"""Strict-inductive non-compressive DA and multi-method feature fusion.

Step 3: preserve a 54D latent representation with either an identity-initialized
linear map or a zero-initialized residual adapter, then apply DANN, CDAN, or the
project's custom C-DANN (DANN plus class-wise centroid alignment).

Step 4: learn the established compressed 16D representation with DANN, CDAN,
C-DANN, MK-MMD, or Deep CORAL, concatenate [raw x_54, latent h_16], and fit the
same fixed-C Logistic Regression probe.

All experiments use SI1-SI4 from run_semi_inductive_extensions.py. Target test
features and labels are excluded from classification, alignment, validation,
normalization, and model selection. Method strengths are fixed before running;
validation selects only the best epoch. Results are exploratory because the
dataset was inspected in earlier work.
"""

from __future__ import annotations

import argparse
import copy
import json
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
import run_dann_multi_splits as existing_dann
import run_semi_inductive_extensions as strict
import run_unified_da_source_target_suite as suite
import shared_experiment_protocol as protocol


ROOT = Path(r"D:\thesis")
FEATURE_PATH = ROOT / "tables" / "manifest_baseline_as_air_features.csv"
OUTPUT_DIR = ROOT / "tables" / "strict_inductive_da_backbones_fusion"
SEEDS = tuple(protocol.MODEL_SEEDS)
EPOCHS = int(protocol.NEURAL_EPOCHS)
EVAL_EVERY = 10
LR_C = 1.0


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    family: str
    backbone: str
    method: str
    probe: str


SPECS = {
    spec.name: spec
    for spec in [
        ExperimentSpec("linear54-dann", "noncompressive-54D", "linear54", "dann", "latent"),
        ExperimentSpec("linear54-cdan", "noncompressive-54D", "linear54", "cdan", "latent"),
        ExperimentSpec("linear54-c-dann", "noncompressive-54D", "linear54", "c-dann", "latent"),
        ExperimentSpec("residual54-dann", "noncompressive-54D", "residual54", "dann", "latent"),
        ExperimentSpec("residual54-cdan", "noncompressive-54D", "residual54", "cdan", "latent"),
        ExperimentSpec("residual54-c-dann", "noncompressive-54D", "residual54", "c-dann", "latent"),
        ExperimentSpec("fusion-dann", "raw-plus-latent-fusion", "compressed16", "dann", "fusion"),
        ExperimentSpec("fusion-cdan", "raw-plus-latent-fusion", "compressed16", "cdan", "fusion"),
        ExperimentSpec("fusion-c-dann", "raw-plus-latent-fusion", "compressed16", "c-dann", "fusion"),
        ExperimentSpec("fusion-mk-mmd", "raw-plus-latent-fusion", "compressed16", "mk-mmd", "fusion"),
        ExperimentSpec("fusion-deep-coral", "raw-plus-latent-fusion", "compressed16", "deep-coral", "fusion"),
    ]
}

STEP3 = [name for name, spec in SPECS.items() if spec.family == "noncompressive-54D"]
STEP4 = [name for name, spec in SPECS.items() if spec.family == "raw-plus-latent-fusion"]

FIXED_STRENGTHS = {
    "dann": {"alignment": 0.2, "centroid": 0.0},
    "cdan": {"alignment": 0.2, "centroid": 0.0},
    "c-dann": {"alignment": 0.2, "centroid": 0.2},
    "mk-mmd": {"alignment": 0.1, "centroid": 0.0},
    "deep-coral": {"alignment": 0.1, "centroid": 0.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", choices=list(SPECS), default=list(SPECS))
    parser.add_argument("--splits", nargs="+", default=[item["split"] for item in strict.STRICT_SPLITS])
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


class Linear54(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(54, 54)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Residual54(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(54, 64),
            nn.ReLU(),
            nn.Dropout(protocol.NEURAL_DROPOUT),
            nn.Linear(64, 54),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.adapter(x)


class DAModel(nn.Module):
    def __init__(self, backbone: str) -> None:
        super().__init__()
        if backbone == "linear54":
            self.feature_extractor, latent_dim = Linear54(), 54
        elif backbone == "residual54":
            self.feature_extractor, latent_dim = Residual54(), 54
        elif backbone == "compressed16":
            self.feature_extractor, latent_dim = suite.FeatureExtractor(54, (16,)), 16
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        self.latent_dim = latent_dim
        self.label_classifier = nn.Linear(latent_dim, len(suite.CLASSES))
        self.dann_domain_classifier = nn.Sequential(nn.Linear(latent_dim, 16), nn.ReLU(), nn.Linear(16, 2))
        self.cdan_domain_classifier = existing_cdan.CDANDomainDiscriminator(latent_dim, len(suite.CLASSES))

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
        logits = self.class_logits_from_h(h)
        conditioned = existing_cdan.multilinear_condition(h, logits, detach_predictions=True)
        value = existing_dann.grad_reverse(conditioned, strength) if reverse else conditioned
        return self.cdan_domain_classifier(value)


def latent(model: DAModel, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.features(existing_dann.to_tensor(x)).cpu().numpy()


def neural_predictions(model: DAModel, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.class_logits(existing_dann.to_tensor(x)).argmax(dim=1).cpu().numpy()


def probe_features(spec: ExperimentSpec, model: DAModel, x: np.ndarray) -> np.ndarray:
    h = latent(model, x)
    return np.concatenate([x, h], axis=1) if spec.probe == "fusion" else h


def train_model(
    arrays: dict,
    spec: ExperimentSpec,
    seed: int,
    select_on_validation: bool,
    fixed_epochs: int | None = None,
) -> tuple[DAModel, dict, list[dict]]:
    existing_dann.set_random_seed(seed)
    model = DAModel(spec.backbone)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=protocol.NEURAL_LEARNING_RATE,
        weight_decay=protocol.NEURAL_WEIGHT_DECAY,
    )
    xs, ys = existing_dann.to_tensor(arrays["x"]["source"]), existing_dann.to_long(arrays["y"]["source"])
    xt, yt = existing_dann.to_tensor(arrays["x"]["target_train"]), existing_dann.to_long(arrays["y"]["target_train"])
    x_cls, y_cls = torch.cat([xs, xt]), torch.cat([ys, yt])
    class_weight = existing_dann.class_weights(
        np.concatenate([arrays["y"]["source"], arrays["y"]["target_train"]]), len(suite.CLASSES)
    )
    domain_y = torch.cat([torch.zeros(len(xs), dtype=torch.long), torch.ones(len(xt), dtype=torch.long)])
    strengths = FIXED_STRENGTHS[spec.method]
    best_state = copy.deepcopy(model.state_dict())
    best_score = (-1.0, -1.0, -1.0)
    best_epoch = 0
    epochs_to_run = int(fixed_epochs or EPOCHS)
    history = []
    for epoch in range(1, epochs_to_run + 1):
        model.train(); optimizer.zero_grad()
        class_loss = F.cross_entropy(model.class_logits(x_cls), y_cls, weight=class_weight)
        source_h, target_h = model.features(xs), model.features(xt)
        alignment_loss = torch.tensor(0.0)
        centroid_loss = torch.tensor(0.0)
        total = class_loss
        if spec.method in {"dann", "c-dann"}:
            h_domain = torch.cat([source_h, target_h])
            alignment_loss = F.cross_entropy(
                model.dann_domain_logits(h_domain, strengths["alignment"]), domain_y
            )
            total = total + alignment_loss
        elif spec.method == "cdan":
            h_domain = torch.cat([source_h, target_h])
            alignment_loss = F.cross_entropy(
                model.cdan_domain_logits(h_domain, strengths["alignment"]), domain_y
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
            centroid_loss = suite.centroid_loss(
                model, source_h, arrays["y"]["source"], target_h, arrays["y"]["target_train"]
            )
            total = total + strengths["centroid"] * centroid_loss
        total.backward(); optimizer.step()

        evaluate = epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == epochs_to_run
        if evaluate:
            val_metric = suite.metric(
                arrays["y"]["validation"], neural_predictions(model, arrays["x"]["validation"])
            ) if select_on_validation else {"accuracy": np.nan, "balanced_accuracy": np.nan, "macro_f1": np.nan}
            history.append({
                "epoch": epoch,
                "classification_loss": float(class_loss.detach()),
                "alignment_loss": float(alignment_loss.detach()),
                "centroid_loss": float(centroid_loss.detach()),
                "total_loss": float(total.detach()),
                "validation_accuracy": val_metric["accuracy"],
                "validation_balanced_accuracy": val_metric["balanced_accuracy"],
                "validation_macro_f1": val_metric["macro_f1"],
            })
            if select_on_validation:
                score = (val_metric["macro_f1"], val_metric["balanced_accuracy"], val_metric["accuracy"])
                if score > best_score:
                    best_score, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        if not select_on_validation and epoch == epochs_to_run:
            best_epoch, best_state = epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_score": best_score}, history


def append_predictions(
    records: list[dict], spec: ExperimentSpec, split_name: str, seed: int,
    test_frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
) -> None:
    for position, (index, sample) in enumerate(test_frame.iterrows()):
        records.append({
            "experiment": spec.name, "family": spec.family, "method": spec.method,
            "split": split_name, "seed": seed, "sample_index": int(index),
            "sample_id": sample.get("sample_id", index), "day_label": sample.get("day_label", ""),
            "batch": sample.get("batch", ""), "true_label": suite.CLASSES[int(y_true[position])],
            "predicted_label": suite.CLASSES[int(y_pred[position])],
            "correct": bool(y_true[position] == y_pred[position]),
        })


def run_group(
    spec: ExperimentSpec,
    split: dict,
    parts: dict,
    train_arrays: dict,
    final_arrays: dict,
    tuning_records: list[dict],
    run_records: list[dict],
    prediction_records: list[dict],
    curve_records: list[dict],
    output_paths: dict[str, Path],
) -> None:
    split_name = split["split"]
    selected_epochs = {}
    for seed in SEEDS:
        model, info, _ = train_model(train_arrays, spec, seed, True)
        val_neural = suite.metric(
            train_arrays["y"]["validation"], neural_predictions(model, train_arrays["x"]["validation"])
        )
        x_train = np.concatenate([train_arrays["x"]["source"], train_arrays["x"]["target_train"]])
        y_train = np.concatenate([train_arrays["y"]["source"], train_arrays["y"]["target_train"]])
        classifier = strict.l2_lr().fit(probe_features(spec, model, x_train), y_train)
        val_lr = suite.metric(
            train_arrays["y"]["validation"], classifier.predict(probe_features(spec, model, train_arrays["x"]["validation"]))
        )
        selected_epochs[seed] = int(info["best_epoch"])
        tuning_records.append({
            "experiment": spec.name, "family": spec.family, "backbone": spec.backbone,
            "method": spec.method, "split": split_name, "seed": seed,
            "alignment_strength": FIXED_STRENGTHS[spec.method]["alignment"],
            "centroid_strength": FIXED_STRENGTHS[spec.method]["centroid"],
            "best_epoch": int(info["best_epoch"]),
            "validation_neural_accuracy": val_neural["accuracy"],
            "validation_neural_balanced_accuracy": val_neural["balanced_accuracy"],
            "validation_neural_macro_f1": val_neural["macro_f1"],
            "validation_lr_accuracy": val_lr["accuracy"],
            "validation_lr_balanced_accuracy": val_lr["balanced_accuracy"],
            "validation_lr_macro_f1": val_lr["macro_f1"],
        })
        save(pd.DataFrame(tuning_records), output_paths["tuning"])

    x_final = np.concatenate([final_arrays["x"]["source"], final_arrays["x"]["target_train"]])
    y_final = np.concatenate([final_arrays["y"]["source"], final_arrays["y"]["target_train"]])
    for seed in SEEDS:
        model, info, history = train_model(final_arrays, spec, seed, False, selected_epochs[seed])
        classifier = strict.l2_lr().fit(probe_features(spec, model, x_final), y_final)
        train_pred = classifier.predict(probe_features(spec, model, x_final))
        test_pred = classifier.predict(probe_features(spec, model, final_arrays["x"]["test"]))
        train_metric = suite.metric(y_final, train_pred)
        test_metric = suite.metric(final_arrays["y"]["test"], test_pred)
        neural_test = suite.metric(
            final_arrays["y"]["test"], neural_predictions(model, final_arrays["x"]["test"])
        )
        run_records.append({
            "experiment": spec.name, "family": spec.family, "backbone": spec.backbone,
            "method": spec.method, "probe": spec.probe, "split": split_name, "seed": seed,
            "latent_dimensions": model.latent_dim,
            "probe_dimensions": model.latent_dim + (54 if spec.probe == "fusion" else 0),
            "alignment_strength": FIXED_STRENGTHS[spec.method]["alignment"],
            "centroid_strength": FIXED_STRENGTHS[spec.method]["centroid"],
            "best_epoch": selected_epochs[seed], "lr_C": LR_C,
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
            curve_records.append({
                "experiment": spec.name, "family": spec.family, "method": spec.method,
                "split": split_name, "seed": seed, **row,
            })
        append_predictions(
            prediction_records, spec, split_name, seed, parts["target_test"],
            final_arrays["y"]["test"], test_pred,
        )
        save(pd.DataFrame(run_records), output_paths["runs"])
        save(pd.DataFrame(prediction_records), output_paths["predictions"])
        save(pd.DataFrame(curve_records), output_paths["curves"])


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, split_name), group in runs.groupby(["experiment", "split"], sort=False):
        first = group.iloc[0]
        row = {
            "experiment": experiment, "family": first["family"], "backbone": first["backbone"],
            "method": first["method"], "probe": first["probe"], "split": split_name,
            "latent_dimensions": first["latent_dimensions"], "probe_dimensions": first["probe_dimensions"],
            "seeds": ",".join(map(str, sorted(group.seed.astype(int).unique()))),
        }
        for metric in (
            "train_accuracy", "train_balanced_accuracy", "train_macro_f1",
            "neural_test_accuracy", "neural_test_balanced_accuracy", "neural_test_macro_f1",
            "test_accuracy", "test_balanced_accuracy", "test_macro_f1", "best_epoch",
        ):
            values = group[metric].astype(float)
            row[metric] = float(values.mean()); row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_min"] = float(values.min()); row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def write_protocol(args: argparse.Namespace, output_paths: dict[str, Path]) -> None:
    payload = {
        "status": "exploratory strict-inductive reanalysis of previously inspected data",
        "classes": suite.CLASSES, "features": 54, "seeds": list(SEEDS), "epochs": EPOCHS,
        "requested_experiments": list(args.experiments), "requested_splits": list(args.splits),
        "test_boundary": "target test features and labels excluded from all training and selection",
        "selection": "validation selects epoch only; method strengths and LR C are fixed",
        "fixed_strengths": FIXED_STRENGTHS, "LR_C": LR_C,
        "backbones": {
            "linear54": "identity-initialized trainable Linear(54,54)",
            "residual54": "h=x+F(x), F:54->64->54, zero-initialized final layer",
            "compressed16": "54->16 ReLU/Dropout feature extractor",
        },
        "naming": {
            "CDAN": "T(h,g)=g outer h conditional domain discriminator",
            "C-DANN": "project custom DANN plus class-wise centroid loss",
        },
        "experiments": {name: spec.__dict__ for name, spec in SPECS.items()},
    }
    output_paths["protocol"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args(); output_paths = paths(); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(FEATURE_PATH); columns = suite.feature_columns(raw)
    split_map = {item["split"]: item for item in strict.STRICT_SPLITS}
    unknown = set(args.splits) - set(split_map)
    if unknown:
        raise ValueError(f"Unknown strict splits: {sorted(unknown)}")
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
        spec = SPECS[experiment]
        for split_name in args.splits:
            number += 1
            if (experiment, split_name) in complete:
                continue
            print(f"[{number}/{total}] {experiment} | {split_name}", flush=True)
            run_records = [row for row in run_records if not (row["experiment"] == experiment and row["split"] == split_name)]
            prediction_records = [row for row in prediction_records if not (row["experiment"] == experiment and row["split"] == split_name)]
            curve_records = [row for row in curve_records if not (row["experiment"] == experiment and row["split"] == split_name)]
            tuning_records = [row for row in tuning_records if not (row["experiment"] == experiment and row["split"] == split_name)]
            split = split_map[split_name]; parts = strict.strict_parts(raw, split)
            train_arrays = strict.arrays_from_parts(parts, columns, False)
            final_arrays = strict.arrays_from_parts(parts, columns, True)
            run_group(
                spec, split, parts, train_arrays, final_arrays, tuning_records, run_records,
                prediction_records, curve_records, output_paths,
            )
    runs = pd.DataFrame(run_records)
    requested = runs[runs.experiment.isin(args.experiments) & runs.split.isin(args.splits)]
    expected = len(args.experiments) * len(args.splits) * len(SEEDS)
    if len(requested) != expected:
        raise RuntimeError(f"Expected {expected} requested rows, found {len(requested)}.")
    save(summarize(runs), output_paths["summary"]); write_protocol(args, output_paths)
    print(f"Outputs written to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
