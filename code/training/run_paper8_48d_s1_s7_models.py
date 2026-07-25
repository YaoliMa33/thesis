"""Paper-defined 8 features x 6 sensors on the original S1-S7 protocol.

Per sensor, this runner extracts the eight features in Vergara et al.
(Sensors and Actuators B 166-167, 2012, Table 3):

* steady state: DeltaR and DeltaR / min(r);
* rising transient: max EMA of first differences for alpha 0.001, 0.01, 0.1;
* decaying transient: min EMA of first differences for the same alpha values.

The exposure and recovery windows use the established phase parser already used
for the 54D table. Raw positive sensor values are used for the normalized steady
feature, exactly as required by the denominator in the paper. EMA difference
features are invariant to an additive baseline.

Outputs are independent from every 54D artifact. The DA experiments retain the
author-requested original S1-S7 transductive protocol: target-test features enter
target_domain without gas labels, while target-test gas labels are evaluation-only.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_dann_multi_splits as existing_dann
import run_manifest_da_baseline_splits as mlbase
import run_unified_da_source_target_suite as suite
import shared_experiment_protocol as protocol


ROOT = Path(r"D:\thesis")
OUTPUT_DIR = ROOT / "tables" / "paper8_48d_s1_s7"
FEATURE_PATH = OUTPUT_DIR / "paper8_48d_features.csv"
SENSORS = tuple(f"s{i}" for i in range(1, 7))
ALPHAS = (0.001, 0.01, 0.1)
SEEDS = tuple(protocol.MODEL_SEEDS)
EPOCHS = int(protocol.NEURAL_EPOCHS)
EVAL_EVERY = 10
LR_C = 1.0

MODEL_CHOICES = (
    "logistic-regression",
    "random-forest",
    "rbf-svm",
    "linear48-dann",
    "fusion48-mk-mmd",
)
ML_NAMES = {
    "logistic-regression": "Logistic Regression",
    "random-forest": "Random Forest",
    "rbf-svm": "RBF SVM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(MODEL_CHOICES))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=[item["split"] for item in protocol.DA_SPLITS],
        help="Original split names or unambiguous aliases S1 through S7.",
    )
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


def resolve_splits(requested: list[str]) -> list[str]:
    available = [item["split"] for item in protocol.DA_SPLITS]
    aliases = {name.split("_", 1)[0].upper(): name for name in available}
    resolved = [aliases.get(value.upper(), value) for value in requested]
    unknown = sorted(set(resolved) - set(available))
    if unknown:
        raise ValueError(f"Unknown S1-S7 splits: {unknown}")
    if len(resolved) != len(set(resolved)):
        raise ValueError("Duplicate splits were requested after resolving S1-S7 aliases.")
    return resolved


def paths() -> dict[str, Path]:
    return {
        "ml_runs": OUTPUT_DIR / "ml_runs.csv",
        "da_runs": OUTPUT_DIR / "da_runs.csv",
        "predictions": OUTPUT_DIR / "predictions.csv",
        "curves": OUTPUT_DIR / "curves.csv",
        "summary": OUTPUT_DIR / "summary.csv",
        "best": OUTPUT_DIR / "best_by_split.csv",
        "protocol": OUTPUT_DIR / "protocol.json",
    }


def save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def read_records(path: Path) -> list[dict]:
    return pd.read_csv(path).to_dict("records") if path.exists() else []


def alpha_name(alpha: float) -> str:
    return {0.001: "0001", 0.01: "001", 0.1: "01"}[alpha]


def ema_difference(values: np.ndarray, alpha: float, path: Path, sensor: str, phase: str) -> np.ndarray:
    if values.ndim != 1 or len(values) < 2:
        raise ValueError(f"{path} {sensor} {phase} requires at least two samples.")
    output = np.empty(len(values) - 1, dtype=float)
    state = 0.0
    for index, difference in enumerate(np.diff(values)):
        state = (1.0 - alpha) * state + alpha * float(difference)
        output[index] = state
    if not np.isfinite(output).all():
        raise ValueError(f"{path} {sensor} {phase} EMA alpha={alpha} is non-finite.")
    return output


def extract_paper8_features(path: Path, meta: dict) -> dict:
    raw = mlbase.load_raw_measurements(path)
    t = mlbase.elapsed_seconds(raw, path)
    duration = (
        float(pd.to_numeric(raw["duration_s"], errors="raise").iloc[0])
        if "duration_s" in raw.columns else float(np.max(t))
    )
    _, exposure_mask, recovery_mask = mlbase.phase_masks(str(meta["label"]), duration, t, path)
    row = {
        "sample_id": meta["sample_id"], "label": meta["label"],
        "target_file": meta["target_file"], "period": meta["period"],
        "day": meta["day"], "day_num": int(meta["day_num"]),
        "day_label": meta.get("day_label", ""), "batch": meta.get("batch", ""),
        "source_kind": meta.get("source_kind", ""),
    }
    for sensor in SENSORS:
        values = pd.to_numeric(raw[sensor], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{path} {sensor} contains non-finite values.")
        rising, decaying = values[exposure_mask], values[recovery_mask]
        minimum = float(np.min(rising))
        if minimum <= 1e-9:
            raise ValueError(f"{path} {sensor} rising minimum {minimum} is invalid for normalized DeltaR.")
        delta = float(np.max(rising) - minimum)
        row[f"{sensor}_delta_r"] = delta
        row[f"{sensor}_delta_r_norm"] = delta / minimum
        for alpha in ALPHAS:
            name = alpha_name(alpha)
            row[f"{sensor}_ema_rise_a{name}"] = float(np.max(ema_difference(rising, alpha, path, sensor, "rising")))
            row[f"{sensor}_ema_decay_a{name}"] = float(np.min(ema_difference(decaying, alpha, path, sensor, "decaying")))
    return row


def feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if column.startswith(tuple(f"{sensor}_" for sensor in SENSORS))]
    if len(columns) != 48:
        raise ValueError(f"Expected 48 paper feature columns, found {len(columns)}.")
    return columns


def build_or_load_features(rebuild: bool) -> pd.DataFrame:
    if FEATURE_PATH.exists() and not rebuild:
        frame = pd.read_csv(FEATURE_PATH)
    else:
        manifest = mlbase.load_current_manifest()
        rows = [extract_paper8_features(Path(meta["raw_path"]), meta) for meta in manifest.to_dict("records")]
        frame = pd.DataFrame(rows)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_csv(FEATURE_PATH, index=False)
    columns = feature_columns(frame)
    if len(frame) != 303:
        raise ValueError(f"Expected 303 trials, found {len(frame)}.")
    if frame["label"].value_counts().to_dict() != {"acetone": 108, "alcohol": 108, "air": 87}:
        raise ValueError(f"Unexpected class counts: {frame['label'].value_counts().to_dict()}")
    values = frame[columns].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("48D feature table contains NaN or infinite values.")
    return frame


def encode_labels(values: pd.Series) -> np.ndarray:
    return mlbase.encode(values.astype(str).to_numpy(), list(protocol.THREE_CLASSES))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def expected_runs(model_slug: str) -> int:
    return len(SEEDS) if model_slug in {"random-forest", "linear48-dann", "fusion48-mk-mmd"} else 1


def run_ml_model(
    frame: pd.DataFrame,
    columns: list[str],
    split: dict,
    model_slug: str,
    run_records: list[dict],
    prediction_records: list[dict],
    output_paths: dict[str, Path],
) -> None:
    model_name = ML_NAMES[model_slug]
    train = frame.loc[split["train"](frame)].copy()
    val = frame.loc[split["val"](frame)].copy()
    test = frame.loc[split["test"](frame)].copy()
    x_train, y_train = train[columns].to_numpy(float), encode_labels(train["label"])
    x_val, y_val = val[columns].to_numpy(float), encode_labels(val["label"])
    x_test, y_test = test[columns].to_numpy(float), encode_labels(test["label"])
    if len(val):
        best_params, _ = mlbase.select_params(model_name, x_train, y_train, x_val, y_val, len(protocol.THREE_CLASSES))
        final = pd.concat([train, val], axis=0)
        final_training = "train+validation refit"
    else:
        best_params = mlbase.DEFAULT_PARAMS[model_name]
        final = train
        final_training = "train only; fixed default"
    x_final, y_final = final[columns].to_numpy(float), encode_labels(final["label"])
    seeds = mlbase.seeds_for_model(model_name)
    for seed in seeds:
        estimator = mlbase.fit_estimator(mlbase.make_estimator(model_name, best_params, seed), x_final, y_final)
        train_pred, test_pred = estimator.predict(x_final), estimator.predict(x_test)
        train_metric, test_metric = metrics(y_final, train_pred), metrics(y_test, test_pred)
        run_records.append({
            "model": model_slug, "model_label": model_name, "family": "ML raw 48D",
            "split": split["split"], "seed": seed, "best_params": mlbase.serialize_params(best_params),
            "final_training": final_training,
            **{f"train_{key}": value for key, value in train_metric.items()},
            **{f"test_{key}": value for key, value in test_metric.items()},
        })
        for position, (index, sample) in enumerate(test.iterrows()):
            prediction_records.append({
                "model": model_slug, "split": split["split"], "seed": seed,
                "sample_index": int(index), "sample_id": sample["sample_id"],
                "true_label": protocol.THREE_CLASSES[int(y_test[position])],
                "predicted_label": protocol.THREE_CLASSES[int(test_pred[position])],
                "correct": bool(y_test[position] == test_pred[position]),
            })
        save(pd.DataFrame(run_records), output_paths["ml_runs"])
        save(pd.DataFrame(prediction_records), output_paths["predictions"])


@dataclass(frozen=True)
class DASpec:
    name: str
    backbone: str
    method: str
    probe: str
    alignment_strength: float


DA_SPECS = {
    "linear48-dann": DASpec("linear48-dann", "linear48", "dann", "latent", 0.2),
    "fusion48-mk-mmd": DASpec("fusion48-mk-mmd", "compressed16", "mk-mmd", "fusion", 0.1),
}


class Paper48DAModel(nn.Module):
    def __init__(self, backbone: str) -> None:
        super().__init__()
        if backbone == "linear48":
            self.extractor = nn.Linear(48, 48)
            nn.init.eye_(self.extractor.weight); nn.init.zeros_(self.extractor.bias)
            latent = 48
        elif backbone == "compressed16":
            self.extractor = suite.FeatureExtractor(48, (16,))
            latent = 16
        else:
            raise ValueError(f"Unknown 48D backbone: {backbone}")
        self.latent_dim = latent
        self.label_classifier = nn.Linear(latent, len(protocol.THREE_CLASSES))
        self.domain_classifier = nn.Sequential(nn.Linear(latent, 16), nn.ReLU(), nn.Linear(16, 2))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor(x)

    def class_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.label_classifier(self.features(x))

    def domain_logits(self, h: torch.Tensor, strength: float) -> torch.Tensor:
        return self.domain_classifier(existing_dann.grad_reverse(h, strength))


def da_predictions(model: Paper48DAModel, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.class_logits(existing_dann.to_tensor(x)).argmax(1).cpu().numpy()


def da_probe(spec: DASpec, model: Paper48DAModel, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        h = model.features(existing_dann.to_tensor(x)).cpu().numpy()
    return np.concatenate([x, h], axis=1) if spec.probe == "fusion" else h


def train_da(
    arrays: dict,
    spec: DASpec,
    mode: str,
    seed: int,
    include_validation: bool,
    fixed_epochs: int | None = None,
) -> tuple[Paper48DAModel, int, list[dict]]:
    x_cls, y_cls, _ = suite.classifier_data(arrays, mode, include_validation)
    x_select, y_select, _ = suite.selection_data(arrays, mode)
    if include_validation:
        x_select = y_select = None
    existing_dann.set_random_seed(seed)
    model = Paper48DAModel(spec.backbone)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol.NEURAL_LEARNING_RATE, weight_decay=protocol.NEURAL_WEIGHT_DECAY)
    x_cls_t, y_cls_t = existing_dann.to_tensor(x_cls), existing_dann.to_long(y_cls)
    class_weight = existing_dann.class_weights(y_cls, len(protocol.THREE_CLASSES))
    xs, xt = existing_dann.to_tensor(arrays["x"]["source_train"]), existing_dann.to_tensor(arrays["x"]["target_domain"])
    domain_y = torch.cat([torch.zeros(len(xs), dtype=torch.long), torch.ones(len(xt), dtype=torch.long)])
    best_state, best_score, best_epoch = copy.deepcopy(model.state_dict()), (-1.0, -1.0, -1.0), 0
    history = []
    epochs = int(fixed_epochs or EPOCHS)
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad()
        class_loss = F.cross_entropy(model.class_logits(x_cls_t), y_cls_t, weight=class_weight)
        hs, ht = model.features(xs), model.features(xt)
        if spec.method == "dann":
            alignment = F.cross_entropy(model.domain_logits(torch.cat([hs, ht]), spec.alignment_strength), domain_y)
            total = class_loss + alignment
        elif spec.method == "mk-mmd":
            alignment = suite.mkmmd_loss(hs, ht)
            total = class_loss + spec.alignment_strength * alignment
        else:
            raise ValueError(spec.method)
        total.backward(); optimizer.step()
        if epoch == 1 or epoch % EVAL_EVERY == 0 or epoch == epochs:
            val = metrics(y_select, da_predictions(model, x_select)) if x_select is not None else {
                "accuracy": np.nan, "balanced_accuracy": np.nan, "macro_f1": np.nan,
            }
            history.append({
                "epoch": epoch, "classification_loss": float(class_loss.detach()),
                "alignment_loss": float(alignment.detach()), "total_loss": float(total.detach()),
                "validation_accuracy": val["accuracy"], "validation_balanced_accuracy": val["balanced_accuracy"],
                "validation_macro_f1": val["macro_f1"],
            })
            if x_select is not None:
                score = (val["macro_f1"], val["balanced_accuracy"], val["accuracy"])
                if score > best_score:
                    best_score, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        if x_select is None and epoch == epochs:
            best_epoch, best_state = epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_epoch, history


def mode_for(split_name: str) -> str:
    return "uda" if int(split_name[1]) <= 3 else "semi"


def da_final_arrays(parts: dict, columns: list[str], mode: str) -> tuple[dict, bool, str]:
    initial = suite.make_arrays(parts, columns)
    if mode == "semi":
        return suite.make_arrays(parts, columns, ("source_train", "target_labeled_train", "target_val")), True, "Source+Target refit"
    if len(initial["x"]["source_val"]):
        return suite.make_arrays(parts, columns, ("source_train", "source_val")), True, "source train+validation refit"
    return initial, False, "source-only; no refit"


def run_da_model(
    frame: pd.DataFrame,
    columns: list[str],
    split: dict,
    model_slug: str,
    run_records: list[dict],
    prediction_records: list[dict],
    curve_records: list[dict],
    output_paths: dict[str, Path],
) -> None:
    spec, mode = DA_SPECS[model_slug], mode_for(split["split"])
    parts = suite.partitions(frame, split)
    initial = suite.make_arrays(parts, columns)
    epochs = {}
    for seed in SEEDS:
        _, best_epoch, _ = train_da(initial, spec, mode, seed, False)
        epochs[seed] = best_epoch
    final, include_validation, final_training = da_final_arrays(parts, columns, mode)
    x_final, y_final, _ = suite.classifier_data(final, mode, include_validation)
    for seed in SEEDS:
        model, _, history = train_da(final, spec, mode, seed, include_validation, epochs[seed])
        classifier = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=LR_C, class_weight="balanced", max_iter=2000, solver="lbfgs")),
        ]).fit(da_probe(spec, model, x_final), y_final)
        train_pred = classifier.predict(da_probe(spec, model, x_final))
        test_pred = classifier.predict(da_probe(spec, model, final["x"]["target_test"]))
        train_metric, test_metric = metrics(y_final, train_pred), metrics(final["y"]["target_test"], test_pred)
        run_records.append({
            "model": model_slug, "model_label": model_slug, "family": "DA 48D",
            "split": split["split"], "mode": mode, "seed": seed,
            "best_params": f"alignment_strength={spec.alignment_strength};LR_C={LR_C}",
            "best_epoch": epochs[seed], "final_training": final_training,
            **{f"train_{key}": value for key, value in train_metric.items()},
            **{f"test_{key}": value for key, value in test_metric.items()},
        })
        for row in history:
            curve_records.append({"model": model_slug, "split": split["split"], "mode": mode, "seed": seed, **row})
        for position, (index, sample) in enumerate(parts["target_test"].iterrows()):
            prediction_records.append({
                "model": model_slug, "split": split["split"], "seed": seed,
                "sample_index": int(index), "sample_id": sample["sample_id"],
                "true_label": protocol.THREE_CLASSES[int(final["y"]["target_test"][position])],
                "predicted_label": protocol.THREE_CLASSES[int(test_pred[position])],
                "correct": bool(final["y"]["target_test"][position] == test_pred[position]),
            })
        save(pd.DataFrame(run_records), output_paths["da_runs"])
        save(pd.DataFrame(prediction_records), output_paths["predictions"])
        save(pd.DataFrame(curve_records), output_paths["curves"])


def summarize(ml_runs: pd.DataFrame, da_runs: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([ml_runs, da_runs], ignore_index=True, sort=False)
    rows = []
    for (model, split), group in combined.groupby(["model", "split"], sort=False):
        first = group.iloc[0]
        row = {"model": model, "model_label": first["model_label"], "family": first["family"], "split": split}
        for metric in ("train_accuracy", "test_accuracy", "test_balanced_accuracy", "test_macro_f1"):
            values = group[metric].astype(float)
            row[metric] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_min"] = float(values.min()); row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def write_protocol(args: argparse.Namespace, output_paths: dict[str, Path]) -> None:
    payload = {
        "paper_features": {
            "citation": "Vergara et al., Sensors and Actuators B 166-167 (2012), Table 3",
            "per_sensor": ["DeltaR", "DeltaR/minR", "rise EMA max x3", "decay EMA min x3"],
            "alphas": list(ALPHAS), "sensors": list(SENSORS), "dimensions": 48,
        },
        "phase_mapping": "established project exposure mask=rising; recovery mask=decaying",
        "classes": list(protocol.THREE_CLASSES), "seeds": list(SEEDS), "epochs": EPOCHS,
        "requested_models": list(args.models), "requested_splits": list(args.splits),
        "DA_protocol": "original S1-S7 transductive; target-test features unlabeled in target_domain",
        "target_test_labels": "evaluation-only",
    }
    output_paths["protocol"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.splits = resolve_splits(args.splits)
    output_paths = paths(); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = build_or_load_features(args.rebuild_features); columns = feature_columns(frame)
    da_split_map = {item["split"]: item for item in protocol.DA_SPLITS}
    ml_split_map = {item["split"]: item for item in protocol.ML_SPLITS}
    ml_records = read_records(output_paths["ml_runs"])
    da_records = read_records(output_paths["da_runs"])
    prediction_records = read_records(output_paths["predictions"])
    curve_records = read_records(output_paths["curves"])
    existing = pd.concat([pd.DataFrame(ml_records), pd.DataFrame(da_records)], ignore_index=True, sort=False)
    complete = set()
    if not existing.empty:
        counts = existing.groupby(["model", "split"]).size()
        complete = {(model, split) for (model, split), count in counts.items() if int(count) == expected_runs(model)}
    total = len(args.models) * len(args.splits); number = 0
    for model_slug in args.models:
        for split_name in args.splits:
            number += 1
            if (model_slug, split_name) in complete:
                continue
            print(f"[{number}/{total}] {model_slug} | {split_name}", flush=True)
            ml_records = [row for row in ml_records if not (row["model"] == model_slug and row["split"] == split_name)]
            da_records = [row for row in da_records if not (row["model"] == model_slug and row["split"] == split_name)]
            prediction_records = [row for row in prediction_records if not (row["model"] == model_slug and row["split"] == split_name)]
            curve_records = [row for row in curve_records if not (row["model"] == model_slug and row["split"] == split_name)]
            if model_slug in ML_NAMES:
                run_ml_model(frame, columns, ml_split_map[split_name], model_slug, ml_records, prediction_records, output_paths)
            else:
                run_da_model(frame, columns, da_split_map[split_name], model_slug, da_records, prediction_records, curve_records, output_paths)
    ml_frame, da_frame = pd.DataFrame(ml_records), pd.DataFrame(da_records)
    summary = summarize(ml_frame, da_frame)
    requested = summary[summary.model.isin(args.models) & summary.split.isin(args.splits)]
    if len(requested) != len(args.models) * len(args.splits):
        raise RuntimeError(f"Expected {len(args.models) * len(args.splits)} summary rows, found {len(requested)}.")
    save(summary, output_paths["summary"])
    best = summary.sort_values(["split", "test_accuracy", "test_balanced_accuracy"], ascending=[True, False, False]).groupby("split", as_index=False).first()
    save(best, output_paths["best"]); write_protocol(args, output_paths)
    print(f"Outputs written to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
