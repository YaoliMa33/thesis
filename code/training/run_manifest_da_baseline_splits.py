from __future__ import annotations

import os
import re
import warnings
import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import shared_experiment_protocol as protocol


OUT_ROOT = Path(r"D:\thesis")
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures" / "ml_learning"
DATA_ROOT = OUT_ROOT / "enose_data"
FEATURE_PATH = TABLE_DIR / "manifest_baseline_as_air_features.csv"
RAW_MANIFEST_PATH = TABLE_DIR / "manifest_reconstructed_from_raw.csv"
CURRENT_MANIFEST_PATH = TABLE_DIR / "feature_space_current_manifest_samples.csv"
HTML_REPORT_PATH = OUT_ROOT / "reports" / "ml_dann_coral_research_flowcharts.html"
INTERACTIVE_REPORT_PATH = OUT_ROOT / "reports" / "ml_results_confusion_explorer.html"
CORE_TEST_PREDICTIONS_PATH = TABLE_DIR / "manifest_da_baseline_split_test_predictions.csv"
PB_TEST_PREDICTIONS_PATH = TABLE_DIR / "manifest_da_baseline_period_i_per_batch_test_predictions.csv"
INTERACTIVE_PREDICTIONS_PATH = TABLE_DIR / "manifest_da_baseline_interactive_test_predictions.csv"

SENSORS = [f"s{i}" for i in range(1, 7)]
THREE_CLASSES = protocol.THREE_CLASSES
MODELS = ["LDA", "Logistic Regression", "kNN", "Random Forest", "RBF SVM", "MLP"]
BATCH_ORDER = protocol.BATCH_ORDER
FINAL_TRAINING = os.environ.get("ENOSE_FINAL_TRAINING", protocol.ML_FINAL_TRAINING).strip().lower()
REBUILD_FEATURES_FROM_RAW = os.environ.get("ENOSE_REBUILD_FEATURES_FROM_RAW", "1").strip() == "1"
RUN_SCOPE = os.environ.get("ENOSE_RUN_SCOPE", "core").strip().lower()
RANDOM_SEEDS = protocol.MODEL_SEEDS
STOCHASTIC_MODELS = {"Random Forest", "MLP"}

MODEL_COLORS = {
    "LDA": "#0072B2",
    "Logistic Regression": "#D55E00",
    "kNN": "#009E73",
    "Random Forest": "#CC79A7",
    "RBF SVM": "#E69F00",
    "MLP": "#56B4E9",
}
PUBLICATION_FIGURES = {
    "heatmap": "manifest_da_baseline_model_split_heatmap",
    "point_ranges": "manifest_da_baseline_split_model_point_ranges",
    "batch_trends": "manifest_da_baseline_period_i_per_batch_trends",
    "pb_table": "manifest_da_baseline_period_i_per_batch_table",
}

GRIDS = {
    "LDA": [{"shrinkage": v} for v in [0.01, 0.1, 0.25, 0.5]],
    "Logistic Regression": [{"C": v} for v in [0.1, 1.0, 10.0]],
    "kNN": [{"n_neighbors": v} for v in [1, 3, 5, 7]],
    "Random Forest": protocol.RF_GRID,
    "RBF SVM": [{"C": c, "gamma": g} for c in [0.1, 1.0, 10.0] for g in ["scale", 0.001, 0.01]],
    "MLP": [{
        "hidden_layer_sizes": protocol.NEURAL_HIDDEN_DIMS,
        "alpha": protocol.NEURAL_WEIGHT_DECAY,
        "learning_rate_init": protocol.NEURAL_LEARNING_RATE,
    }],
}

DEFAULT_PARAMS = {
    "LDA": {"shrinkage": 0.1},
    "Logistic Regression": {"C": 1.0},
    "kNN": {"n_neighbors": 3},
    "Random Forest": protocol.RF_DEFAULT,
    "RBF SVM": {"C": 1.0, "gamma": "scale"},
    "MLP": {
        "hidden_layer_sizes": protocol.NEURAL_HIDDEN_DIMS,
        "alpha": protocol.NEURAL_WEIGHT_DECAY,
        "learning_rate_init": protocol.NEURAL_LEARNING_RATE,
    },
}

SPLITS = protocol.ML_SPLITS

EXPERIMENT_SPLITS = SPLITS

PERIOD_I_TRAINED_BATCH_TEST_SPLITS = protocol.PERIOD_I_PER_BATCH_SPLITS

TASKS = [
    ("three_class_with_air", THREE_CLASSES),
]


def natural_key(value: str) -> tuple:
    return protocol.natural_key(value)


def later_period_mask(df: pd.DataFrame) -> pd.Series:
    return protocol.target_period_mask(df)


def batch_in(df: pd.DataFrame, batches: list[str]) -> pd.Series:
    if "batch" not in df.columns:
        raise ValueError("Batch-based splits require a 'batch' column in the feature table.")
    return df["batch"].isin(batches)


def period_i_internal_mask(df: pd.DataFrame, partition: str, scheme: str = "25_5") -> pd.Series:
    protocol_partition = "source_train" if partition == "train" else "source_val"
    return protocol.period_i_holdout(df, protocol_partition, scheme)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if re.match(r"s\d+_", c)]


def raw_path_for_metadata(row: pd.Series) -> Path:
    label = str(row["label"])
    target_file = str(row["target_file"]).replace("\\", "/")
    stem = Path(target_file).stem
    if label in {"alcohol", "acetone"}:
        old_match = re.fullmatch(fr"{label}_o(\d+)", stem)
        new_match = re.fullmatch(fr"{label}_(\d+)", stem)
        if old_match:
            return DATA_ROOT / label / "enose_data_old" / f"{label}_{old_match.group(1)}.csv"
        if new_match:
            return DATA_ROOT / label / "enose_data" / f"{label}_{new_match.group(1)}.csv"
    if label == "air":
        old_match = re.fullmatch(r"air_ob(\d+)", stem)
        new_match = re.fullmatch(r"air_(\d+)", stem)
        ref_match = re.fullmatch(r"air_ref(\d+)", stem)
        if old_match:
            return DATA_ROOT / "air" / "enose_data_old" / "baseline" / f"baseline_{old_match.group(1)}.csv"
        if new_match:
            return DATA_ROOT / "air" / "enose_data" / "air" / f"air_{new_match.group(1)}.csv"
        if ref_match:
            return DATA_ROOT / "air" / "enose_data" / "reference" / f"reference_{ref_match.group(1)}.csv"
    raise ValueError(f"Cannot map metadata target_file to a raw CSV path: label={label}, target_file={target_file}")


def build_reconstructed_manifest_from_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    required = ["label", "target_file", "period", "day", "day_num"]
    missing = [col for col in required if col not in metadata.columns]
    if missing:
        raise ValueError(f"Metadata table is missing required columns for manifest reconstruction: {missing}")
    rows = []
    for row in metadata[required].drop_duplicates().to_dict("records"):
        series = pd.Series(row)
        raw_path = raw_path_for_metadata(series)
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw CSV required by reconstructed manifest does not exist: {raw_path}")
        rows.append(
            {
                **row,
                "raw_path": str(raw_path),
            }
        )
    manifest = pd.DataFrame(rows).sort_values(["label", "period", "day_num", "target_file"], key=lambda s: s.map(str)).reset_index(drop=True)
    expected_counts = {"acetone": 92, "air": 73, "alcohol": 92}
    counts = manifest["label"].value_counts().to_dict()
    if counts != expected_counts:
        raise ValueError(f"Reconstructed manifest label counts do not match expected raw data counts: {counts} != {expected_counts}")
    return manifest


def load_current_manifest() -> pd.DataFrame:
    if not CURRENT_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Current manifest required for the all-data ML run does not exist: {CURRENT_MANIFEST_PATH}")
    manifest = pd.read_csv(CURRENT_MANIFEST_PATH)
    required = ["sample_id", "label", "target_file", "period", "day", "day_num", "batch", "source_kind"]
    missing = [col for col in required if col not in manifest.columns]
    if missing:
        raise ValueError(f"Current manifest is missing required columns: {missing}")
    invalid_labels = sorted(set(manifest["label"].astype(str)).difference(THREE_CLASSES))
    if invalid_labels:
        raise ValueError(f"Current manifest contains labels outside the three-class task: {invalid_labels}")
    duplicate_files = manifest["target_file"][manifest["target_file"].duplicated()].astype(str).tolist()
    if duplicate_files:
        raise ValueError(f"Current manifest has duplicate target_file entries: {duplicate_files[:10]}")

    manifest = manifest.copy()
    manifest["raw_path"] = manifest["target_file"].map(lambda value: str(DATA_ROOT / str(value).replace("\\", "/")))
    missing_files = [path for path in manifest["raw_path"].astype(str) if not Path(path).exists()]
    if missing_files:
        raise FileNotFoundError(f"Current manifest references missing raw files: {missing_files[:10]}")
    batch_values = set(manifest["batch"].astype(str))
    expected_batches = {"source", *BATCH_ORDER}
    unexpected_batches = sorted(batch_values.difference(expected_batches))
    if unexpected_batches:
        raise ValueError(f"Current manifest contains unrecognized batch values: {unexpected_batches}")
    for batch in BATCH_ORDER:
        sub = manifest[manifest["batch"].eq(batch)]
        missing_classes = sorted(set(THREE_CLASSES).difference(set(sub["label"].astype(str))))
        if missing_classes:
            raise ValueError(f"{batch} is missing classes required for batch split evaluation: {missing_classes}")
    return manifest.sort_values(["batch", "label", "day_num", "target_file"], key=lambda s: s.map(str)).reset_index(drop=True)


def load_or_create_reconstructed_manifest() -> pd.DataFrame:
    if RAW_MANIFEST_PATH.exists():
        manifest = pd.read_csv(RAW_MANIFEST_PATH)
    else:
        if not FEATURE_PATH.exists():
            raise FileNotFoundError(
                f"{RAW_MANIFEST_PATH} does not exist and {FEATURE_PATH} is unavailable for one-time metadata reconstruction."
            )
        metadata = pd.read_csv(FEATURE_PATH)
        manifest = build_reconstructed_manifest_from_metadata(metadata)
        manifest.to_csv(RAW_MANIFEST_PATH, index=False)
    required = ["label", "target_file", "period", "day", "day_num", "raw_path"]
    missing = [col for col in required if col not in manifest.columns]
    if missing:
        raise ValueError(f"Reconstructed manifest is missing required columns: {missing}")
    missing_files = [path for path in manifest["raw_path"].astype(str) if not Path(path).exists()]
    if missing_files:
        raise FileNotFoundError(f"Reconstructed manifest references missing raw files: {missing_files[:10]}")
    return manifest


def elapsed_seconds(raw: pd.DataFrame, path: Path) -> np.ndarray:
    if "arduino_time" in raw.columns:
        t = pd.to_numeric(raw["arduino_time"], errors="raise").to_numpy(float)
        return (t - t[0]) / 1000.0
    if "time_s" in raw.columns:
        t = pd.to_numeric(raw["time_s"], errors="raise").to_numpy(float)
        return t - t[0]
    raise ValueError(f"{path} has neither arduino_time nor time_s.")


def load_raw_measurements(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    time_cols = [col for col in ["arduino_time", "time_s"] if col in raw.columns]
    if not time_cols:
        raise ValueError(f"{path} has neither arduino_time nor time_s.")
    required = time_cols + SENSORS
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    numeric_required = raw[required].apply(pd.to_numeric, errors="coerce")
    empty_rows = numeric_required.isna().all(axis=1)
    partial_invalid_rows = numeric_required.isna().any(axis=1) & ~empty_rows
    if partial_invalid_rows.any():
        bad_rows = raw.index[partial_invalid_rows].tolist()[:10]
        raise ValueError(f"{path} has non-numeric or partially missing measurement rows: {bad_rows}")
    if empty_rows.any():
        raw = raw.loc[~empty_rows].copy()
    if raw.empty:
        raise ValueError(f"{path} has no valid measurement rows.")
    return raw


def phase_masks(label: str, duration: float, t: np.ndarray, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(t) == 0:
        raise ValueError(f"{path} has no time samples.")
    total = float(np.max(t))
    if label == "air":
        baseline_end = min(60.0, max(20.0, total * 0.35))
        exposure_start = baseline_end
        exposure_end = total
        recovery_start = max(exposure_start, total * 0.70)
    else:
        baseline_end = min(60.0, max(5.0, total * 0.45))
        exposure_start = 60.0
        if duration >= 175:
            exposure_end = 120.0
        elif duration >= 128:
            exposure_end = 70.0
        elif duration >= 123:
            exposure_end = 65.0
        else:
            exposure_end = min(total, exposure_start + max(1.0, duration - 120.0))
        exposure_end = min(total, exposure_end)
        recovery_start = exposure_end
    baseline = t <= baseline_end
    exposure = (t >= exposure_start) & (t <= exposure_end)
    recovery = t >= recovery_start
    if baseline.sum() < 3 or exposure.sum() < 3 or recovery.sum() < 3:
        raise ValueError(
            f"{path} has insufficient phase samples: baseline={baseline.sum()}, exposure={exposure.sum()}, recovery={recovery.sum()}."
        )
    return baseline, exposure, recovery


def slope(x: np.ndarray, y: np.ndarray, path: Path, sensor: str) -> float:
    if len(x) < 2:
        raise ValueError(f"{path} {sensor} recovery phase has fewer than two samples.")
    xm = x - np.mean(x)
    ym = y - np.mean(y)
    denom = float(np.dot(xm, xm))
    if denom <= 0:
        raise ValueError(f"{path} {sensor} recovery time vector has zero variance.")
    return float(np.dot(xm, ym) / denom)


def extract_legacy9_features(path: Path, meta: dict) -> dict:
    raw = load_raw_measurements(path)
    label = str(meta["label"])
    t = elapsed_seconds(raw, path)
    duration = float(pd.to_numeric(raw["duration_s"], errors="raise").iloc[0]) if "duration_s" in raw.columns else float(np.max(t))
    bmask, emask, rmask = phase_masks(label, duration, t, path)
    row = {
        "sample_id": meta.get("sample_id", Path(str(meta["target_file"])).stem),
        "label": label,
        "target_file": meta["target_file"],
        "period": meta["period"],
        "day": meta["day"],
        "day_num": int(meta["day_num"]),
        "day_label": meta.get("day_label", ""),
        "batch": meta.get("batch", ""),
        "source_kind": meta.get("source_kind", ""),
    }
    for sensor in SENSORS:
        x = pd.to_numeric(raw[sensor], errors="raise").to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError(f"{path} {sensor} contains NaN or infinite values.")
        baseline = float(np.median(x[bmask]))
        if abs(baseline) <= 1e-9:
            raise ValueError(f"{path} {sensor} baseline median is too close to zero.")
        z = (x - baseline) / abs(baseline)
        exp = z[emask]
        rec = z[rmask]
        rec_t = t[rmask]
        tail_n = int(np.ceil(len(exp) * 0.25))
        rec_tail_n = int(np.ceil(len(rec) * 0.25))
        row[f"{sensor}_exp_mean"] = float(np.mean(exp))
        row[f"{sensor}_exp_absmax"] = float(np.max(np.abs(exp)))
        row[f"{sensor}_exp_span"] = float(np.max(exp) - np.min(exp))
        row[f"{sensor}_exp_tail"] = float(np.mean(exp[-tail_n:]))
        row[f"{sensor}_exp_auc_abs"] = float(np.mean(np.abs(exp)))
        row[f"{sensor}_rec_mean"] = float(np.mean(rec))
        row[f"{sensor}_rec_tail"] = float(np.mean(rec[-rec_tail_n:]))
        row[f"{sensor}_rec_slope"] = slope(rec_t, rec, path, sensor)
        row[f"{sensor}_overall_std"] = float(np.std(z))
    return row


def build_feature_table_from_raw() -> pd.DataFrame:
    manifest = load_current_manifest()
    rows = []
    for meta in manifest.to_dict("records"):
        rows.append(extract_legacy9_features(Path(meta["raw_path"]), meta))
    features = pd.DataFrame(rows)
    cols = feature_columns(features)
    if len(cols) != 54:
        raise ValueError(f"Expected 54 sensor feature columns, found {len(cols)}.")
    return features


def validate_partition_mask(mask: pd.Series, split_name: str, partition: str, index: pd.Index) -> pd.Series:
    if not isinstance(mask, pd.Series):
        raise TypeError(f"{split_name} {partition} mask must be a pandas Series.")
    if not mask.index.equals(index):
        raise ValueError(f"{split_name} {partition} mask index does not match the task dataframe index.")
    if mask.dtype != bool:
        raise TypeError(f"{split_name} {partition} mask must be boolean.")
    return mask


def validate_split_parts(split_name: str, train_mask: pd.Series, val_mask: pd.Series, test_mask: pd.Series) -> None:
    overlap_train_val = train_mask & val_mask
    overlap_train_test = train_mask & test_mask
    overlap_val_test = val_mask & test_mask
    if overlap_train_val.any() or overlap_train_test.any() or overlap_val_test.any():
        raise ValueError(f"{split_name} has overlapping train/validation/test partitions.")


def require_classes(df: pd.DataFrame, classes: list[str], split_name: str, task_name: str, partition: str) -> None:
    present = set(df["label"].astype(str).unique())
    missing = [label for label in classes if label not in present]
    if missing:
        raise ValueError(f"{split_name} / {task_name} {partition} is missing classes: {missing}.")


def validate_feature_matrix(x_train: np.ndarray, matrices: dict[str, np.ndarray]) -> None:
    if x_train.ndim != 2 or x_train.shape[0] == 0 or x_train.shape[1] == 0:
        raise ValueError("Training feature matrix must be a non-empty 2D array.")
    std = np.std(x_train, axis=0)
    if not np.isfinite(x_train).all():
        raise ValueError("Training feature matrix contains NaN or infinite values.")
    if np.any(std < 1e-9):
        bad = np.where(std < 1e-9)[0].tolist()
        raise ValueError(f"Training feature matrix has near-constant columns: {bad[:20]}.")
    for name, matrix in matrices.items():
        if matrix.ndim != 2:
            raise ValueError(f"{name} feature matrix must be 2D.")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} feature matrix contains NaN or infinite values.")


def encode(labels: np.ndarray, classes: list[str]) -> np.ndarray:
    mapping = {label: i for i, label in enumerate(classes)}
    return np.array([mapping[str(label)] for label in labels], dtype=int)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on an empty evaluation set.")
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion": cm,
    }


def partition_metric_records(
    split_name: str,
    task_name: str,
    model_name: str,
    partition: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    random_state: int | None,
) -> dict:
    metrics = metric_dict(y_true, y_pred, n_classes)
    return {
        "split": split_name,
        "task": task_name,
        "model": model_name,
        "partition": partition,
        "random_state": random_state,
        "n": len(y_true),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
    }


def confusion_records(
    split_name: str,
    task_name: str,
    model_name: str,
    partition: str,
    classes: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    random_state: int | None,
) -> list[dict]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    rows = []
    for i, true_label in enumerate(classes):
        for j, pred_label in enumerate(classes):
            rows.append(
                {
                    "split": split_name,
                    "task": task_name,
                    "model": model_name,
                    "partition": partition,
                    "random_state": random_state,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "count": int(cm[i, j]),
                }
            )
    return rows


def class_metric_records(
    split_name: str,
    task_name: str,
    model_name: str,
    partition: str,
    classes: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    random_state: int | None,
) -> list[dict]:
    labels = list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    predicted = np.bincount(y_pred, minlength=len(classes))
    rows = []
    for i, label in enumerate(classes):
        rows.append(
            {
                "split": split_name,
                "task": task_name,
                "model": model_name,
                "partition": partition,
                "random_state": random_state,
                "label": label,
                "support": int(support[i]),
                "predicted_n": int(predicted[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
        )
    return rows


def no_validation_metrics(n_classes: int) -> dict:
    return {
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "macro_f1": np.nan,
        "confusion": np.zeros((n_classes, n_classes), dtype=int),
    }


def seeds_for_model(model: str) -> tuple[int | None, ...]:
    if model in STOCHASTIC_MODELS:
        if len(RANDOM_SEEDS) < 2:
            raise ValueError("At least two fixed random seeds are required to estimate training variance.")
        return RANDOM_SEEDS
    return (None,)


def make_estimator(model: str, params: dict, random_state: int | None = None):
    if model in STOCHASTIC_MODELS and random_state is None:
        raise ValueError(f"{model} requires an explicit random_state for reproducible repeated training.")
    if model not in STOCHASTIC_MODELS and random_state is not None:
        raise ValueError(f"{model} is deterministic in the configured pipeline and must not receive a random seed.")
    if model == "LDA":
        estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=params["shrinkage"])
    elif model == "Logistic Regression":
        estimator = LogisticRegression(C=params["C"], class_weight="balanced", max_iter=2000, solver="lbfgs")
    elif model == "kNN":
        estimator = KNeighborsClassifier(n_neighbors=params["n_neighbors"])
    elif model == "Random Forest":
        estimator = RandomForestClassifier(
            n_estimators=protocol.RF_N_ESTIMATORS,
            criterion=protocol.RF_CRITERION,
            max_features=params["max_features"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight="balanced",
            bootstrap=protocol.RF_BOOTSTRAP,
            random_state=random_state,
            n_jobs=protocol.RF_N_JOBS,
        )
    elif model == "RBF SVM":
        estimator = SVC(C=params["C"], gamma=params["gamma"], kernel="rbf", class_weight="balanced")
    elif model == "MLP":
        estimator = MLPClassifier(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            alpha=params["alpha"],
            learning_rate_init=params["learning_rate_init"],
            activation=protocol.MLP_ACTIVATION,
            solver=protocol.MLP_SOLVER,
            max_iter=protocol.NEURAL_EPOCHS,
            early_stopping=protocol.MLP_EARLY_STOPPING,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model: {model}")
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )


def fixed_estimator_params(model: str) -> dict:
    if model == "Random Forest":
        return {
            "n_estimators": protocol.RF_N_ESTIMATORS,
            "criterion": protocol.RF_CRITERION,
            "class_weight": "balanced",
            "bootstrap": protocol.RF_BOOTSTRAP,
            "n_jobs": protocol.RF_N_JOBS,
        }
    if model == "MLP":
        return {
            "activation": protocol.MLP_ACTIVATION,
            "solver": protocol.MLP_SOLVER,
            "max_iter": protocol.NEURAL_EPOCHS,
            "early_stopping": protocol.MLP_EARLY_STOPPING,
        }
    return {}


def serialize_params(params: dict) -> str:
    return ";".join(f"{key}={value}" for key, value in params.items())


def fit_estimator(estimator, x_train: np.ndarray, y_train: np.ndarray):
    model = clone(estimator)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        model.fit(x_train, y_train)
    return model


def select_params(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
) -> tuple[dict, tuple[float, float, float]]:
    best_params, best_score = None, None
    for params in GRIDS[model_name]:
        validation_metrics = []
        for random_state in seeds_for_model(model_name):
            estimator = fit_estimator(make_estimator(model_name, params, random_state), x_train, y_train)
            prediction = estimator.predict(x_val)
            validation_metrics.append(metric_dict(y_val, prediction, n_classes))
        score = tuple(
            float(np.mean([metrics[name] for metrics in validation_metrics]))
            for name in ["macro_f1", "balanced_accuracy", "accuracy"]
        )
        if best_score is None or score > best_score:
            best_score = score
            best_params = params
    if best_params is None or best_score is None:
        raise ValueError(f"No hyperparameter candidate was evaluated for {model_name}.")
    return best_params, best_score


def class_counts(df: pd.DataFrame, classes: list[str]) -> str:
    counts = df["label"].value_counts().reindex(classes, fill_value=0)
    return "; ".join(f"{label}:{int(counts[label])}" for label in classes)


def describe_period_days(df: pd.DataFrame) -> str:
    chunks = []
    for period in ["Period_I", "Period_II", "Period_III", "Period_IV"]:
        sub = df[df["period"].eq(period)]
        if sub.empty:
            continue
        days = sorted(int(d) for d in sub["day_num"].dropna().unique() if int(d) > 0)
        if days:
            chunks.append(f"{period} Day{min(days)}-Day{max(days)}")
        else:
            chunks.append(f"{period}")
    return "; ".join(chunks)


def describe_batches(df: pd.DataFrame) -> str:
    if "batch" not in df.columns or df.empty:
        return ""
    ordered = ["source"] + BATCH_ORDER
    present = set(df["batch"].astype(str))
    return "; ".join("Period_I" if batch == "source" else batch for batch in ordered if batch in present)


def assignment_records(split_name: str, task_name: str, partition: str, df: pd.DataFrame) -> list[dict]:
    rows = []
    for row in df.sort_values("target_file", key=lambda s: s.map(natural_key)).to_dict("records"):
        rows.append(
            {
                "split": split_name,
                "task": task_name,
                "partition": partition,
                "target_file": row["target_file"],
                "label": row["label"],
                "period": row["period"],
                "day": row["day"],
                "day_num": row["day_num"],
                "day_label": row.get("day_label", ""),
                "batch": row.get("batch", ""),
                "sample_id": row.get("sample_id", ""),
            }
        )
    return rows


def test_prediction_records(
    split_name: str,
    task_name: str,
    model: str,
    random_state: int | None,
    classes: list[str],
    test_df: pd.DataFrame,
    predicted: np.ndarray,
    test_batch: str = "",
    fit_id: str = "",
) -> list[dict]:
    if len(test_df) != len(predicted):
        raise ValueError(
            f"Prediction length mismatch for {split_name} / {model}: {len(test_df)} trials, {len(predicted)} predictions."
        )
    predicted_labels = np.asarray(classes, dtype=object)[np.asarray(predicted, dtype=int)]
    records = []
    for (_, sample), predicted_label in zip(test_df.iterrows(), predicted_labels):
        true_label = str(sample["label"])
        records.append(
            {
                "experiment": short_split(split_name),
                "split": split_name,
                "task": task_name,
                "model": model,
                "random_state": random_state,
                "test_batch": test_batch,
                "fit_id": fit_id,
                "target_file": sample["target_file"],
                "sample_id": sample.get("sample_id", ""),
                "true_label": true_label,
                "predicted_label": str(predicted_label),
                "correct": true_label == str(predicted_label),
                "period": sample.get("period", ""),
                "day_label": sample.get("day_label", ""),
                "batch": sample.get("batch", ""),
            }
        )
    return records


def summarize_repeated_values(values: list[float], stochastic: bool) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return {"mean": np.nan, "variance": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    if stochastic:
        if len(finite) != len(RANDOM_SEEDS):
            raise ValueError(
                f"Expected {len(RANDOM_SEEDS)} finite repeated values for a stochastic model, got {len(finite)}."
            )
        return {
            "mean": float(np.mean(finite)),
            "variance": float(np.var(finite, ddof=protocol.SAMPLE_STD_DDOF)),
            "std": float(np.std(finite, ddof=protocol.SAMPLE_STD_DDOF)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        }
    if len(finite) != 1:
        raise ValueError(f"A deterministic model must produce exactly one value, got {len(finite)}.")
    return {"mean": float(finite[0]), "variance": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}


def evaluate_split(df: pd.DataFrame, split: dict, task_name: str, classes: list[str]) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict], list[dict], list[dict]]:
    task_df = df[df["label"].isin(classes)].copy().reset_index(drop=True)
    train_mask = validate_partition_mask(split["train"](task_df), split["split"], "train", task_df.index)
    val_mask = validate_partition_mask(split["val"](task_df), split["split"], "validation", task_df.index)
    test_mask = validate_partition_mask(split["test"](task_df), split["split"], "test", task_df.index)
    validate_split_parts(split["split"], train_mask, val_mask, test_mask)
    source_train_mask = validate_partition_mask(
        split["source_train"](task_df), split["split"], "source_train", task_df.index
    )
    target_labeled_mask = validate_partition_mask(
        split["target_labeled_train"](task_df), split["split"], "target_labeled_train", task_df.index
    )
    train_df = pd.concat(
        [task_df[source_train_mask], task_df[target_labeled_mask]],
        axis=0,
    ).copy()
    if train_df.index.duplicated().any() or not train_df.index.to_series().isin(task_df[train_mask].index).all():
        raise ValueError(f"{split['split']} has invalid ordered training membership.")
    if set(train_df.index) != set(task_df[train_mask].index):
        raise ValueError(f"{split['split']} ordered training membership differs from the declared train mask.")
    val_df = task_df[val_mask].copy()
    test_df = task_df[test_mask].copy()
    if len(train_df) == 0:
        raise ValueError(f"{split['split']} / {task_name} has an empty training partition.")
    if len(test_df) == 0:
        raise ValueError(f"{split['split']} / {task_name} has an empty test partition.")
    require_classes(train_df, classes, split["split"], task_name, "train")
    require_classes(test_df, classes, split["split"], task_name, "test")
    if len(val_df) > 0:
        require_classes(val_df, classes, split["split"], task_name, "validation")

    cols = feature_columns(task_df)
    if not cols:
        raise ValueError("No sensor feature columns were found.")
    x_train_raw = train_df[cols].to_numpy(float)
    x_val_raw = val_df[cols].to_numpy(float)
    x_test_raw = test_df[cols].to_numpy(float)
    validate_feature_matrix(
        x_train_raw,
        {
            "validation": x_val_raw,
            "test": x_test_raw,
        },
    )
    x_train = x_train_raw
    x_val = x_val_raw
    x_test = x_test_raw
    y_train = encode(train_df["label"].to_numpy(str), classes)
    y_val = encode(val_df["label"].to_numpy(str), classes) if len(val_df) else np.array([], dtype=int)
    y_test = encode(test_df["label"].to_numpy(str), classes)

    split_rows = []
    assignment_rows = []
    for name, part_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        split_rows.append(
            {
                "split": split["split"],
                "task": task_name,
                "partition": name,
                "n": len(part_df),
                "class_counts": class_counts(part_df, classes) if len(part_df) else "",
                "period_days": describe_period_days(part_df) if len(part_df) else "",
                "batches": describe_batches(part_df) if len(part_df) else "",
            }
        )
        assignment_rows.extend(assignment_records(split["split"], task_name, name, part_df))

    rows, seed_rows, cm_rows, partition_metric_rows, class_metric_rows, prediction_rows = [], [], [], [], [], []
    for model in MODELS:
        stochastic = model in STOCHASTIC_MODELS
        model_seeds = seeds_for_model(model)
        if len(val_df) == 0:
            best_params = DEFAULT_PARAMS[model]
            selection_method = "fixed_default_no_validation"
        else:
            best_params, _best_score = select_params(model, x_train, y_train, x_val, y_val, len(classes))
            selection_method = (
                "validation_metrics_mean_over_fixed_seeds"
                if stochastic
                else "validation_macro_f1_balanced_accuracy_accuracy"
            )

        model_seed_rows = []
        for random_state in model_seeds:
            selection_estimator = fit_estimator(
                make_estimator(model, best_params, random_state), x_train, y_train
            )
            pred_val = selection_estimator.predict(x_val) if len(val_df) else np.array([], dtype=int)
            val_metrics = metric_dict(y_val, pred_val, len(classes)) if len(val_df) else no_validation_metrics(len(classes))

            if FINAL_TRAINING == "train_val_refit" and len(val_df):
                final_df = pd.concat([train_df, val_df], ignore_index=True)
                x_final_raw = final_df[cols].to_numpy(float)
                x_final = x_final_raw
                y_final = encode(final_df["label"].to_numpy(str), classes)
                require_classes(final_df, classes, split["split"], task_name, "final_train")
                validate_feature_matrix(x_final_raw, {"test": x_test_raw})
                final_estimator = fit_estimator(
                    make_estimator(model, best_params, random_state), x_final, y_final
                )
                pred_train = final_estimator.predict(x_final)
                pred_test = final_estimator.predict(x_test)
                train_eval_n = len(final_df)
                final_training_used = "train_val_refit"
            elif FINAL_TRAINING == "train_only" or (FINAL_TRAINING == "train_val_refit" and not len(val_df)):
                pred_train = selection_estimator.predict(x_train)
                pred_test = selection_estimator.predict(x_test)
                y_final = y_train
                train_eval_n = len(train_df)
                final_training_used = "train_only" if FINAL_TRAINING == "train_only" else "train_only_no_validation"
            else:
                raise ValueError(f"Unknown ENOSE_FINAL_TRAINING value: {FINAL_TRAINING}")

            train_metrics = metric_dict(y_final, pred_train, len(classes))
            test_metrics = metric_dict(y_test, pred_test, len(classes))
            seed_row = {
                "split": split["split"],
                "description": split["description"],
                "task": task_name,
                "model": model,
                "random_state": random_state,
                "tuning_random_state": None,
                "tuning_seed_values": ";".join(str(seed) for seed in model_seeds) if stochastic and len(val_df) else "",
                "protocol_version": protocol.ML_PROTOCOL_VERSION,
                "preprocessing": "Pipeline(StandardScaler, classifier)",
                "best_params": serialize_params(best_params),
                "fixed_estimator_params": serialize_params(fixed_estimator_params(model)),
                "selection_method": selection_method,
                "final_training": final_training_used,
                "train_n": train_eval_n,
                "selection_train_n": len(train_df),
                "validation_n": len(val_df),
                "test_n": len(test_df),
                "train_accuracy": train_metrics["accuracy"],
                "validation_accuracy": val_metrics["accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "validation_macro_f1": val_metrics["macro_f1"],
                "test_macro_f1": test_metrics["macro_f1"],
            }
            seed_rows.append(seed_row)
            model_seed_rows.append(seed_row)
            prediction_rows.extend(
                test_prediction_records(
                    split["split"], task_name, model, random_state, classes, test_df, pred_test
                )
            )

            evaluated_partitions = [("train", y_final, pred_train), ("test", y_test, pred_test)]
            if len(val_df):
                evaluated_partitions.insert(1, ("validation", y_val, pred_val))
            for partition, y_true_part, y_pred_part in evaluated_partitions:
                partition_metric_rows.append(
                    partition_metric_records(
                        split["split"], task_name, model, partition, y_true_part, y_pred_part, len(classes), random_state
                    )
                )
                cm_rows.extend(
                    confusion_records(
                        split["split"], task_name, model, partition, classes, y_true_part, y_pred_part, random_state
                    )
                )
                class_metric_rows.extend(
                    class_metric_records(
                        split["split"], task_name, model, partition, classes, y_true_part, y_pred_part, random_state
                    )
                )

        summary_row = {
            "split": split["split"],
            "description": split["description"],
            "task": task_name,
            "model": model,
            "training_randomness": "stochastic" if stochastic else "deterministic",
            "protocol_version": protocol.ML_PROTOCOL_VERSION,
            "preprocessing": "Pipeline(StandardScaler, classifier)",
            "seed_count": len(model_seeds) if stochastic else 0,
            "seed_values": ";".join(str(seed) for seed in model_seeds if seed is not None),
            "tuning_random_state": None,
            "tuning_seed_values": ";".join(str(seed) for seed in model_seeds) if stochastic and len(val_df) else "",
            "best_params": serialize_params(best_params),
            "fixed_estimator_params": serialize_params(fixed_estimator_params(model)),
            "selection_method": selection_method,
            "final_training": model_seed_rows[0]["final_training"],
            "train_n": model_seed_rows[0]["train_n"],
            "selection_train_n": len(train_df),
            "validation_n": len(val_df),
            "test_n": len(test_df),
        }
        for partition in ["train", "validation", "test"]:
            for metric in ["accuracy", "balanced_accuracy", "macro_f1"]:
                field = f"{partition}_{metric}"
                stats = summarize_repeated_values([row[field] for row in model_seed_rows], stochastic)
                summary_row[field] = stats["mean"]
                summary_row[f"{field}_variance"] = stats["variance"]
                summary_row[f"{field}_std"] = stats["std"]
                summary_row[f"{field}_min"] = stats["min"]
                summary_row[f"{field}_max"] = stats["max"]
        rows.append(summary_row)
    return rows, seed_rows, cm_rows, split_rows, partition_metric_rows, class_metric_rows, assignment_rows, prediction_rows


def evaluate_fixed_period_i_per_batch(
    df: pd.DataFrame,
    task_name: str,
    classes: list[str],
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict], list[dict], list[dict]]:
    task_df = df[df["label"].isin(classes)].copy().reset_index(drop=True)
    first_split = PERIOD_I_TRAINED_BATCH_TEST_SPLITS[0]
    train_mask = validate_partition_mask(first_split["train"](task_df), first_split["split"], "train", task_df.index)
    val_mask = validate_partition_mask(first_split["val"](task_df), first_split["split"], "validation", task_df.index)
    train_df = task_df[train_mask].copy()
    val_df = task_df[val_mask].copy()
    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(f"Fixed per-batch {task_name} requires non-empty train and validation partitions.")
    require_classes(train_df, classes, "PB_fixed_period_i", task_name, "train")
    require_classes(val_df, classes, "PB_fixed_period_i", task_name, "validation")

    test_parts: dict[str, tuple[dict, pd.DataFrame]] = {}
    for split in PERIOD_I_TRAINED_BATCH_TEST_SPLITS:
        split_train_mask = validate_partition_mask(split["train"](task_df), split["split"], "train", task_df.index)
        split_val_mask = validate_partition_mask(split["val"](task_df), split["split"], "validation", task_df.index)
        if not split_train_mask.equals(train_mask) or not split_val_mask.equals(val_mask):
            raise ValueError(f"{split['split']} does not reuse the fixed Period I train/validation membership.")
        test_mask = validate_partition_mask(split["test"](task_df), split["split"], "test", task_df.index)
        validate_split_parts(split["split"], train_mask, val_mask, test_mask)
        test_df = task_df[test_mask].copy()
        if len(test_df) == 0:
            raise ValueError(f"{split['split']} / {task_name} has an empty test partition.")
        require_classes(test_df, classes, split["split"], task_name, "test")
        batch = per_batch_label(split["split"])
        if batch in test_parts:
            raise ValueError(f"Duplicate fixed per-batch test definition: {batch}.")
        test_parts[batch] = (split, test_df)
    if list(test_parts) != BATCH_ORDER:
        raise ValueError(f"Fixed per-batch tests must follow {BATCH_ORDER}; got {list(test_parts)}.")

    cols = feature_columns(task_df)
    if not cols:
        raise ValueError("No sensor feature columns were found.")
    x_train_raw = train_df[cols].to_numpy(float)
    x_val_raw = val_df[cols].to_numpy(float)
    x_tests_raw = {batch: part_df[cols].to_numpy(float) for batch, (_split, part_df) in test_parts.items()}
    validate_feature_matrix(x_train_raw, {"validation": x_val_raw, **{f"test_{k}": v for k, v in x_tests_raw.items()}})
    x_train = x_train_raw
    x_val = x_val_raw
    x_tests = x_tests_raw
    y_train = encode(train_df["label"].to_numpy(str), classes)
    y_val = encode(val_df["label"].to_numpy(str), classes)
    y_tests = {
        batch: encode(part_df["label"].to_numpy(str), classes)
        for batch, (_split, part_df) in test_parts.items()
    }

    split_rows, assignment_rows = [], []
    for batch, (split, test_df) in test_parts.items():
        for partition, part_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
            split_rows.append(
                {
                    "split": split["split"],
                    "task": task_name,
                    "partition": partition,
                    "n": len(part_df),
                    "class_counts": class_counts(part_df, classes),
                    "period_days": describe_period_days(part_df),
                    "batches": describe_batches(part_df),
                    "test_batch": batch,
                    "fit_scope": "fixed_period_i_all_batches",
                }
            )
            assignment_rows.extend(assignment_records(split["split"], task_name, partition, part_df))

    rows, seed_rows, cm_rows, partition_metric_rows, class_metric_rows, prediction_rows = [], [], [], [], [], []
    for model in MODELS:
        stochastic = model in STOCHASTIC_MODELS
        model_seeds = seeds_for_model(model)
        best_params, _best_score = select_params(model, x_train, y_train, x_val, y_val, len(classes))
        selection_method = (
            "validation_metrics_mean_over_fixed_seeds"
            if stochastic
            else "validation_macro_f1_balanced_accuracy_accuracy"
        )
        batch_seed_rows: dict[str, list[dict]] = {batch: [] for batch in BATCH_ORDER}

        for random_state in model_seeds:
            selection_estimator = fit_estimator(
                make_estimator(model, best_params, random_state), x_train, y_train
            )
            pred_val = selection_estimator.predict(x_val)
            val_metrics = metric_dict(y_val, pred_val, len(classes))

            if FINAL_TRAINING == "train_val_refit":
                final_df = pd.concat([train_df, val_df], ignore_index=True)
                x_final_raw = final_df[cols].to_numpy(float)
                x_final = x_final_raw
                y_final = encode(final_df["label"].to_numpy(str), classes)
                require_classes(final_df, classes, "PB_fixed_period_i", task_name, "final_train")
                validate_feature_matrix(x_final_raw, {f"test_{k}": v for k, v in x_tests_raw.items()})
                final_estimator = fit_estimator(
                    make_estimator(model, best_params, random_state), x_final, y_final
                )
                pred_train = final_estimator.predict(x_final)
                train_eval_n = len(final_df)
                final_training_used = "train_val_refit"
            elif FINAL_TRAINING == "train_only":
                final_estimator = selection_estimator
                x_final_raw = x_train_raw
                x_final = x_train
                y_final = y_train
                pred_train = final_estimator.predict(x_final)
                train_eval_n = len(train_df)
                final_training_used = "train_only"
            else:
                raise ValueError(f"Unknown ENOSE_FINAL_TRAINING value: {FINAL_TRAINING}")

            train_metrics = metric_dict(y_final, pred_train, len(classes))
            fit_id = f"{task_name}|{model}|seed={random_state if random_state is not None else 'deterministic'}"
            for batch, (split, test_df) in test_parts.items():
                y_test = y_tests[batch]
                pred_test = final_estimator.predict(x_tests[batch])
                test_metrics = metric_dict(y_test, pred_test, len(classes))
                seed_row = {
                    "split": split["split"],
                    "description": split["description"],
                    "task": task_name,
                    "model": model,
                    "test_batch": batch,
                    "fit_scope": "fixed_period_i_all_batches",
                    "fit_id": fit_id,
                    "random_state": random_state,
                    "tuning_random_state": None,
                    "tuning_seed_values": ";".join(str(seed) for seed in model_seeds) if stochastic else "",
                    "protocol_version": protocol.ML_PROTOCOL_VERSION,
                    "preprocessing": "Pipeline(StandardScaler, classifier)",
                    "best_params": serialize_params(best_params),
                    "fixed_estimator_params": serialize_params(fixed_estimator_params(model)),
                    "selection_method": selection_method,
                    "final_training": final_training_used,
                    "train_n": train_eval_n,
                    "selection_train_n": len(train_df),
                    "validation_n": len(val_df),
                    "test_n": len(test_df),
                    "train_accuracy": train_metrics["accuracy"],
                    "validation_accuracy": val_metrics["accuracy"],
                    "test_accuracy": test_metrics["accuracy"],
                    "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                    "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
                    "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                    "train_macro_f1": train_metrics["macro_f1"],
                    "validation_macro_f1": val_metrics["macro_f1"],
                    "test_macro_f1": test_metrics["macro_f1"],
                }
                seed_rows.append(seed_row)
                batch_seed_rows[batch].append(seed_row)
                prediction_rows.extend(
                    test_prediction_records(
                        split["split"],
                        task_name,
                        model,
                        random_state,
                        classes,
                        test_df,
                        pred_test,
                        test_batch=batch,
                        fit_id=fit_id,
                    )
                )

                evaluated_partitions = [
                    ("train", y_final, pred_train),
                    ("validation", y_val, pred_val),
                    ("test", y_test, pred_test),
                ]
                for partition, y_true_part, y_pred_part in evaluated_partitions:
                    metric_row = partition_metric_records(
                        split["split"], task_name, model, partition, y_true_part, y_pred_part, len(classes), random_state
                    )
                    metric_row["fit_id"] = fit_id
                    partition_metric_rows.append(metric_row)
                    confusion = confusion_records(
                        split["split"], task_name, model, partition, classes, y_true_part, y_pred_part, random_state
                    )
                    for record in confusion:
                        record["fit_id"] = fit_id
                    cm_rows.extend(confusion)
                    class_records = class_metric_records(
                        split["split"], task_name, model, partition, classes, y_true_part, y_pred_part, random_state
                    )
                    for record in class_records:
                        record["fit_id"] = fit_id
                    class_metric_rows.extend(class_records)

        for batch, (split, test_df) in test_parts.items():
            model_seed_rows = batch_seed_rows[batch]
            summary_row = {
                "split": split["split"],
                "description": split["description"],
                "task": task_name,
                "model": model,
                "test_batch": batch,
                "fit_scope": "fixed_period_i_all_batches",
                "training_randomness": "stochastic" if stochastic else "deterministic",
                "protocol_version": protocol.ML_PROTOCOL_VERSION,
                "preprocessing": "Pipeline(StandardScaler, classifier)",
                "seed_count": len(model_seeds) if stochastic else 0,
                "seed_values": ";".join(str(seed) for seed in model_seeds if seed is not None),
                "tuning_random_state": None,
                "tuning_seed_values": ";".join(str(seed) for seed in model_seeds) if stochastic else "",
                "best_params": serialize_params(best_params),
                "fixed_estimator_params": serialize_params(fixed_estimator_params(model)),
                "selection_method": selection_method,
                "final_training": batch_seed_rows[batch][0]["final_training"],
                "train_n": model_seed_rows[0]["train_n"],
                "selection_train_n": len(train_df),
                "validation_n": len(val_df),
                "test_n": len(test_df),
            }
            for partition in ["train", "validation", "test"]:
                for metric in ["accuracy", "balanced_accuracy", "macro_f1"]:
                    field = f"{partition}_{metric}"
                    stats = summarize_repeated_values([row[field] for row in model_seed_rows], stochastic)
                    summary_row[field] = stats["mean"]
                    summary_row[f"{field}_variance"] = stats["variance"]
                    summary_row[f"{field}_std"] = stats["std"]
                    summary_row[f"{field}_min"] = stats["min"]
                    summary_row[f"{field}_max"] = stats["max"]
            rows.append(summary_row)
    return rows, seed_rows, cm_rows, split_rows, partition_metric_rows, class_metric_rows, assignment_rows, prediction_rows


def fmt(v: float) -> str:
    return "NA" if pd.isna(v) else f"{v:.3f}"


def fmt_mean_std(row: pd.Series, field: str) -> str:
    if row["training_randomness"] == "stochastic":
        return f"{fmt(row[field])} +/- {fmt(row[f'{field}_std'])}"
    return fmt(row[field])


def fmt_seed_range(row: pd.Series, field: str) -> str:
    if row["training_randomness"] == "stochastic":
        return (
            f"{fmt(row[field])} +/- {fmt(row[f'{field}_std'])} "
            f"[{fmt(row[f'{field}_min'])}, {fmt(row[f'{field}_max'])}]"
        )
    return fmt(row[field])


def pb_fit_count_label(model_name: str) -> str:
    if model_name not in GRIDS:
        raise ValueError(f"No hyperparameter grid is defined for model: {model_name}")
    seed_count = len(seeds_for_model(model_name))
    tuning_fits = len(GRIDS[model_name]) * seed_count
    final_fits = seed_count
    return f"{len(GRIDS[model_name])}*{seed_count}+{final_fits}={tuning_fits + final_fits}"


def short_split(name: str) -> str:
    return name.split("_", 1)[0]


def choose_best_by_validation(results: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for split_name, sub in results.groupby("split", sort=False):
        if int(sub["validation_n"].iloc[0]) == 0:
            continue
        ordered = sub.sort_values(
            ["validation_macro_f1", "validation_balanced_accuracy", "validation_accuracy", "model"],
            ascending=[False, False, False, True],
        )
        row = ordered.iloc[0].copy()
        row["summary_selection_note"] = "model selected only by validation metrics; test metrics used only for final evaluation"
        selected.append(row)
    if not selected:
        raise ValueError("Cannot create selected summary because no split has a validation partition.")
    return pd.DataFrame(selected).reset_index(drop=True)


def choose_fixed_validation_model_for_batches(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        raise ValueError("Cannot select a fixed per-batch model from an empty results table.")
    consistency_fields = [
        "best_params",
        "validation_accuracy",
        "validation_balanced_accuracy",
        "validation_macro_f1",
    ]
    for model, sub in results.groupby("model", sort=False):
        for field in consistency_fields:
            values = sub[field].drop_duplicates()
            if len(values) != 1:
                raise ValueError(
                    f"Fixed per-batch evaluation requires identical {field} for {model}; got {len(values)} values."
                )
    reference = results.sort_values("split").groupby("model", sort=False).first().reset_index()
    selected_model = reference.sort_values(
        ["validation_macro_f1", "validation_balanced_accuracy", "validation_accuracy", "model"],
        ascending=[False, False, False, True],
    ).iloc[0]["model"]
    selected = results[results["model"].eq(selected_model)].copy()
    selected["summary_selection_note"] = (
        "one model selected once using the fixed validation partition; the same fitted estimators evaluate all test batches"
    )
    return selected.sort_values("split").reset_index(drop=True)


def write_summary_svg(results: pd.DataFrame, path: Path) -> None:
    cols = [
        ("Split", 54),
        ("Model", 122),
        ("Best hyperparameters", 420),
        ("Tr Acc mean +/- SD [min, max]", 210),
        ("Tr BA mean +/- SD [min, max]", 210),
        ("Tr F1 mean +/- SD [min, max]", 210),
        ("Val Acc mean +/- SD [min, max]", 210),
        ("Val BA mean +/- SD [min, max]", 210),
        ("Val F1 mean +/- SD [min, max]", 210),
        ("Test Acc mean +/- SD [min, max]", 210),
        ("Test BA mean +/- SD [min, max]", 210),
        ("Test F1 mean +/- SD [min, max]", 210),
        ("Selection", 380),
    ]
    row_h = 31
    gap_h = 14
    split_names = [s["split"] for s in EXPERIMENT_SPLITS]
    rows_h = row_h * (len(results) + 1) + gap_h * (len(split_names) - 1)
    width = 32 + sum(w for _, w in cols) + 32
    height = 118 + rows_h + 176
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="32" font-family="Arial" font-size="22" font-weight="700">Machine-learning Baseline Multi-split Results for Current E-nose Data</text>',
        '<text x="24" y="56" font-family="Arial" font-size="13" fill="#555">Three-class only. S-splits use chronological days across all available later periods; B-splits use the declared batch groups.</text>',
        '<text x="24" y="74" font-family="Arial" font-size="12" fill="#555">Overfitting evidence is reported as raw train, validation, and test metrics. Class-level precision/recall/F1 and all partition confusion matrices are written to CSV.</text>',
    ]

    x0, y = 24, 92
    x = x0
    for label, w in cols:
        lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{row_h}" fill="#f1f4f8" stroke="#d8dee8"/>')
        lines.append(f'<text x="{x + 5}" y="{y + 20}" font-family="Arial" font-size="10" font-weight="700">{label}</text>')
        x += w
    y += row_h

    for split_idx, split_name in enumerate(split_names):
        sub = results[results["split"].eq(split_name)].copy()
        if sub.empty:
            raise ValueError(f"No rows available for split {split_name}.")
        sub = sub.sort_values("model", key=lambda s: s.map({name: i for i, name in enumerate(MODELS)}))
        for row_idx, (_, row) in enumerate(sub.iterrows()):
            vals = [
                short_split(row["split"]) if row_idx == 0 else "",
                row["model"],
                row["best_params"],
                fmt_seed_range(row, "train_accuracy"),
                fmt_seed_range(row, "train_balanced_accuracy"),
                fmt_seed_range(row, "train_macro_f1"),
                fmt_seed_range(row, "validation_accuracy"),
                fmt_seed_range(row, "validation_balanced_accuracy"),
                fmt_seed_range(row, "validation_macro_f1"),
                fmt_seed_range(row, "test_accuracy"),
                fmt_seed_range(row, "test_balanced_accuracy"),
                fmt_seed_range(row, "test_macro_f1"),
                row["selection_method"],
            ]
            fill = "#ffffff" if row_idx % 2 == 0 else "#fbfcfe"
            x = x0
            for (_label, w), val in zip(cols, vals):
                lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{row_h}" fill="{fill}" stroke="#d8dee8"/>')
                lines.append(f'<text x="{x + 5}" y="{y + 20}" font-family="Arial" font-size="9">{escape(str(val))}</text>')
                x += w
            y += row_h
        if split_idx != len(split_names) - 1:
            lines.append(f'<rect x="{x0}" y="{y}" width="{sum(w for _, w in cols)}" height="{gap_h}" fill="#ffffff" stroke="none"/>')
            y += gap_h

    y += 22
    legend = [f"{short_split(split['split'])}: {split['description']}" for split in EXPERIMENT_SPLITS]
    legend.append("Random Forest and MLP are reported as mean +/- sample SD across fixed seeds 0-4; deterministic models are single fits. Random seeds are not hyperparameters.")
    for line in legend:
        lines.append(f'<text x="24" y="{y}" font-family="Arial" font-size="12" fill="#555">{line}</text>')
        y += 17
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def configure_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_publication_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ordered_results(results: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    required = {
        "split",
        "model",
        "test_balanced_accuracy",
        "test_macro_f1",
        "training_randomness",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(f"Publication figures require result columns: {missing}")
    split_names = [split["split"] for split in EXPERIMENT_SPLITS]
    expected = pd.MultiIndex.from_product([split_names, MODELS], names=["split", "model"])
    indexed = results.set_index(["split", "model"])
    if indexed.index.has_duplicates:
        raise ValueError("Publication figures require exactly one aggregate row per split and model.")
    absent = expected.difference(indexed.index)
    if len(absent):
        raise ValueError(f"Publication figure data are incomplete; missing split/model rows: {list(absent)}")
    return split_names, indexed.loc[expected].reset_index()


def draw_model_split_heatmap(results: pd.DataFrame, stem: Path) -> None:
    split_names, ordered = ordered_results(results)
    split_labels = [short_split(name) for name in split_names]
    metrics = [
        ("test_balanced_accuracy", "Test balanced accuracy"),
        ("test_macro_f1", "Test macro-F1"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(11.7, 6.8), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    image = None
    for ax, (metric, title) in zip(axes, metrics):
        matrix = np.empty((len(MODELS), len(split_names)), dtype=float)
        annotations: list[list[str]] = []
        for model_idx, model in enumerate(MODELS):
            row_text = []
            for split_idx, split_name in enumerate(split_names):
                row = ordered[(ordered["split"].eq(split_name)) & (ordered["model"].eq(model))].iloc[0]
                value = float(row[metric])
                matrix[model_idx, split_idx] = value
                std_value = pd.to_numeric(pd.Series([row.get(f"{metric}_std", np.nan)]), errors="coerce").iloc[0]
                if model in STOCHASTIC_MODELS and pd.notna(std_value):
                    row_text.append(f"{value:.3f}\n±{float(std_value):.3f}")
                else:
                    row_text.append(f"{value:.3f}")
            annotations.append(row_text)
        image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks(np.arange(len(split_labels)), labels=split_labels)
        ax.set_yticks(np.arange(len(MODELS)), labels=MODELS)
        ax.set_xlabel("Split")
        ax.set_ylabel("Model")
        for model_idx in range(len(MODELS)):
            for split_idx in range(len(split_names)):
                color = "white" if matrix[model_idx, split_idx] < 0.58 else "black"
                ax.text(split_idx, model_idx, annotations[model_idx][split_idx], ha="center", va="center", fontsize=7, color=color)
        ax.set_xticks(np.arange(-0.5, len(split_names), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(MODELS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=axes, fraction=0.022, pad=0.02)
    colorbar.set_label("Metric value")
    fig.suptitle("Model performance across train/validation/test splits", fontweight="bold")
    fig.text(
        0.5,
        -0.005,
        "Random Forest and MLP: mean ± sample SD across seeds 0–9; other models: deterministic single fit.",
        ha="center",
        fontsize=8,
    )
    save_publication_figure(fig, stem)


def draw_split_model_point_ranges(results: pd.DataFrame, stem: Path) -> None:
    split_names, ordered = ordered_results(results)
    if len(split_names) != 7:
        raise ValueError(f"The grouped model figure requires exactly seven splits; got {split_names}.")
    expected = pd.MultiIndex.from_product([split_names, MODELS], names=["split", "model"])
    indexed = ordered.set_index(["split", "model"])
    if indexed.index.has_duplicates:
        raise ValueError("The grouped model figure requires exactly one aggregate row per split and model.")
    absent = expected.difference(indexed.index)
    if len(absent):
        raise ValueError(f"The grouped model figure is missing split/model rows: {list(absent)}")

    def split_color(model: str, split_index: int) -> tuple[float, float, float]:
        base = np.asarray(matplotlib.colors.to_rgb(MODEL_COLORS[model]), dtype=float)
        strength = np.linspace(0.32, 1.0, len(split_names))[split_index]
        return tuple(1.0 - (1.0 - base) * strength)

    metrics = [
        ("test_balanced_accuracy", "Test balanced accuracy (BA)"),
        ("test_macro_f1", "Test macro-F1"),
    ]
    group_centres = np.arange(len(MODELS), dtype=float)
    bar_width = 0.105
    split_offsets = (np.arange(len(split_names)) - (len(split_names) - 1) / 2.0) * bar_width
    fig, axes = plt.subplots(2, 1, figsize=(16.5, 10.2), sharex=True, sharey=True, constrained_layout=True)

    for ax, (metric, title) in zip(axes, metrics):
        for model_index, model in enumerate(MODELS):
            for split_index, split_name in enumerate(split_names):
                row = indexed.loc[(split_name, model)]
                value = float(row[metric])
                if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"Invalid {metric} for {split_name}/{model}: {value}")
                x = group_centres[model_index] + split_offsets[split_index]
                color = split_color(model, split_index)
                ax.bar(
                    x,
                    value - 0.5,
                    bottom=0.5,
                    width=bar_width * 0.88,
                    color=color,
                    edgecolor=MODEL_COLORS[model],
                    linewidth=0.55,
                    zorder=3,
                )
                if value >= 0.965:
                    label_y, vertical_alignment = value - 0.006, "top"
                else:
                    label_y, vertical_alignment = value + 0.006, "bottom"
                ax.text(
                    x,
                    label_y,
                    f"{value:.3f}",
                    ha="center",
                    va=vertical_alignment,
                    rotation=90,
                    fontsize=6.3,
                    color="#172033",
                    clip_on=True,
                    zorder=4,
                )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylim(0.5, 1.0)
        ax.set_yticks(np.arange(0.5, 1.01, 0.1))
        ax.set_ylabel("Test metric value")
        ax.grid(axis="y", color="#d9dee7", linewidth=0.65)
        ax.set_axisbelow(True)

    axes[-1].set_xticks(group_centres, labels=MODELS)
    axes[-1].set_xlabel("Model (S1–S7 from light to dark within each model)")
    axes[-1].tick_params(axis="x", labelrotation=0)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=split_color("LDA", split_index),
            markeredgecolor=MODEL_COLORS["LDA"],
            label=short_split(split_name),
            markersize=8,
        )
        for split_index, split_name in enumerate(split_names)
    ]
    axes[0].legend(
        handles=legend_handles,
        title="Split order within every model",
        ncol=7,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )
    fig.suptitle("Seven chronological splits within each machine-learning model", fontweight="bold", y=1.02)
    fig.text(
        0.5,
        -0.015,
        "Random Forest and MLP values are means across fixed seeds 0–4; deterministic models are single fits.",
        ha="center",
        fontsize=8,
        color="#4F5D73",
    )
    save_publication_figure(fig, stem)


def draw_period_i_batch_trends(per_batch_results: pd.DataFrame, stem: Path) -> None:
    required = {
        "test_batch",
        "model",
        "test_n",
        "test_balanced_accuracy",
        "test_macro_f1",
    }
    missing = sorted(required.difference(per_batch_results.columns))
    if missing:
        raise ValueError(f"Per-batch figure requires result columns: {missing}")
    expected = pd.MultiIndex.from_product([BATCH_ORDER, MODELS], names=["test_batch", "model"])
    indexed = per_batch_results.set_index(["test_batch", "model"])
    if indexed.index.has_duplicates:
        raise ValueError("Per-batch figure requires exactly one aggregate row per batch and model.")
    absent = expected.difference(indexed.index)
    if len(absent):
        raise ValueError(f"Per-batch figure data are incomplete; missing batch/model rows: {list(absent)}")
    ordered = indexed.loc[expected]
    test_sizes = []
    for batch in BATCH_ORDER:
        batch_sizes = ordered.loc[batch, "test_n"].astype(int).unique()
        if len(batch_sizes) != 1:
            raise ValueError(f"Models disagree on test n for {batch}: {batch_sizes.tolist()}")
        test_sizes.append(int(batch_sizes[0]))

    metrics = [
        ("test_balanced_accuracy", "Test balanced accuracy"),
        ("test_macro_f1", "Test macro-F1"),
    ]
    x = np.arange(len(BATCH_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 5.2), sharey=True, constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics):
        for model in MODELS:
            subset = ordered.xs(model, level="model").loc[BATCH_ORDER]
            values = subset[metric].astype(float).to_numpy()
            line_width = 2.6 if model == "Logistic Regression" else 1.5
            zorder = 4 if model == "Logistic Regression" else 2
            ax.plot(x, values, marker="o", markersize=4.5, linewidth=line_width, color=MODEL_COLORS[model], label=model, zorder=zorder)
            if model in STOCHASTIC_MODELS:
                low = subset[f"{metric}_min"].astype(float).to_numpy()
                high = subset[f"{metric}_max"].astype(float).to_numpy()
                ax.fill_between(x, low, high, color=MODEL_COLORS[model], alpha=0.15, linewidth=0, zorder=1)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlim(-0.15, len(BATCH_ORDER) - 0.85)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x, labels=[f"{batch}\n(n={size})" for batch, size in zip(BATCH_ORDER, test_sizes)])
        ax.set_xlabel("Test batch")
        ax.grid(color="#d9dee7", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Metric value")
    axes[1].legend(loc="lower right", frameon=True, ncol=2)
    fig.suptitle("Fixed Period I train/validation partitions: performance by test batch", fontweight="bold")
    fig.text(
        0.5,
        -0.015,
        "The same fitted estimator is reused across batch1–batch5. Shading shows seed min–max for Random Forest and MLP.",
        ha="center",
        fontsize=8,
    )
    save_publication_figure(fig, stem)


def draw_period_i_per_batch_table(per_batch_results: pd.DataFrame, stem: Path) -> None:
    required = {
        "split",
        "test_batch",
        "model",
        "training_randomness",
        "train_n",
        "validation_n",
        "test_n",
        "validation_accuracy",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
    }
    missing = sorted(required.difference(per_batch_results.columns))
    if missing:
        raise ValueError(f"Per-batch table requires result columns: {missing}")
    expected = pd.MultiIndex.from_product([BATCH_ORDER, MODELS], names=["test_batch", "model"])
    indexed = per_batch_results.set_index(["test_batch", "model"])
    if indexed.index.has_duplicates:
        raise ValueError("Per-batch table requires exactly one aggregate row per batch and model.")
    absent = expected.difference(indexed.index)
    if len(absent):
        raise ValueError(f"Per-batch table data are incomplete; missing batch/model rows: {list(absent)}")
    ordered = indexed.loc[expected].reset_index()

    headers = [
        "PB",
        "Model",
        "n (train/val/test)",
        "Fit calls (tune+final)",
        "Validation Acc",
        "Test Acc",
        "Test BA",
        "Test macro-F1",
    ]
    rows: list[list[str]] = []
    for _, row in ordered.iterrows():
        rows.append(
            [
                short_split(str(row["split"])),
                str(row["model"]),
                f"{int(row['train_n'])}/{int(row['validation_n'])}/{int(row['test_n'])}",
                pb_fit_count_label(str(row["model"])),
                fmt_seed_range(row, "validation_accuracy"),
                fmt_seed_range(row, "test_accuracy"),
                fmt_seed_range(row, "test_balanced_accuracy"),
                fmt_seed_range(row, "test_macro_f1"),
            ]
        )

    fig, ax = plt.subplots(figsize=(16.5, 11.7))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="left",
        colLoc="left",
        colWidths=[0.05, 0.12, 0.095, 0.105, 0.145, 0.145, 0.145, 0.145],
        bbox=[0.01, 0.055, 0.98, 0.875],
        edges="closed",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    for col_idx in range(len(headers)):
        cell = table[(0, col_idx)]
        cell.set_facecolor("#17365D")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#9AA9BC")
        cell.set_linewidth(0.7)
    for row_idx, row in enumerate(rows, start=1):
        batch_idx = int(row[0][2:]) - 1
        fill = "#F3F6FA" if batch_idx % 2 == 0 else "#FFFFFF"
        for col_idx in range(len(headers)):
            cell = table[(row_idx, col_idx)]
            cell.set_facecolor(fill)
            cell.set_edgecolor("#C8D0DC")
            cell.set_linewidth(0.45)
            if col_idx in {0, 1}:
                cell.set_text_props(fontweight="bold")
    fig.text(0.015, 0.972, "Fixed Period I train/validation: PB1-PB5 test results", fontsize=14, fontweight="bold")
    fig.text(0.015, 0.948, "All six models are reported; test metrics are not used for model selection.", fontsize=9, color="#4F5D73")
    fig.text(
        0.015,
        0.018,
        "Fit calls count hyperparameter-candidate fits + final estimator fits for the entire PB1-PB5 experiment; final fits are reused across all five test batches.",
        fontsize=8,
        color="#4F5D73",
    )
    fig.text(
        0.015,
        0.006,
        "Random Forest and MLP: mean +/- sample SD [min, max] across seeds 0-4. Other models: deterministic single fit.",
        fontsize=8,
        color="#4F5D73",
    )
    save_publication_figure(fig, stem)


def write_publication_figures(results: pd.DataFrame, per_batch_results: pd.DataFrame) -> None:
    configure_publication_style()
    draw_model_split_heatmap(results, FIG_DIR / PUBLICATION_FIGURES["heatmap"])
    draw_split_model_point_ranges(results, FIG_DIR / PUBLICATION_FIGURES["point_ranges"])
    draw_period_i_batch_trends(per_batch_results, FIG_DIR / PUBLICATION_FIGURES["batch_trends"])
    draw_period_i_per_batch_table(per_batch_results, FIG_DIR / PUBLICATION_FIGURES["pb_table"])


def per_batch_label(split_name: str) -> str:
    for batch in BATCH_ORDER:
        if split_name.endswith(f"_{batch}"):
            return batch
    return split_name


def build_interactive_explorer_html(path: Path) -> str:
    if not path.exists():
        return ""
    predictions = pd.read_csv(path)
    required = {
        "experiment", "model", "random_state", "target_file", "true_label",
        "predicted_label", "correct", "day_label", "batch",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Interactive predictions are missing columns: {missing}")
    if predictions["correct"].dtype != bool:
        mapped = predictions["correct"].astype(str).str.lower().map({"true": True, "false": False})
        if mapped.isna().any():
            raise ValueError("Interactive prediction correctness values are not Boolean.")
        predictions["correct"] = mapped
    predictions["seed_label"] = predictions["random_state"].apply(
        lambda value: "deterministic" if pd.isna(value) else str(int(float(value)))
    )
    entries = []
    for (experiment, model, seed_label), group in predictions.groupby(
        ["experiment", "model", "seed_label"], sort=False
    ):
        matrix = confusion_matrix(
            pd.Categorical(group["true_label"], categories=THREE_CLASSES).codes,
            pd.Categorical(group["predicted_label"], categories=THREE_CLASSES).codes,
            labels=list(range(len(THREE_CLASSES))),
        ).tolist()
        errors = group.loc[
            ~group["correct"],
            ["target_file", "true_label", "predicted_label", "day_label", "batch"],
        ].to_dict("records")
        entries.append(
            {
                "experiment": experiment,
                "model": model,
                "seed": seed_label,
                "matrix": matrix,
                "errors": errors,
            }
        )
    payload = json.dumps(
        {"classes": THREE_CLASSES, "models": MODELS, "entries": entries},
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    template = r'''
<div class="ml-interactive" id="ml-interactive-explorer">
  <h3>Interactive confusion matrix and error explorer</h3>
  <p class="ml-results-note">Select S1-S7 or PB1-PB5, click a model, then inspect a seed. For Random Forest and MLP, the seed-average view reports the mean confusion matrix and each file's error frequency across all fixed seeds.</p>
  <div class="ml-explorer-controls">
    <label>Split / batch <select id="ml-exp-select"></select></label>
    <label>Seed <select id="ml-seed-select"></select></label>
  </div>
  <div class="ml-model-buttons" id="ml-model-buttons"></div>
  <div class="ml-explorer-summary" id="ml-explorer-summary"></div>
  <div class="ml-explorer-grid">
    <div><h4>Confusion matrix</h4><div id="ml-confusion"></div><p class="ml-results-note">Rows: true class. Columns: predicted class.</p></div>
    <div><h4>Misclassified trials</h4><div class="ml-error-table-wrap"><table class="ml-results-table" id="ml-error-table"><thead></thead><tbody></tbody></table></div></div>
  </div>
</div>
<script type="application/json" id="ml-interactive-data">__PAYLOAD__</script>
<script>
(() => {
  const payload=JSON.parse(document.getElementById('ml-interactive-data').textContent);
  const expSelect=document.getElementById('ml-exp-select'), seedSelect=document.getElementById('ml-seed-select');
  const buttons=document.getElementById('ml-model-buttons'), summary=document.getElementById('ml-explorer-summary');
  const confusion=document.getElementById('ml-confusion'), errorTable=document.getElementById('ml-error-table');
  let activeModel=payload.models[0];
  const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  [...new Set(payload.entries.map(item=>item.experiment))].forEach(value=>expSelect.add(new Option(value,value)));
  payload.models.forEach(model=>{const button=document.createElement('button');button.type='button';button.dataset.model=model;button.onclick=()=>{activeModel=model;updateButtons();updateSeeds();render();};buttons.appendChild(button);});
  const query=new URLSearchParams(window.location.search),requestedExperiment=query.get('experiment'),requestedModel=query.get('model');
  if(requestedExperiment&&[...expSelect.options].some(option=>option.value===requestedExperiment))expSelect.value=requestedExperiment;
  if(requestedModel&&payload.models.includes(requestedModel))activeModel=requestedModel;
  const entriesFor=model=>payload.entries.filter(item=>item.experiment===expSelect.value&&item.model===model);
  const currentEntries=()=>entriesFor(activeModel);
  function updateButtons(){[...buttons.children].forEach(button=>{const entries=entriesFor(button.dataset.model),matrix=entries.length>1?average(entries).matrix:entries[0].matrix,metric=calculate(matrix);button.classList.toggle('active',button.dataset.model===activeModel);button.innerHTML=`<span>${esc(button.dataset.model)}</span><strong>Accuracy ${metric.accuracy.toFixed(3)}</strong><small>BA ${metric.ba.toFixed(3)} | F1 ${metric.f1.toFixed(3)}</small>`;});}
  function updateSeeds(){const entries=currentEntries();seedSelect.innerHTML='';if(entries.length>1)seedSelect.add(new Option('seed average','__average__'));entries.forEach(item=>seedSelect.add(new Option(item.seed==='deterministic'?'deterministic':`seed ${item.seed}`,item.seed)));}
  function calculate(matrix){const n=matrix.flat().reduce((a,b)=>a+b,0), diagonal=matrix.reduce((sum,row,i)=>sum+row[i],0);const recalls=matrix.map((row,i)=>row.reduce((a,b)=>a+b,0)?row[i]/row.reduce((a,b)=>a+b,0):0);const f1s=matrix.map((row,i)=>{const tp=row[i],actual=row.reduce((a,b)=>a+b,0),predicted=matrix.reduce((sum,r)=>sum+r[i],0),precision=predicted?tp/predicted:0,recall=actual?tp/actual:0;return precision+recall?2*precision*recall/(precision+recall):0;});return{n,accuracy:diagonal/n,ba:recalls.reduce((a,b)=>a+b,0)/recalls.length,f1:f1s.reduce((a,b)=>a+b,0)/f1s.length};}
  function average(entries){const matrix=payload.classes.map((_,i)=>payload.classes.map((_,j)=>entries.reduce((sum,item)=>sum+item.matrix[i][j],0)/entries.length));const map=new Map();entries.forEach(item=>item.errors.forEach(error=>{const current=map.get(error.target_file)||{...error,wrong_count:0,predictions:{}};current.wrong_count++;current.predictions[error.predicted_label]=(current.predictions[error.predicted_label]||0)+1;map.set(error.target_file,current);}));return{matrix,errors:[...map.values()].sort((a,b)=>b.wrong_count-a.wrong_count||a.target_file.localeCompare(b.target_file)),seedCount:entries.length,aggregate:true};}
  function selected(){const entries=currentEntries();if(seedSelect.value==='__average__')return average(entries);const item=entries.find(entry=>entry.seed===seedSelect.value);return{matrix:item.matrix,errors:item.errors,seedCount:1,aggregate:false};}
  function render(){const data=selected(),metric=calculate(data.matrix),max=Math.max(...data.matrix.flat(),1);summary.innerHTML=`<strong>${esc(expSelect.value)} - ${esc(activeModel)}</strong> &nbsp; n=${Math.round(metric.n)} &nbsp; Accuracy=${metric.accuracy.toFixed(3)} &nbsp; BA=${metric.ba.toFixed(3)} &nbsp; macro-F1=${metric.f1.toFixed(3)} &nbsp; Errors=${data.aggregate?data.errors.length+' files':data.errors.length}`;let html='<table class="ml-confusion-table"><thead><tr><th>True \\ Pred.</th>'+payload.classes.map(c=>`<th>${esc(c)}</th>`).join('')+'</tr></thead><tbody>';data.matrix.forEach((row,i)=>{html+=`<tr><th>${esc(payload.classes[i])}</th>`+row.map(value=>`<td style="background:rgba(23,54,93,${0.08+0.72*value/max})">${data.aggregate?value.toFixed(1):value}</td>`).join('')+'</tr>';});confusion.innerHTML=html+'</tbody></table>';const headers=data.aggregate?['File','True','Predicted across wrong seeds','Wrong seeds','Day','Batch']:['File','True','Predicted','Day','Batch'];errorTable.querySelector('thead').innerHTML='<tr>'+headers.map(h=>`<th>${h}</th>`).join('')+'</tr>';errorTable.querySelector('tbody').innerHTML=data.errors.map(error=>{if(data.aggregate){const predictions=Object.entries(error.predictions).map(([label,count])=>`${label}: ${count}`).join('; ');return`<tr><td>${esc(error.target_file)}</td><td>${esc(error.true_label)}</td><td>${esc(predictions)}</td><td>${error.wrong_count}/${data.seedCount}</td><td>${esc(error.day_label)}</td><td>${esc(error.batch)}</td></tr>`;}return`<tr><td>${esc(error.target_file)}</td><td>${esc(error.true_label)}</td><td>${esc(error.predicted_label)}</td><td>${esc(error.day_label)}</td><td>${esc(error.batch)}</td></tr>`;}).join('')||`<tr><td colspan="${headers.length}">No misclassified trials.</td></tr>`;}
  expSelect.onchange=()=>{updateButtons();updateSeeds();render();};seedSelect.onchange=render;updateButtons();updateSeeds();render();
})();
</script>
'''
    return template.replace("__PAYLOAD__", payload)


def write_standalone_interactive_report(path: Path, predictions_path: Path) -> None:
    explorer = build_interactive_explorer_html(predictions_path)
    if not explorer:
        raise FileNotFoundError(f"Interactive prediction data do not exist: {predictions_path}")
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ML Results - Confusion Matrix and Error Explorer</title>
<style>
  :root {{ color-scheme: light; --navy:#17365d; --line:#d8dee8; --muted:#5a667a; --panel:#fbfcfe; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:#f3f5f8; color:#172033; font-family:Arial,Helvetica,sans-serif; }}
  main {{ width:min(1500px,calc(100% - 32px)); margin:24px auto; background:white; padding:26px; border-radius:10px; box-shadow:0 4px 18px rgba(23,54,93,.08); }}
  h1,h2,h3,h4 {{ color:var(--navy); }}
  h1 {{ margin:0 0 8px; }}
  .scope {{ display:flex; flex-wrap:wrap; gap:8px; margin:14px 0 22px; }}
  .scope span {{ padding:6px 9px; background:#eef3f8; border:1px solid var(--line); border-radius:5px; font-size:13px; }}
  .ml-results-note {{ color:var(--muted); font-size:13px; margin-top:8px; }}
  .ml-results-table-wrap,.ml-error-table-wrap {{ width:100%; overflow:auto; }}
  .ml-results-table {{ width:max-content; min-width:100%; border-collapse:collapse; font-size:12px; }}
  .ml-results-table th,.ml-results-table td {{ border:1px solid var(--line); padding:7px 10px; text-align:left; white-space:nowrap; }}
  .ml-results-table th {{ background:#eef3f8; color:var(--navy); }}
  .ml-interactive {{ margin:0; padding:18px; border:1px solid var(--line); border-radius:8px; background:var(--panel); }}
  .ml-explorer-controls {{ display:flex; flex-wrap:wrap; gap:14px; margin:12px 0; }}
  .ml-explorer-controls label {{ font-weight:700; color:var(--navy); }}
  .ml-explorer-controls select {{ margin-left:6px; padding:6px 9px; }}
  .ml-model-buttons {{ display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 16px; }}
  .ml-model-buttons button {{ min-width:178px; display:grid; gap:3px; padding:11px 13px; text-align:left; border:1px solid #9aa9bc; border-radius:6px; background:white; color:var(--navy); cursor:pointer; }}
  .ml-model-buttons button span {{ font-weight:700; }}
  .ml-model-buttons button small {{ opacity:.82; }}
  .ml-model-buttons button.active {{ background:var(--navy); color:white; }}
  .ml-explorer-summary {{ padding:10px 12px; background:#eef3f8; border-radius:5px; margin-bottom:14px; }}
  .ml-explorer-grid {{ display:grid; grid-template-columns:minmax(300px,.75fr) minmax(560px,1.7fr); gap:20px; align-items:start; }}
  .ml-confusion-table {{ border-collapse:collapse; width:100%; text-align:center; }}
  .ml-confusion-table th,.ml-confusion-table td {{ border:1px solid #c8d0dc; padding:12px; }}
  .ml-confusion-table th {{ background:#eef3f8; color:var(--navy); }}
  .ml-error-table-wrap {{ max-height:520px; }}
  @media(max-width:980px) {{ .ml-explorer-grid {{ grid-template-columns:1fr; }} main {{ width:100%; margin:0; border-radius:0; }} }}
</style>
</head>
<body><main>
  <h1>Machine-learning results explorer</h1>
  <p class="ml-results-note">Data-only report. Test metrics are displayed after validation-only hyperparameter selection.</p>
  <div class="scope">
    <span>Classes: air / alcohol / acetone</span><span>Input: 54 trial-level features</span>
    <span>Splits: S1-S7</span><span>Fixed train/validation batches: PB1-PB5</span><span>Stochastic seeds: 0-4</span>
  </div>
  {explorer}
</main></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def write_results_html(selected: pd.DataFrame, splits: pd.DataFrame, per_batch_results: pd.DataFrame, per_batch_selected: pd.DataFrame, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Existing HTML report required for update does not exist: {path}")

    def table(headers: list[str], rows: list[list[object]]) -> str:
        head = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
        body = []
        for row in rows:
            body.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>")
        return "<div class=\"ml-results-table-wrap\"><table class=\"ml-results-table\"><thead><tr>" + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"

    selected_rows = []
    for _, row in selected.iterrows():
        selected_rows.append(
            [
                short_split(row["split"]),
                row["model"],
                row["train_n"],
                row["validation_n"],
                row["test_n"],
                fmt_seed_range(row, "train_accuracy"),
                fmt_seed_range(row, "validation_accuracy"),
                fmt_seed_range(row, "test_accuracy"),
                fmt_seed_range(row, "test_balanced_accuracy"),
                fmt_seed_range(row, "test_macro_f1"),
            ]
        )

    split_rows = []
    for _, row in splits.iterrows():
        split_rows.append(
            [
                short_split(row["split"]),
                row["partition"],
                row["n"],
                row["class_counts"],
                row.get("batches", ""),
                row["period_days"],
            ]
        )

    per_batch_selected_rows = []
    for _, row in per_batch_selected.iterrows():
        per_batch_selected_rows.append(
            [
                per_batch_label(row["split"]),
                row["model"],
                row["train_n"],
                row["validation_n"],
                row["test_n"],
                pb_fit_count_label(str(row["model"])),
                fmt_seed_range(row, "validation_accuracy"),
                fmt_seed_range(row, "test_accuracy"),
                fmt_seed_range(row, "test_balanced_accuracy"),
                fmt_seed_range(row, "test_macro_f1"),
            ]
        )

    per_batch_all_rows = []
    for _, row in per_batch_results.sort_values(["split", "model"]).iterrows():
        per_batch_all_rows.append(
            [
                per_batch_label(row["split"]),
                row["model"],
                row["train_n"],
                row["validation_n"],
                row["test_n"],
                pb_fit_count_label(str(row["model"])),
                fmt_seed_range(row, "validation_accuracy"),
                fmt_seed_range(row, "test_accuracy"),
                fmt_seed_range(row, "test_balanced_accuracy"),
                fmt_seed_range(row, "test_macro_f1"),
                row["best_params"],
            ]
        )

    style = """
<style id="ml-results-style">
  .ml-results-table-wrap { width: 100%; overflow-x: auto; margin-top: 14px; }
  .ml-results-table { width: max-content; min-width: 100%; border-collapse: collapse; font-size: 12px; table-layout: auto; }
  .ml-results-table th, .ml-results-table td { border: 1px solid #d8dee8; padding: 7px 10px; text-align: left; vertical-align: top; white-space: nowrap; }
  .ml-results-table th { background: #eef3f8; color: #102a53; font-weight: 700; }
  .ml-results-note { color: #5a667a; font-size: 13px; margin-top: 8px; }
  .ml-publication-figure { margin: 22px 0 30px; }
  .ml-publication-figure img { display: block; width: 100%; height: auto; border: 1px solid #e2e7ef; background: white; }
  .ml-publication-figure figcaption { color: #5a667a; font-size: 12px; margin-top: 7px; }
  .ml-interactive { margin: 28px 0; padding: 18px; border: 1px solid #d8dee8; border-radius: 8px; background: #fbfcfe; }
  .ml-explorer-controls { display: flex; flex-wrap: wrap; gap: 14px; margin: 12px 0; }
  .ml-explorer-controls label { font-weight: 700; color: #17365d; }
  .ml-explorer-controls select { margin-left: 6px; padding: 5px 8px; }
  .ml-model-buttons { display: flex; flex-wrap: wrap; gap: 7px; margin: 10px 0 16px; }
  .ml-model-buttons button { min-width: 168px; display: grid; gap: 3px; padding: 10px 12px; text-align: left; border: 1px solid #9aa9bc; border-radius: 6px; background: white; color: #17365d; cursor: pointer; }
  .ml-model-buttons button span { font-weight: 700; }
  .ml-model-buttons button small { opacity: 0.82; }
  .ml-model-buttons button.active { background: #17365d; color: white; }
  .ml-explorer-summary { padding: 9px 11px; background: #eef3f8; border-radius: 5px; margin-bottom: 14px; }
  .ml-explorer-grid { display: grid; grid-template-columns: minmax(300px, 0.8fr) minmax(520px, 1.7fr); gap: 20px; align-items: start; }
  .ml-confusion-table { border-collapse: collapse; width: 100%; text-align: center; }
  .ml-confusion-table th, .ml-confusion-table td { border: 1px solid #c8d0dc; padding: 10px; }
  .ml-confusion-table th { background: #eef3f8; color: #17365d; }
  .ml-error-table-wrap { max-height: 430px; overflow: auto; }
  @media (max-width: 980px) { .ml-explorer-grid { grid-template-columns: 1fr; } }
</style>
"""
    figure_block = "".join(
        [
            "<h3>Publication-ready comparison figures</h3>\n",
            "<p class=\"ml-results-note\">The embedded SVG files are vector graphics. Matching vector PDF and 600 dpi PNG files are stored in the same figure directory.</p>\n",
            f"<figure class=\"ml-publication-figure\"><img src=\"../figures/ml_learning/{PUBLICATION_FIGURES['heatmap']}.svg\" alt=\"Model by split heatmap\"><figcaption>Test balanced accuracy and macro-F1 across all models and splits.</figcaption></figure>\n",
            f"<figure class=\"ml-publication-figure\"><img src=\"../figures/ml_learning/{PUBLICATION_FIGURES['point_ranges']}.svg\" alt=\"Within-split model point ranges\"><figcaption>Within-split model comparison; Random Forest and MLP ranges show seed min–max.</figcaption></figure>\n",
            f"<figure class=\"ml-publication-figure\"><img src=\"../figures/ml_learning/{PUBLICATION_FIGURES['batch_trends']}.svg\" alt=\"Fixed Period I per-batch trends\"><figcaption>Fixed train/validation partitions with the same fitted estimators tested on batch1–batch5.</figcaption></figure>\n",
            f"<figure class=\"ml-publication-figure\"><img src=\"../figures/ml_learning/{PUBLICATION_FIGURES['pb_table']}.svg\" alt=\"PB1-PB5 all-model results table\"><figcaption>PB1-PB5 results for all six models, including seed variability for Random Forest and MLP.</figcaption></figure>\n",
        ]
    )
    block = (
        "<!-- ML_BASELINE_RESULTS_START -->\n"
        + style
        + "<div class=\"section\" id=\"ml-baseline-results\">\n"
        + "<h2>ML baseline results on current data</h2>\n"
        + "<p class=\"ml-results-note\">Random Forest and MLP metrics are mean +/- sample SD [min, max] across fixed seeds 0-4 after hyperparameters are selected by validation metrics averaged over those seeds. Deterministic models are single fits. Random seeds are experimental controls, not hyperparameters. Splits without validation use fixed defaults and are omitted from the selected-model table.</p>\n"
        + build_interactive_explorer_html(INTERACTIVE_PREDICTIONS_PATH)
        + figure_block
        + table(
            ["Split", "Validation-selected model", "Train n", "Val n", "Test n", "Train Acc", "Val Acc", "Test Acc", "Test BA", "Test macro-F1"],
            selected_rows,
        )
        + "<h3>Partition audit</h3>\n"
        + table(["Split", "Partition", "n", "Class counts", "Batches", "Period days"], split_rows)
        + "<h3>Fixed Period I training: per-batch test</h3>\n"
        + "<p class=\"ml-results-note\">Training and validation are fixed once. Each deterministic model is fitted once, and each Random Forest/MLP seed is fitted once; the same fitted estimators then predict batch1-batch5 separately. Only the test batch changes.</p>\n"
        + table(
            ["Test batch", "Validation-selected model", "Train n", "Val n", "Test n", "Fit calls (tune+final)", "Val Acc", "Test Acc", "Test BA", "Test macro-F1"],
            per_batch_selected_rows,
        )
        + "<h3>Fixed Period I training: per-batch all models</h3>\n"
        + "<p class=\"ml-results-note\">Fit calls count hyperparameter-candidate fits plus final estimator fits for the complete PB1-PB5 experiment. Final fitted estimators are reused across all five test batches.</p>\n"
        + table(
            ["Test batch", "Model", "Train n", "Val n", "Test n", "Fit calls (tune+final)", "Val Acc", "Test Acc", "Test BA", "Test macro-F1", "Best hyperparameters"],
            per_batch_all_rows,
        )
        + "</div>\n"
        + "<!-- ML_BASELINE_RESULTS_END -->"
    )
    html = path.read_text(encoding="utf-8")
    pattern = re.compile(r"<!-- ML_BASELINE_RESULTS_START -->.*?<!-- ML_BASELINE_RESULTS_END -->", re.S)
    if pattern.search(html):
        html = pattern.sub(lambda _match: block, html)
    else:
        if "</body>" not in html:
            raise ValueError(f"Existing HTML report does not contain a closing body tag: {path}")
        html = html.replace("</body>", block + "\n</body>")
    path.write_text(html, encoding="utf-8")


def main() -> None:
    if RUN_SCOPE not in {"core", "per_batch", "all", "figures", "interactive"}:
        raise ValueError("ENOSE_RUN_SCOPE must be 'core', 'per_batch', 'all', 'figures', or 'interactive'.")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if RUN_SCOPE == "interactive":
        write_standalone_interactive_report(INTERACTIVE_REPORT_PATH, INTERACTIVE_PREDICTIONS_PATH)
        print(f"Standalone interactive ML report: {INTERACTIVE_REPORT_PATH}")
        return
    if RUN_SCOPE == "figures":
        required = {
            "results": TABLE_DIR / "manifest_da_baseline_split_results.csv",
            "selected": TABLE_DIR / "manifest_da_baseline_selected_summary.csv",
            "splits": TABLE_DIR / "manifest_da_baseline_split_splits.csv",
            "per_batch_results": TABLE_DIR / "manifest_da_baseline_period_i_per_batch_results.csv",
            "per_batch_selected": TABLE_DIR / "manifest_da_baseline_period_i_per_batch_selected_summary.csv",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Figures-only mode requires existing result tables; missing: {missing}")
        results = pd.read_csv(required["results"])
        selected = pd.read_csv(required["selected"])
        splits = pd.read_csv(required["splits"])
        per_batch_results = pd.read_csv(required["per_batch_results"])
        per_batch_selected = pd.read_csv(required["per_batch_selected"])
        write_summary_svg(results, FIG_DIR / "manifest_da_baseline_split_summary.svg")
        write_publication_figures(results, per_batch_results)
        write_results_html(selected, splits, per_batch_results, per_batch_selected, HTML_REPORT_PATH)
        write_standalone_interactive_report(INTERACTIVE_REPORT_PATH, INTERACTIVE_PREDICTIONS_PATH)
        print("Publication figures and existing HTML report updated from existing result tables.")
        return
    if REBUILD_FEATURES_FROM_RAW:
        features = build_feature_table_from_raw()
        features.to_csv(FEATURE_PATH, index=False)
        print(f"Rebuilt feature table from raw data: {FEATURE_PATH}")
        print(f"Current manifest: {CURRENT_MANIFEST_PATH}")
    else:
        if not FEATURE_PATH.exists():
            raise FileNotFoundError(f"Required feature table not found: {FEATURE_PATH}")
        features = pd.read_csv(FEATURE_PATH)
        print(f"Feature table: {FEATURE_PATH}")

    if RUN_SCOPE in {"core", "all"}:
        all_rows, all_seed_rows, all_cm, all_splits = [], [], [], []
        all_partition_metrics, all_class_metrics, all_assignments, all_predictions = [], [], [], []
        for split in EXPERIMENT_SPLITS:
            for task_name, classes in TASKS:
                rows, seed_rows, cm_rows, split_rows, partition_metric_rows, class_metric_rows, assignment_rows, prediction_rows = evaluate_split(features, split, task_name, classes)
                all_rows.extend(rows)
                all_seed_rows.extend(seed_rows)
                all_cm.extend(cm_rows)
                all_splits.extend(split_rows)
                all_partition_metrics.extend(partition_metric_rows)
                all_class_metrics.extend(class_metric_rows)
                all_assignments.extend(assignment_rows)
                all_predictions.extend(prediction_rows)

        results = pd.DataFrame(all_rows)
        seed_results = pd.DataFrame(all_seed_rows)
        confusions = pd.DataFrame(all_cm)
        splits = pd.DataFrame(all_splits)
        partition_metrics = pd.DataFrame(all_partition_metrics)
        class_metrics = pd.DataFrame(all_class_metrics)
        assignments = pd.DataFrame(all_assignments)
        core_predictions = pd.DataFrame(all_predictions)
        selected = choose_best_by_validation(results)

        results.to_csv(TABLE_DIR / "manifest_da_baseline_split_results.csv", index=False)
        seed_results.to_csv(TABLE_DIR / "manifest_da_baseline_seed_results.csv", index=False)
        confusions.to_csv(TABLE_DIR / "manifest_da_baseline_split_confusions.csv", index=False)
        splits.to_csv(TABLE_DIR / "manifest_da_baseline_split_splits.csv", index=False)
        assignments.to_csv(TABLE_DIR / "manifest_da_baseline_split_assignments.csv", index=False)
        partition_metrics.to_csv(TABLE_DIR / "manifest_da_baseline_partition_metrics.csv", index=False)
        class_metrics.to_csv(TABLE_DIR / "manifest_da_baseline_class_metrics.csv", index=False)
        core_predictions.to_csv(CORE_TEST_PREDICTIONS_PATH, index=False)
        selected.to_csv(TABLE_DIR / "manifest_da_baseline_selected_summary.csv", index=False)
        write_summary_svg(results, FIG_DIR / "manifest_da_baseline_split_summary.svg")
        if RUN_SCOPE == "core":
            print(results[["split", "model", "training_randomness", "seed_count", "test_accuracy", "test_accuracy_variance", "test_accuracy_min", "test_accuracy_max"]].to_string(index=False))
            return
    else:
        required_core = {
            "results": TABLE_DIR / "manifest_da_baseline_split_results.csv",
            "selected": TABLE_DIR / "manifest_da_baseline_selected_summary.csv",
            "splits": TABLE_DIR / "manifest_da_baseline_split_splits.csv",
            "predictions": CORE_TEST_PREDICTIONS_PATH,
        }
        missing = [str(path) for path in required_core.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Per-batch-only mode requires existing core outputs; missing: {missing}")
        results = pd.read_csv(required_core["results"])
        selected = pd.read_csv(required_core["selected"])
        splits = pd.read_csv(required_core["splits"])
        core_predictions = pd.read_csv(required_core["predictions"])

    per_batch_rows, per_batch_seed_rows, per_batch_cm, per_batch_splits = [], [], [], []
    per_batch_partition_metrics, per_batch_class_metrics, per_batch_assignments, per_batch_predictions = [], [], [], []
    for task_name, classes in TASKS:
        rows, seed_rows, cm_rows, split_rows, partition_metric_rows, class_metric_rows, assignment_rows, prediction_rows = evaluate_fixed_period_i_per_batch(features, task_name, classes)
        per_batch_rows.extend(rows)
        per_batch_seed_rows.extend(seed_rows)
        per_batch_cm.extend(cm_rows)
        per_batch_splits.extend(split_rows)
        per_batch_partition_metrics.extend(partition_metric_rows)
        per_batch_class_metrics.extend(class_metric_rows)
        per_batch_assignments.extend(assignment_rows)
        per_batch_predictions.extend(prediction_rows)
    per_batch_results = pd.DataFrame(per_batch_rows)
    per_batch_seed_results = pd.DataFrame(per_batch_seed_rows)
    per_batch_confusions = pd.DataFrame(per_batch_cm)
    per_batch_split_audit = pd.DataFrame(per_batch_splits)
    per_batch_partition_metrics_df = pd.DataFrame(per_batch_partition_metrics)
    per_batch_class_metrics_df = pd.DataFrame(per_batch_class_metrics)
    per_batch_assignments_df = pd.DataFrame(per_batch_assignments)
    per_batch_predictions_df = pd.DataFrame(per_batch_predictions)
    per_batch_selected = choose_fixed_validation_model_for_batches(per_batch_results)

    per_batch_results.to_csv(TABLE_DIR / "manifest_da_baseline_period_i_per_batch_results.csv", index=False)
    per_batch_seed_results.to_csv(TABLE_DIR / "manifest_da_baseline_period_i_per_batch_seed_results.csv", index=False)
    per_batch_confusions.to_csv(TABLE_DIR / "manifest_da_baseline_period_i_per_batch_confusions.csv", index=False)
    per_batch_split_audit.to_csv(TABLE_DIR / "manifest_da_baseline_period_i_per_batch_splits.csv", index=False)
    per_batch_assignments_df.to_csv(TABLE_DIR / "manifest_da_baseline_period_i_per_batch_assignments.csv", index=False)
    per_batch_partition_metrics_df.to_csv(TABLE_DIR / "manifest_da_baseline_period_i_per_batch_partition_metrics.csv", index=False)
    per_batch_class_metrics_df.to_csv(TABLE_DIR / "manifest_da_baseline_period_i_per_batch_class_metrics.csv", index=False)
    per_batch_predictions_df.to_csv(PB_TEST_PREDICTIONS_PATH, index=False)
    interactive_predictions = pd.concat([core_predictions, per_batch_predictions_df], ignore_index=True)
    interactive_predictions.to_csv(INTERACTIVE_PREDICTIONS_PATH, index=False)
    per_batch_selected.to_csv(TABLE_DIR / "manifest_da_baseline_period_i_per_batch_selected_summary.csv", index=False)
    write_summary_svg(results, FIG_DIR / "manifest_da_baseline_split_summary.svg")
    write_publication_figures(results, per_batch_results)
    write_results_html(selected, splits, per_batch_results, per_batch_selected, HTML_REPORT_PATH)
    write_standalone_interactive_report(INTERACTIVE_REPORT_PATH, INTERACTIVE_PREDICTIONS_PATH)
    print(per_batch_results[["test_batch", "model", "training_randomness", "seed_count", "test_accuracy", "test_accuracy_variance", "test_accuracy_min", "test_accuracy_max"]].to_string(index=False))


if __name__ == "__main__":
    main()
