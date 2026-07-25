from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


SENSORS = [f"s{i}" for i in range(1, 7)]
CLASSES = ["air", "alcohol", "acetone"]


def locate_output_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if parent.name.lower() == "thesis":
            return parent
        if (parent / "thesis_out").exists():
            return parent / "thesis_out"
    return path.parents[2] / "thesis_out"


def locate_data_root(output_root: Path) -> Path:
    candidates = [
        output_root / "enose_data",
        output_root.parent / "enose_data",
        Path(r"D:\datasets\enose_data"),
        Path(r"D:\datasets\enose_export\enose_data"),
    ]
    for candidate in candidates:
        if all((candidate / label).exists() for label in CLASSES):
            return candidate
    raise FileNotFoundError("Could not locate enose_data with air/alcohol/acetone folders.")


OUT_ROOT = locate_output_root()
DATA_ROOT = locate_data_root(OUT_ROOT)
TABLE_DIR = OUT_ROOT / "tables"
PROCESSED_DIR = OUT_ROOT / "processed"


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def relative_time_seconds(df: pd.DataFrame) -> np.ndarray:
    t = df["arduino_time"].to_numpy(dtype=float)
    return (t - t[0]) / 1000.0


def baseline_mask(t: np.ndarray, label: str) -> np.ndarray:
    # Baseline correction is the only phase-aware operation here.
    if label == "air":
        end = min(60.0, max(20.0, float(np.nanmax(t)) * 0.35))
    else:
        end = 60.0
    mask = t <= end
    if mask.sum() < 3:
        mask = np.arange(len(t)) < max(3, int(len(t) * 0.2))
    return mask


def read_corrected_sequences(path: Path) -> dict:
    df = pd.read_csv(path)
    label = str(df["gas_type"].iloc[0]).strip().lower()
    t = relative_time_seconds(df)
    bmask = baseline_mask(t, label)
    sequences: dict[str, np.ndarray | str | int | float] = {
        "sample_id": path.stem,
        "label": label,
        "duration_s": int(float(df["duration_s"].iloc[0])),
        "collection_date": pd.Timestamp(path.stat().st_mtime_ns).date().isoformat(),
        "n_points": len(df),
        "total_seconds": float(np.nanmax(t)),
    }
    for sensor in SENSORS:
        x = df[sensor].to_numpy(dtype=float)
        b0 = float(np.nanmedian(x[bmask]))
        sequences[sensor] = (x - b0) / max(abs(b0), 1e-9)
    return sequences


def load_sequences() -> list[dict]:
    rows = []
    for label in CLASSES:
        for path in sorted((DATA_ROOT / label).glob("*.csv")):
            rows.append(read_corrected_sequences(path))
    return rows


def sequences_to_frame(rows: list[dict]) -> pd.DataFrame:
    max_len = max(int(row["n_points"]) for row in rows)
    flat_rows = []
    for row in rows:
        out = {
            "sample_id": row["sample_id"],
            "label": row["label"],
            "duration_s": row["duration_s"],
            "collection_date": row["collection_date"],
            "n_points": row["n_points"],
            "total_seconds": row["total_seconds"],
        }
        for sensor in SENSORS:
            seq = np.asarray(row[sensor], dtype=float)
            for i, value in enumerate(seq):
                out[f"{sensor}_p{i:04d}"] = float(value)
            for i in range(len(seq), max_len):
                out[f"{sensor}_p{i:04d}"] = np.nan
        flat_rows.append(out)
    return pd.DataFrame(flat_rows)


def encode(y: np.ndarray) -> np.ndarray:
    return np.array([{label: i for i, label in enumerate(CLASSES)}[v] for v in y], dtype=int)


def sensor_columns(columns: list[str], sensors: list[str]) -> list[str]:
    selected = []
    for sensor in sensors:
        selected.extend([c for c in columns if c.startswith(sensor + "_p")])
    return selected


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0)
    std[std < 1e-9] = 1.0
    return mean, std


def apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (x - mean) / std
    return z


def masked_centroid_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    centroids = []
    for c in range(len(CLASSES)):
        centroids.append(np.nanmean(x_train[y_train == c], axis=0))
    centroids = np.vstack(centroids)
    preds = []
    for x in x_test:
        scores = []
        for centroid in centroids:
            valid = np.isfinite(x) & np.isfinite(centroid)
            if valid.sum() == 0:
                scores.append(np.inf)
            else:
                scores.append(float(np.mean((x[valid] - centroid[valid]) ** 2)))
        preds.append(int(np.argmin(scores)))
    return np.array(preds, dtype=int)


def masked_gnb_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    means, vars_, priors = [], [], []
    for c in range(len(CLASSES)):
        xc = x_train[y_train == c]
        mu = np.nanmean(xc, axis=0)
        var = np.nanvar(xc, axis=0)
        means.append(np.nan_to_num(mu, nan=0.0))
        vars_.append(np.nan_to_num(var, nan=1.0) + 1e-5)
        priors.append(np.log(len(xc) / len(x_train)))
    means = np.vstack(means)
    vars_ = np.vstack(vars_)
    preds = []
    for x in x_test:
        scores = []
        valid_x = np.isfinite(x)
        for c in range(len(CLASSES)):
            valid = valid_x & np.isfinite(means[c]) & np.isfinite(vars_[c])
            if valid.sum() == 0:
                scores.append(-np.inf)
                continue
            ll = -0.5 * (
                np.log(2 * np.pi * vars_[c, valid])
                + ((x[valid] - means[c, valid]) ** 2) / vars_[c, valid]
            )
            scores.append(float(np.mean(ll) + priors[c]))
        preds.append(int(np.argmax(scores)))
    return np.array(preds, dtype=int)


CLASSIFIERS = {
    "original_points_centroid": masked_centroid_predict,
    "original_points_gnb": masked_gnb_predict,
}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    recalls, f1s = [], []
    for c in range(len(CLASSES)):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        recall = tp / max(1, tp + fn)
        precision = tp / max(1, tp + fp)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": float(np.trace(cm) / max(1, cm.sum())),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "confusion": cm,
    }


def stratified_splits(y: np.ndarray, n_splits: int = 100, test_frac: float = 0.30, seed: int = 23):
    rng = np.random.default_rng(seed)
    class_indices = [np.where(y == c)[0] for c in range(len(CLASSES))]
    for split_id in range(n_splits):
        train, test = [], []
        for idx in class_indices:
            perm = rng.permutation(idx)
            n_test = max(1, int(round(len(idx) * test_frac)))
            test.extend(perm[:n_test].tolist())
            train.extend(perm[n_test:].tolist())
        yield split_id, np.array(train, dtype=int), np.array(test, dtype=int)


def leave_group_out(groups: np.ndarray, y: np.ndarray):
    for split_id, group in enumerate(sorted(set(groups))):
        test = np.where(groups == group)[0]
        train = np.where(groups != group)[0]
        if len(test) >= 3 and len(set(y[train])) == len(CLASSES):
            yield split_id, train, test, str(group)


def evaluate(df: pd.DataFrame, cols: list[str], train: np.ndarray, test: np.ndarray, classifier: str) -> dict:
    x = df[cols].to_numpy(dtype=float)
    y = encode(df["label"].to_numpy())
    mean, std = fit_standardizer(x[train])
    x_train = apply_standardizer(x[train], mean, std)
    x_test = apply_standardizer(x[test], mean, std)
    pred = CLASSIFIERS[classifier](x_train, y[train], x_test)
    return metrics(y[test], pred)


def run_ablation(df: pd.DataFrame, splits, mode: str) -> pd.DataFrame:
    columns = df.columns.tolist()
    cases = {"all": SENSORS}
    for sensor in SENSORS:
        cases[f"without_{sensor}"] = [s for s in SENSORS if s != sensor]
    rows = []
    for split in splits:
        if len(split) == 4:
            split_id, train, test, group = split
        else:
            split_id, train, test = split
            group = ""
        for clf in CLASSIFIERS:
            full = evaluate(df, sensor_columns(columns, SENSORS), train, test, clf)
            for case, sensors in cases.items():
                cols = sensor_columns(columns, sensors)
                m = evaluate(df, cols, train, test, clf)
                rows.append({
                    "mode": mode,
                    "split_id": split_id,
                    "group": group,
                    "classifier": clf,
                    "case": case,
                    "removed_sensor": "" if case == "all" else case.replace("without_", ""),
                    "n_original_point_columns": len(cols),
                    "accuracy": m["accuracy"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "macro_f1": m["macro_f1"],
                    "delta_balanced_accuracy_vs_all": full["balanced_accuracy"] - m["balanced_accuracy"],
                    "delta_macro_f1_vs_all": full["macro_f1"] - m["macro_f1"],
                })
    return pd.DataFrame(rows)


def run_single_sensor(df: pd.DataFrame, splits, mode: str) -> pd.DataFrame:
    columns = df.columns.tolist()
    rows = []
    for split in splits:
        if len(split) == 4:
            split_id, train, test, group = split
        else:
            split_id, train, test = split
            group = ""
        for clf in CLASSIFIERS:
            for sensor in SENSORS:
                m = evaluate(df, sensor_columns(columns, [sensor]), train, test, clf)
                rows.append({
                    "mode": mode,
                    "split_id": split_id,
                    "group": group,
                    "classifier": clf,
                    "sensor": sensor,
                    "balanced_accuracy": m["balanced_accuracy"],
                    "macro_f1": m["macro_f1"],
                })
    return pd.DataFrame(rows)


def summarize(ablation: pd.DataFrame, single: pd.DataFrame) -> pd.DataFrame:
    abl = ablation[ablation["case"] != "all"].groupby(["mode", "classifier", "removed_sensor"], as_index=False).agg(
        mean_delta_balanced_accuracy=("delta_balanced_accuracy_vs_all", "mean"),
        mean_delta_macro_f1=("delta_macro_f1_vs_all", "mean"),
        positive_delta_rate=("delta_balanced_accuracy_vs_all", lambda s: float(np.mean(np.asarray(s) > 0))),
        mean_balanced_accuracy_without=("balanced_accuracy", "mean"),
    )
    one = single.groupby(["mode", "classifier", "sensor"], as_index=False).agg(
        single_sensor_mean_balanced_accuracy=("balanced_accuracy", "mean"),
        single_sensor_mean_macro_f1=("macro_f1", "mean"),
    ).rename(columns={"sensor": "removed_sensor"})
    return abl.merge(one, on=["mode", "classifier", "removed_sensor"], how="left")


def main() -> None:
    ensure_dirs()
    seq_rows = load_sequences()
    df = sequences_to_frame(seq_rows)
    y = encode(df["label"].to_numpy())
    dates = df["collection_date"].to_numpy()
    random_splits = list(stratified_splits(y))
    date_splits = list(leave_group_out(dates, y))
    nb_test = np.where(np.isin(dates, ["2026-06-04", "2026-06-05"]))[0]
    nb_train = np.where(~np.isin(dates, ["2026-06-04", "2026-06-05"]))[0]
    new_bottle_splits = [(0, nb_train, nb_test, "2026-06-04_2026-06-05")]

    ablation = pd.concat([
        run_ablation(df, random_splits, "original_points_stratified_random_holdout"),
        run_ablation(df, date_splits, "original_points_leave_collection_date_out"),
        run_ablation(df, new_bottle_splits, "original_points_new_bottle_holdout"),
    ], ignore_index=True)
    single = pd.concat([
        run_single_sensor(df, random_splits, "original_points_stratified_random_holdout"),
        run_single_sensor(df, date_splits, "original_points_leave_collection_date_out"),
        run_single_sensor(df, new_bottle_splits, "original_points_new_bottle_holdout"),
    ], ignore_index=True)
    summary = summarize(ablation, single)

    df.to_csv(PROCESSED_DIR / "original_points_baseline_corrected_matrix.csv", index=False)
    ablation.to_csv(TABLE_DIR / "original_points_sensor_ablation_all_results.csv", index=False)
    single.to_csv(TABLE_DIR / "original_points_single_sensor_results.csv", index=False)
    summary.to_csv(TABLE_DIR / "original_points_sensor_importance_summary.csv", index=False)
    with (TABLE_DIR / "original_points_analysis_overview.txt").open("w", encoding="utf-8") as f:
        f.write(f"n_samples: {len(df)}\n")
        f.write(f"class_counts: {dict(Counter(df['label']))}\n")
        f.write(f"max_original_points_per_sensor: {max(int(row['n_points']) for row in seq_rows)}\n")
        f.write("processing: baseline correction only; no exposure/recovery feature extraction; no resampling\n")
        f.write("variable_length_handling: shorter recordings are padded with NaN; masked classifiers compare only observed original point columns\n")
    print(TABLE_DIR / "original_points_sensor_importance_summary.csv")
    print(PROCESSED_DIR / "original_points_baseline_corrected_matrix.csv")


if __name__ == "__main__":
    main()
