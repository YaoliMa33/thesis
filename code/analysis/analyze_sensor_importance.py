from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


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
        if (candidate / "air").exists() and (candidate / "alcohol").exists() and (candidate / "acetone").exists():
            return candidate
    raise FileNotFoundError("Could not locate enose_data with air/alcohol/acetone folders.")


OUT_ROOT = locate_output_root()
DATA_ROOT = locate_data_root(OUT_ROOT)
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
PROCESSED_DIR = OUT_ROOT / "processed"

SENSORS = [f"s{i}" for i in range(1, 7)]
CLASSES = ["air", "alcohol", "acetone"]
GAS_SECONDS_BY_DURATION = {
    180: 60.0,
    125: 5.0,
    130: 10.0,
}


@dataclass(frozen=True)
class SampleMeta:
    sample_id: str
    path: Path
    label: str
    duration_s: int
    collection_date: str
    distance_cm: float
    wind_level: str
    total_seconds: float
    n_rows: int


def ensure_dirs() -> None:
    for directory in (TABLE_DIR, FIG_DIR, PROCESSED_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def rel_time_seconds(df: pd.DataFrame) -> np.ndarray:
    if "arduino_time" in df.columns:
        t = df["arduino_time"].to_numpy(dtype=float)
        return (t - t[0]) / 1000.0
    t = df["time_s"].to_numpy(dtype=float)
    return t - t[0]


def phase_masks(label: str, duration_s: int, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    total = float(np.nanmax(t)) if len(t) else 0.0
    if label == "air":
        baseline_end = min(20.0, max(5.0, total * 0.35))
        exposure_start = baseline_end
        exposure_end = max(exposure_start + 1.0, total)
        recovery_start = max(exposure_start, total * 0.70)
    else:
        gas_seconds = GAS_SECONDS_BY_DURATION.get(int(duration_s), max(1.0, float(duration_s) - 120.0))
        baseline_end = min(60.0, max(5.0, total * 0.45))
        exposure_start = 60.0
        exposure_end = min(total, exposure_start + gas_seconds)
        if exposure_end <= exposure_start:
            exposure_start = baseline_end
            exposure_end = min(total, baseline_end + max(1.0, gas_seconds))
        recovery_start = exposure_end

    baseline = t <= baseline_end
    exposure = (t >= exposure_start) & (t <= exposure_end)
    recovery = t >= recovery_start

    if baseline.sum() < 3:
        baseline = np.arange(len(t)) < max(3, int(len(t) * 0.20))
    if exposure.sum() < 3:
        start = max(0, int(len(t) * 0.45))
        stop = max(start + 3, int(len(t) * 0.65))
        exposure = np.zeros(len(t), dtype=bool)
        exposure[start:min(stop, len(t))] = True
    if recovery.sum() < 3:
        start = max(0, int(len(t) * 0.75))
        recovery = np.zeros(len(t), dtype=bool)
        recovery[start:] = True

    info = {
        "baseline_end_s": baseline_end,
        "exposure_start_s": exposure_start,
        "exposure_end_s": exposure_end,
        "recovery_start_s": recovery_start,
    }
    return baseline, exposure, recovery, info


def slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.nanstd(x) == 0:
        return 0.0
    xm = x - np.nanmean(x)
    ym = y - np.nanmean(y)
    denom = float(np.dot(xm, xm))
    return float(np.dot(xm, ym) / denom) if denom else 0.0


def extract_features(path: Path) -> tuple[dict, SampleMeta, dict]:
    df = pd.read_csv(path)
    label = str(df.get("gas_type", df.get("label")).iloc[0]).strip().lower()
    duration_s = int(float(df.get("duration_s", pd.Series([0])).iloc[0]))
    distance = float(df.get("distance_cm", pd.Series([np.nan])).iloc[0])
    wind = str(df.get("wind_level", pd.Series([""])).iloc[0])
    t = rel_time_seconds(df)
    baseline_mask, exposure_mask, recovery_mask, phase_info = phase_masks(label, duration_s, t)

    features: dict[str, float | str] = {
        "sample_id": path.stem,
        "label": label,
        "duration_s": duration_s,
        "collection_date": path.stat().st_mtime_ns,
        "distance_cm": distance,
        "wind_level": wind,
    }
    baseline_rows = []

    for sensor in SENSORS:
        x = df[sensor].to_numpy(dtype=float)
        b0 = float(np.nanmedian(x[baseline_mask]))
        denom = max(abs(b0), 1e-9)
        z = (x - b0) / denom
        exp = z[exposure_mask]
        rec = z[recovery_mask]
        exp_t = t[exposure_mask]
        rec_t = t[recovery_mask]
        abs_exp = np.abs(exp)
        tail_n = max(1, int(math.ceil(len(exp) * 0.25)))
        rec_tail_n = max(1, int(math.ceil(len(rec) * 0.25)))

        features[f"{sensor}_baseline"] = b0
        features[f"{sensor}_exp_mean"] = float(np.nanmean(exp))
        features[f"{sensor}_exp_absmax"] = float(np.nanmax(abs_exp))
        features[f"{sensor}_exp_span"] = float(np.nanmax(exp) - np.nanmin(exp))
        features[f"{sensor}_exp_tail"] = float(np.nanmean(exp[-tail_n:]))
        features[f"{sensor}_exp_auc_abs"] = float(np.nanmean(abs_exp))
        features[f"{sensor}_rec_mean"] = float(np.nanmean(rec))
        features[f"{sensor}_rec_tail"] = float(np.nanmean(rec[-rec_tail_n:]))
        features[f"{sensor}_rec_slope"] = slope(rec_t, rec)
        features[f"{sensor}_overall_std"] = float(np.nanstd(z))
        baseline_rows.append({"sample_id": path.stem, "sensor": sensor, "baseline": b0})

    meta = SampleMeta(
        sample_id=path.stem,
        path=path,
        label=label,
        duration_s=duration_s,
        collection_date=pd.Timestamp(path.stat().st_mtime_ns).date().isoformat(),
        distance_cm=distance,
        wind_level=wind,
        total_seconds=float(np.nanmax(t)),
        n_rows=len(df),
    )
    features["collection_date"] = meta.collection_date
    return features, meta, phase_info


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    metas = []
    phases = []
    for label in CLASSES:
        for path in sorted((DATA_ROOT / label).glob("*.csv")):
            features, meta, phase_info = extract_features(path)
            rows.append(features)
            metas.append(meta.__dict__ | {"path": str(meta.path)})
            phases.append({"sample_id": meta.sample_id, **phase_info})

    features = pd.DataFrame(rows)
    meta_df = pd.DataFrame(metas)
    phase_df = pd.DataFrame(phases)
    features = features[features["label"].isin(CLASSES)].reset_index(drop=True)
    return features, meta_df, phase_df


def feature_columns_for(sensors: list[str], all_columns: list[str]) -> list[str]:
    cols = []
    for sensor in sensors:
        cols.extend([c for c in all_columns if c.startswith(sensor + "_") and not c.endswith("_baseline")])
    return cols


def encode_labels(y: np.ndarray) -> np.ndarray:
    idx = {label: i for i, label in enumerate(CLASSES)}
    return np.array([idx[v] for v in y], dtype=int)


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std < 1e-9] = 1.0
    return mean, std


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict_centroid(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    centroids = []
    for c in range(len(CLASSES)):
        centroids.append(np.mean(x_train[y_train == c], axis=0))
    centroids = np.vstack(centroids)
    d2 = ((x_test[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d2, axis=1)


def fit_predict_gnb(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    means, vars_, priors = [], [], []
    for c in range(len(CLASSES)):
        xc = x_train[y_train == c]
        means.append(np.mean(xc, axis=0))
        vars_.append(np.var(xc, axis=0) + 1e-6)
        priors.append(len(xc) / len(x_train))
    means = np.vstack(means)
    vars_ = np.vstack(vars_)
    priors = np.log(np.array(priors))
    scores = []
    for c in range(len(CLASSES)):
        ll = -0.5 * (np.log(2 * np.pi * vars_[c]) + ((x_test - means[c]) ** 2) / vars_[c]).sum(axis=1)
        scores.append(ll + priors[c])
    return np.argmax(np.vstack(scores).T, axis=1)


def fit_predict_lda(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    n_features = x_train.shape[1]
    means = []
    priors = []
    centered = []
    for c in range(len(CLASSES)):
        xc = x_train[y_train == c]
        mu = np.mean(xc, axis=0)
        means.append(mu)
        priors.append(len(xc) / len(x_train))
        centered.append(xc - mu)
    centered_all = np.vstack(centered)
    cov = centered_all.T @ centered_all / max(1, len(x_train) - len(CLASSES))
    diag = np.diag(np.diag(cov))
    cov = 0.75 * cov + 0.25 * diag + 1e-4 * np.eye(n_features)
    inv_cov = np.linalg.pinv(cov)
    means = np.vstack(means)
    scores = x_test @ inv_cov @ means.T
    scores -= 0.5 * np.sum((means @ inv_cov) * means, axis=1)
    scores += np.log(np.array(priors))
    return np.argmax(scores, axis=1)


CLASSIFIERS = {
    "lda": fit_predict_lda,
    "gnb": fit_predict_gnb,
    "centroid": fit_predict_centroid,
}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    recalls = []
    f1s = []
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


def stratified_splits(y: np.ndarray, n_splits: int = 100, test_frac: float = 0.30, seed: int = 7):
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


def leave_group_out_splits(groups: np.ndarray, y: np.ndarray):
    for split_id, group in enumerate(sorted(set(groups))):
        test = np.where(groups == group)[0]
        train = np.where(groups != group)[0]
        if len(set(y[train])) < len(CLASSES) or len(test) < 3:
            continue
        yield split_id, train, test, str(group)


def new_bottle_split(dates: np.ndarray):
    test = np.where(np.isin(dates, ["2026-06-04", "2026-06-05"]))[0]
    train = np.where(~np.isin(dates, ["2026-06-04", "2026-06-05"]))[0]
    return train, test


def evaluate_split(df: pd.DataFrame, cols: list[str], train: np.ndarray, test: np.ndarray, classifier: str) -> tuple[dict, np.ndarray]:
    x = df[cols].to_numpy(dtype=float)
    y = encode_labels(df["label"].to_numpy())
    mean, std = standardize_fit(x[train])
    x_train = standardize_apply(x[train], mean, std)
    x_test = standardize_apply(x[test], mean, std)
    y_train, y_test = y[train], y[test]
    pred = CLASSIFIERS[classifier](x_train, y_train, x_test)
    return metrics(y_test, pred), pred


def aggregate_ablation(df: pd.DataFrame, splits, all_cols: list[str], mode_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    confusion_rows = []
    sensor_sets = {"all": SENSORS}
    for sensor in SENSORS:
        sensor_sets[f"without_{sensor}"] = [s for s in SENSORS if s != sensor]

    for split in splits:
        if len(split) == 4:
            split_id, train, test, group = split
        else:
            split_id, train, test = split
            group = ""
        for classifier in CLASSIFIERS:
            baseline_cols = feature_columns_for(sensor_sets["all"], all_cols)
            baseline_metrics, _ = evaluate_split(df, baseline_cols, train, test, classifier)
            for case_name, sensors in sensor_sets.items():
                cols = feature_columns_for(sensors, all_cols)
                m, _ = evaluate_split(df, cols, train, test, classifier)
                rows.append({
                    "mode": mode_name,
                    "split_id": split_id,
                    "group": group,
                    "classifier": classifier,
                    "case": case_name,
                    "removed_sensor": "" if case_name == "all" else case_name.replace("without_", ""),
                    "n_features": len(cols),
                    "accuracy": m["accuracy"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "macro_f1": m["macro_f1"],
                    "delta_balanced_accuracy_vs_all": baseline_metrics["balanced_accuracy"] - m["balanced_accuracy"],
                    "delta_macro_f1_vs_all": baseline_metrics["macro_f1"] - m["macro_f1"],
                })
                cm = m["confusion"]
                for i, true_label in enumerate(CLASSES):
                    for j, pred_label in enumerate(CLASSES):
                        confusion_rows.append({
                            "mode": mode_name,
                            "split_id": split_id,
                            "group": group,
                            "classifier": classifier,
                            "case": case_name,
                            "true_label": true_label,
                            "pred_label": pred_label,
                            "count": int(cm[i, j]),
                        })
    return pd.DataFrame(rows), pd.DataFrame(confusion_rows)


def evaluate_single_sensors(df: pd.DataFrame, splits, all_cols: list[str], mode_name: str) -> pd.DataFrame:
    rows = []
    for split in splits:
        if len(split) == 4:
            split_id, train, test, group = split
        else:
            split_id, train, test = split
            group = ""
        for classifier in CLASSIFIERS:
            for sensor in SENSORS:
                cols = feature_columns_for([sensor], all_cols)
                m, _ = evaluate_split(df, cols, train, test, classifier)
                rows.append({
                    "mode": mode_name,
                    "split_id": split_id,
                    "group": group,
                    "classifier": classifier,
                    "sensor": sensor,
                    "accuracy": m["accuracy"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "macro_f1": m["macro_f1"],
                })
    return pd.DataFrame(rows)


def evaluate_sensor_triplets(df: pd.DataFrame, splits, all_cols: list[str], mode_name: str) -> pd.DataFrame:
    rows = []
    for split in splits:
        if len(split) == 4:
            split_id, train, test, group = split
        else:
            split_id, train, test = split
            group = ""
        for classifier in CLASSIFIERS:
            for combo in combinations(SENSORS, 3):
                cols = feature_columns_for(list(combo), all_cols)
                m, _ = evaluate_split(df, cols, train, test, classifier)
                rows.append({
                    "mode": mode_name,
                    "split_id": split_id,
                    "group": group,
                    "classifier": classifier,
                    "sensors": "+".join(combo),
                    "accuracy": m["accuracy"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "macro_f1": m["macro_f1"],
                })
    return pd.DataFrame(rows)


def fisher_scores(df: pd.DataFrame, all_cols: list[str]) -> pd.DataFrame:
    y = df["label"].to_numpy()
    rows = []
    for sensor in SENSORS:
        scores = []
        for col in feature_columns_for([sensor], all_cols):
            values = df[col].to_numpy(dtype=float)
            overall = np.nanmean(values)
            between = 0.0
            within = 0.0
            for label in CLASSES:
                vc = values[y == label]
                between += len(vc) * (np.nanmean(vc) - overall) ** 2
                within += np.nansum((vc - np.nanmean(vc)) ** 2)
            scores.append(float(between / max(within, 1e-12)))
        rows.append({
            "sensor": sensor,
            "fisher_mean": float(np.mean(scores)),
            "fisher_max": float(np.max(scores)),
            "fisher_top3_mean": float(np.mean(sorted(scores, reverse=True)[:3])),
        })
    return pd.DataFrame(rows)


def sensor_response_summary(df: pd.DataFrame, all_cols: list[str]) -> pd.DataFrame:
    rows = []
    for sensor in SENSORS:
        sensor_cols = feature_columns_for([sensor], all_cols)
        exp_col = f"{sensor}_exp_absmax"
        for label in CLASSES:
            part = df[df["label"] == label]
            rows.append({
                "sensor": sensor,
                "label": label,
                "n": len(part),
                "median_exp_absmax": float(np.median(part[exp_col])),
                "mean_exp_absmax": float(np.mean(part[exp_col])),
                "median_feature_abs": float(np.median(np.abs(part[sensor_cols].to_numpy(dtype=float)))),
            })
    return pd.DataFrame(rows)


def pca_scores(df: pd.DataFrame, all_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = df[all_cols].to_numpy(dtype=float)
    mean, std = standardize_fit(x)
    z = standardize_apply(x, mean, std)
    _, s, vt = np.linalg.svd(z, full_matrices=False)
    components = vt[:3]
    scores = z @ components.T
    explained = (s ** 2) / max(1, len(z) - 1)
    explained_ratio = explained / explained.sum()
    score_df = pd.DataFrame({
        "sample_id": df["sample_id"],
        "label": df["label"],
        "collection_date": df["collection_date"],
        "PC1": scores[:, 0],
        "PC2": scores[:, 1],
        "PC3": scores[:, 2],
    })
    loading_rows = []
    for sensor in SENSORS:
        idx = [all_cols.index(c) for c in feature_columns_for([sensor], all_cols)]
        contrib = float(np.sum((components[:, idx] ** 2) * explained_ratio[:3, None]))
        loading_rows.append({
            "sensor": sensor,
            "weighted_pc1_pc2_pc3_contribution": contrib,
            "pc1_abs_loading_sum": float(np.sum(np.abs(components[0, idx]))),
            "pc2_abs_loading_sum": float(np.sum(np.abs(components[1, idx]))),
            "pc3_abs_loading_sum": float(np.sum(np.abs(components[2, idx]))),
        })
    return score_df, pd.DataFrame(loading_rows)


def summarize_ablation(ablation: pd.DataFrame, mode_name: str) -> pd.DataFrame:
    sub = ablation[(ablation["mode"] == mode_name) & (ablation["case"] != "all")]
    grouped = sub.groupby(["classifier", "removed_sensor"], as_index=False).agg(
        mean_delta_balanced_accuracy=("delta_balanced_accuracy_vs_all", "mean"),
        std_delta_balanced_accuracy=("delta_balanced_accuracy_vs_all", "std"),
        mean_delta_macro_f1=("delta_macro_f1_vs_all", "mean"),
        mean_balanced_accuracy_without=("balanced_accuracy", "mean"),
        mean_macro_f1_without=("macro_f1", "mean"),
        positive_delta_rate=("delta_balanced_accuracy_vs_all", lambda s: float(np.mean(np.asarray(s) > 0))),
    )
    all_rows = ablation[(ablation["mode"] == mode_name) & (ablation["case"] == "all")]
    all_summary = all_rows.groupby("classifier", as_index=False).agg(
        full_mean_balanced_accuracy=("balanced_accuracy", "mean"),
        full_mean_macro_f1=("macro_f1", "mean"),
        full_std_balanced_accuracy=("balanced_accuracy", "std"),
    )
    return grouped.merge(all_summary, on="classifier", how="left")


def svg_bar(summary: pd.DataFrame, path: Path) -> None:
    lda = summary[summary["classifier"] == "lda"].copy()
    lda = lda.sort_values("mean_delta_balanced_accuracy", ascending=False)
    width, height = 760, 340
    left, top = 90, 35
    chart_w, chart_h = 610, 240
    vals = lda["mean_delta_balanced_accuracy"].to_numpy()
    max_abs = max(0.01, float(np.max(np.abs(vals))))
    zero_x = left + chart_w * 0.5
    rows = []
    for i, row in enumerate(lda.itertuples(index=False)):
        y = top + i * 36
        v = float(row.mean_delta_balanced_accuracy)
        bar_w = (abs(v) / max_abs) * (chart_w * 0.45)
        x = zero_x if v >= 0 else zero_x - bar_w
        color = "#2563eb" if v >= 0 else "#dc2626"
        rows.append(f'<text x="24" y="{y + 17}" font-size="14">{row.removed_sensor.upper()}</text>')
        rows.append(f'<rect x="{x:.1f}" y="{y}" width="{bar_w:.1f}" height="22" fill="{color}" rx="3"/>')
        rows.append(f'<text x="{x + (bar_w + 6 if v >= 0 else -54):.1f}" y="{y + 16}" font-size="12">{v:+.3f}</text>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="24" y="24" font-size="18" font-family="Arial">LDA leave-one-sensor-out importance</text>
<line x1="{zero_x}" y1="{top - 8}" x2="{zero_x}" y2="{top + chart_h}" stroke="#111827" stroke-width="1"/>
<text x="{left}" y="{height - 28}" font-size="12" font-family="Arial">Positive: removing the sensor hurts balanced accuracy; negative: removing it helps.</text>
<g font-family="Arial">{''.join(rows)}</g>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def svg_pca(scores: pd.DataFrame, path: Path) -> None:
    width, height = 720, 520
    left, top, chart_w, chart_h = 70, 45, 590, 400
    x = scores["PC1"].to_numpy(dtype=float)
    y = scores["PC2"].to_numpy(dtype=float)
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    colors = {"air": "#111827", "alcohol": "#2563eb", "acetone": "#dc2626"}

    def sx(v):
        return left + (v - x_min) / max(1e-9, x_max - x_min) * chart_w

    def sy(v):
        return top + chart_h - (v - y_min) / max(1e-9, y_max - y_min) * chart_h

    dots = []
    for row in scores.itertuples(index=False):
        dots.append(
            f'<circle cx="{sx(row.PC1):.1f}" cy="{sy(row.PC2):.1f}" r="4" fill="{colors[row.label]}" fill-opacity="0.75"/>'
        )
    legend = []
    for i, label in enumerate(CLASSES):
        lx = left + i * 115
        ly = height - 32
        legend.append(f'<circle cx="{lx}" cy="{ly}" r="5" fill="{colors[label]}"/>')
        legend.append(f'<text x="{lx + 10}" y="{ly + 4}" font-size="13">{label}</text>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="24" y="25" font-size="18" font-family="Arial">Baseline-corrected feature space, PCA projection</text>
<rect x="{left}" y="{top}" width="{chart_w}" height="{chart_h}" fill="#f9fafb" stroke="#d1d5db"/>
<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#111827"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#111827"/>
<text x="{left + chart_w / 2 - 20}" y="{height - 58}" font-size="13" font-family="Arial">PC1</text>
<text x="18" y="{top + chart_h / 2}" font-size="13" font-family="Arial" transform="rotate(-90 18 {top + chart_h / 2})">PC2</text>
<g font-family="Arial">{''.join(dots)}{''.join(legend)}</g>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    features, meta, phases = load_dataset()
    all_cols = feature_columns_for(SENSORS, features.columns.tolist())
    y = encode_labels(features["label"].to_numpy())
    dates = features["collection_date"].to_numpy()

    random_splits = list(stratified_splits(y))
    date_splits = list(leave_group_out_splits(dates, y))
    nb_train, nb_test = new_bottle_split(dates)
    new_bottle_splits = [(0, nb_train, nb_test, "2026-06-04_2026-06-05")]

    ab_random, cm_random = aggregate_ablation(features, random_splits, all_cols, "stratified_random_holdout")
    ab_date, cm_date = aggregate_ablation(features, date_splits, all_cols, "leave_collection_date_out")
    ab_nb, cm_nb = aggregate_ablation(features, new_bottle_splits, all_cols, "new_bottle_holdout")
    ablation = pd.concat([ab_random, ab_date, ab_nb], ignore_index=True)
    confusion = pd.concat([cm_random, cm_date, cm_nb], ignore_index=True)

    single_random = evaluate_single_sensors(features, random_splits, all_cols, "stratified_random_holdout")
    single_date = evaluate_single_sensors(features, date_splits, all_cols, "leave_collection_date_out")
    single_nb = evaluate_single_sensors(features, new_bottle_splits, all_cols, "new_bottle_holdout")
    single = pd.concat([single_random, single_date, single_nb], ignore_index=True)

    triplet_random = evaluate_sensor_triplets(features, random_splits, all_cols, "stratified_random_holdout")
    triplet_date = evaluate_sensor_triplets(features, date_splits, all_cols, "leave_collection_date_out")
    triplet_nb = evaluate_sensor_triplets(features, new_bottle_splits, all_cols, "new_bottle_holdout")
    triplet = pd.concat([triplet_random, triplet_date, triplet_nb], ignore_index=True)

    fisher = fisher_scores(features, all_cols)
    response = sensor_response_summary(features, all_cols)
    pca_score_df, pca_loading_df = pca_scores(features, all_cols)

    random_summary = summarize_ablation(ablation, "stratified_random_holdout")
    date_summary = summarize_ablation(ablation, "leave_collection_date_out")
    nb_summary = summarize_ablation(ablation, "new_bottle_holdout")
    single_summary = single.groupby(["mode", "classifier", "sensor"], as_index=False).agg(
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_macro_f1=("macro_f1", "mean"),
        std_balanced_accuracy=("balanced_accuracy", "std"),
    )
    triplet_summary = triplet.groupby(["mode", "classifier", "sensors"], as_index=False).agg(
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_macro_f1=("macro_f1", "mean"),
        std_balanced_accuracy=("balanced_accuracy", "std"),
    ).sort_values(["mode", "classifier", "mean_balanced_accuracy"], ascending=[True, True, False])

    sensor_summary = (
        random_summary[random_summary["classifier"] == "lda"]
        .rename(columns={
            "mean_delta_balanced_accuracy": "random_holdout_lda_delta_bacc",
            "positive_delta_rate": "random_positive_delta_rate",
        })[["removed_sensor", "random_holdout_lda_delta_bacc", "random_positive_delta_rate"]]
        .merge(
            date_summary[date_summary["classifier"] == "lda"]
            .rename(columns={"mean_delta_balanced_accuracy": "leave_date_lda_delta_bacc"})[
                ["removed_sensor", "leave_date_lda_delta_bacc"]
            ],
            on="removed_sensor",
            how="left",
        )
        .merge(
            nb_summary[nb_summary["classifier"] == "lda"]
            .rename(columns={"mean_delta_balanced_accuracy": "new_bottle_lda_delta_bacc"})[
                ["removed_sensor", "new_bottle_lda_delta_bacc"]
            ],
            on="removed_sensor",
            how="left",
        )
        .merge(
            single_summary[(single_summary["mode"] == "stratified_random_holdout") & (single_summary["classifier"] == "lda")]
            .rename(columns={"sensor": "removed_sensor", "mean_balanced_accuracy": "single_sensor_lda_bacc"})[
                ["removed_sensor", "single_sensor_lda_bacc"]
            ],
            on="removed_sensor",
            how="left",
        )
        .merge(
            fisher.rename(columns={"sensor": "removed_sensor"}),
            on="removed_sensor",
            how="left",
        )
        .merge(
            pca_loading_df.rename(columns={"sensor": "removed_sensor"}),
            on="removed_sensor",
            how="left",
        )
        .sort_values("random_holdout_lda_delta_bacc", ascending=False)
    )

    features.to_csv(PROCESSED_DIR / "sample_features_baseline_corrected.csv", index=False)
    meta.to_csv(PROCESSED_DIR / "sample_metadata.csv", index=False)
    phases.to_csv(PROCESSED_DIR / "phase_windows_used.csv", index=False)
    ablation.to_csv(TABLE_DIR / "sensor_ablation_all_results.csv", index=False)
    random_summary.to_csv(TABLE_DIR / "sensor_ablation_random_holdout_summary.csv", index=False)
    date_summary.to_csv(TABLE_DIR / "sensor_ablation_leave_date_out_summary.csv", index=False)
    nb_summary.to_csv(TABLE_DIR / "sensor_ablation_new_bottle_summary.csv", index=False)
    single_summary.to_csv(TABLE_DIR / "single_sensor_summary.csv", index=False)
    triplet_summary.to_csv(TABLE_DIR / "sensor_triplet_summary.csv", index=False)
    sensor_summary.to_csv(TABLE_DIR / "sensor_importance_summary.csv", index=False)
    response.to_csv(TABLE_DIR / "sensor_response_summary.csv", index=False)
    confusion.to_csv(TABLE_DIR / "sensor_ablation_confusions.csv", index=False)
    pca_score_df.to_csv(PROCESSED_DIR / "pca_3d_feature_space_scores.csv", index=False)
    pca_loading_df.to_csv(TABLE_DIR / "pca_sensor_contributions.csv", index=False)
    svg_bar(sensor_summary.rename(columns={"removed_sensor": "removed_sensor", "random_holdout_lda_delta_bacc": "mean_delta_balanced_accuracy"}).assign(classifier="lda"), FIG_DIR / "sensor_ablation_importance.svg")
    svg_pca(pca_score_df, FIG_DIR / "pca_feature_space_pc1_pc2.svg")

    overview = {
        "n_samples": len(features),
        "class_counts": dict(Counter(features["label"])),
        "duration_counts": dict(Counter(features["duration_s"])),
        "date_counts": dict(Counter(features["collection_date"])),
        "n_random_splits": len(random_splits),
        "n_leave_date_splits": len(date_splits),
        "new_bottle_test_counts": dict(Counter(features.iloc[nb_test]["label"])),
    }
    with (TABLE_DIR / "analysis_overview.txt").open("w", encoding="utf-8") as f:
        for key, value in overview.items():
            f.write(f"{key}: {value}\n")

    print("Wrote analysis outputs:")
    print(TABLE_DIR / "sensor_importance_summary.csv")
    print(TABLE_DIR / "sensor_ablation_random_holdout_summary.csv")
    print(TABLE_DIR / "sensor_ablation_leave_date_out_summary.csv")
    print(TABLE_DIR / "sensor_ablation_new_bottle_summary.csv")
    print(PROCESSED_DIR / "pca_3d_feature_space_scores.csv")
    print(FIG_DIR / "sensor_ablation_importance.svg")
    print(FIG_DIR / "pca_feature_space_pc1_pc2.svg")


if __name__ == "__main__":
    main()
