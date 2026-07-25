from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path(r"D:\thesis\enose_data")
RECORD_PATH = Path(r"C:\Users\yaoli\Desktop\record_data.xlsx")
OUT_ROOT = Path(r"D:\thesis")
FIG_ROOT = OUT_ROOT / "figures"
TABLE_ROOT = OUT_ROOT / "tables"

RAW_DIR = FIG_ROOT / "feature_space_raw"
RAW_CUT_DIR = FIG_ROOT / "feature_space_raw_cut"
UCI_DIR = FIG_ROOT / "feature_space_uci"

SENSORS = [f"s{i}" for i in range(1, 7)]
GASES = ["air", "alcohol", "acetone"]
COLORS = {"air": "#2d72b2", "alcohol": "#52a046", "acetone": "#c35359"}
ALPHAS = [0.1, 0.01, 0.001]
BATCH_LABELS = ["source", "batch1", "batch2", "batch3", "batch4", "batch5", "unassigned"]

RAW_AXES = [
    ("s1_s4_s5", ["s1", "s4", "s5"]),
    ("s2_s4_s5", ["s2", "s4", "s5"]),
]

UCI_SPACES = [
    ("s1_s4_s5", ["s1", "s4", "s5"]),
    ("s2_s4_s5", ["s2", "s4", "s5"]),
]

FEATURE_FAMILIES = [
    "deltaR_{sensor}",
    "normDeltaR_{sensor}",
    "riseEMA_a0.1_{sensor}",
    "riseEMA_a0.01_{sensor}",
    "riseEMA_a0.001_{sensor}",
    "recEMA_a0.1_{sensor}",
    "recEMA_a0.01_{sensor}",
    "recEMA_a0.001_{sensor}",
]


def number_from_name(text: str) -> int:
    match = re.search(r"(\d+)", str(text))
    return int(match.group(1)) if match else -1


def day_sort_key(value: str) -> tuple[int, int, str]:
    text = str(value)
    if text.startswith("Period I") or text == "source":
        period = 1
    elif "batch1" in text:
        period = 2
    elif "batch2" in text:
        period = 3
    elif "batch3" in text:
        period = 4
    elif "batch4" in text:
        period = 5
    elif "batch5" in text:
        period = 6
    else:
        period = 9
    match = re.search(r"Day\s*(\d+)", text)
    day = int(match.group(1)) if match else 0
    return period, day, text


def normalize_period(text: str) -> str:
    clean = str(text).strip().replace(" ", "_")
    return clean if clean.startswith("Period_") else str(text).strip()


def batch_for_day(day_num: int, period: str) -> str:
    if period == "Period_I":
        return "source"
    if 1 <= day_num <= 2:
        return "batch1"
    if 3 <= day_num <= 9:
        return "batch2"
    if 10 <= day_num <= 17:
        return "batch3"
    if 18 <= day_num <= 21:
        return "batch4"
    if 22 <= day_num <= 24:
        return "batch5"
    return "unassigned"


def parse_day_heading(text: str) -> tuple[int, str] | None:
    match = re.match(r"^Day\s*(\d+)(.*)$", str(text).strip(), flags=re.IGNORECASE)
    if not match:
        return None
    day_num = int(match.group(1))
    suffix = match.group(2).strip("_ -")
    label = f"Day {day_num}" + (f" {suffix}" if suffix else "")
    return day_num, label


def parse_record_book() -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    period_i: dict[str, dict[str, object]] = {}
    target: dict[str, dict[str, object]] = {}
    if not RECORD_PATH.exists():
        return period_i, target
    sheet = pd.read_excel(RECORD_PATH)
    current_period = ""
    current_day_num = -1
    current_day_label = ""
    current_date = ""
    trial_re = re.compile(r"^(alcohol|acetone|air|reference|baseline)_(\d+)$", re.IGNORECASE)
    for _, row in sheet.iterrows():
        value = row.get("Gas")
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        if text.lower().startswith("period"):
            current_period = normalize_period(text)
            current_day_num = -1
            current_day_label = ""
            current_date = ""
            continue
        day = parse_day_heading(text)
        if day is not None:
            current_day_num, current_day_label = day
            time_value = row.get("Time")
            current_date = "" if pd.isna(time_value) else str(time_value).strip()
            continue
        if not trial_re.match(text):
            continue
        meta = {
            "period": current_period,
            "day": current_day_label,
            "day_num": current_day_num,
            "day_label": f"{current_period.replace('_', ' ')} {current_day_label}".strip(),
            "record_date": current_date,
            "batch": batch_for_day(current_day_num, current_period),
        }
        if current_period == "Period_I":
            period_i[text.lower()] = meta
        else:
            target[text.lower()] = meta
    return period_i, target


def file_metadata(path: Path, period_i_map: dict[str, dict[str, object]], target_map: dict[str, dict[str, object]]) -> dict[str, object] | None:
    name = path.name
    stem = path.stem.lower()
    folder = path.parent.name.lower()
    source_kind = "new"
    lookup_key = stem
    if folder == "alcohol" and re.match(r"^alcohol_o\d+$", stem):
        lookup_key = stem.replace("alcohol_o", "alcohol_")
        label = "alcohol"
        source_kind = "old"
        meta = dict(period_i_map.get(lookup_key, {}))
    elif folder == "acetone" and re.match(r"^acetone_o\d+$", stem):
        lookup_key = stem.replace("acetone_o", "acetone_")
        label = "acetone"
        source_kind = "old"
        meta = dict(period_i_map.get(lookup_key, {}))
    elif folder == "alcohol" and re.match(r"^alcohol_\d+$", stem):
        label = "alcohol"
        meta = dict(target_map.get(lookup_key, {}))
    elif folder == "acetone" and re.match(r"^acetone_\d+$", stem):
        label = "acetone"
        meta = dict(target_map.get(lookup_key, {}))
    elif folder == "air" and re.match(r"^air_\d+$", stem):
        label = "air"
        meta = dict(target_map.get(lookup_key, {}))
    elif folder == "air" and re.match(r"^reference_\d+$", stem):
        label = "air"
        source_kind = "reference_as_air"
        meta = dict(target_map.get(lookup_key, {}))
    elif folder == "air" and re.match(r"^baseline_\d+$", stem):
        label = "air"
        source_kind = "baseline_as_air"
        meta = {
            "period": "Period_I",
            "day": "baseline",
            "day_num": 0,
            "day_label": "Period I baseline",
            "record_date": "",
            "batch": "source",
        }
    else:
        return None
    period = str(meta.get("period", ""))
    day_num = int(meta.get("day_num", -1)) if str(meta.get("day_num", "")).strip() else -1
    batch = str(meta.get("batch", batch_for_day(day_num, period)))
    return {
        "sample_id": path.stem,
        "label": label,
        "period": period,
        "day": meta.get("day", ""),
        "day_num": day_num,
        "day_label": meta.get("day_label", "Unknown"),
        "record_date": meta.get("record_date", ""),
        "batch": batch,
        "source_kind": source_kind,
        "target_file": str(path.relative_to(DATA_ROOT)),
        "path": path,
    }


def elapsed_seconds(df: pd.DataFrame) -> np.ndarray:
    if "arduino_time" in df.columns:
        t = pd.to_numeric(df["arduino_time"], errors="coerce").to_numpy(float)
        if len(t) and np.isfinite(t[0]):
            elapsed = (t - t[0]) / 1000.0
            if np.nanmax(elapsed) > 5:
                return elapsed
    if "time_s" in df.columns:
        t = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(float)
        return t - t[0]
    return np.arange(len(df), dtype=float)


def phase_masks(label: str, duration: float, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = float(np.nanmax(t)) if len(t) else duration
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
    if baseline.sum() < 3:
        baseline = np.arange(len(t)) < max(3, int(len(t) * 0.2))
    if exposure.sum() < 3:
        start = max(0, int(len(t) * 0.45))
        stop = min(len(t), max(start + 3, int(len(t) * 0.65)))
        exposure = np.zeros(len(t), dtype=bool)
        exposure[start:stop] = True
    if recovery.sum() < 3:
        start = max(0, int(len(t) * 0.75))
        recovery = np.zeros(len(t), dtype=bool)
        recovery[start:] = True
    return baseline, exposure, recovery


def ema_diff(values: np.ndarray, alpha: float, mode: str) -> float:
    clean = values[np.isfinite(values)]
    if clean.size < 2:
        return float("nan")
    y = 0.0
    out = []
    for d in np.diff(clean):
        y = (1.0 - alpha) * y + alpha * d
        out.append(y)
    arr = np.asarray(out, dtype=float)
    return float(np.nanmax(arr) if mode == "max" else np.nanmin(arr))


def load_trials() -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    period_i_map, target_map = parse_record_book()
    manifest_rows = []
    for gas_dir in GASES:
        folder = DATA_ROOT / gas_dir
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.csv"), key=lambda p: (p.parent.name, number_from_name(p.name), p.name)):
            meta = file_metadata(path, period_i_map, target_map)
            if meta is not None:
                manifest_rows.append(meta)
    manifest = pd.DataFrame(manifest_rows)
    records = []
    corrected_by_sample: dict[str, dict[str, np.ndarray]] = {}
    absolute_by_sample: dict[str, dict[str, np.ndarray]] = {}
    for _, row in manifest.iterrows():
        sid = str(row["sample_id"])
        label = str(row["label"])
        path = Path(row["path"])
        if not path.exists():
            continue
        df = pd.read_csv(path)
        t = elapsed_seconds(df)
        duration = float(df["duration_s"].iloc[0]) if "duration_s" in df.columns else float(np.nanmax(t))
        bmask, emask, rmask = phase_masks(label, duration, t)
        corrected: dict[str, np.ndarray] = {}
        absolute: dict[str, np.ndarray] = {}
        for sensor in SENSORS:
            x = pd.to_numeric(df[sensor], errors="coerce").to_numpy(float)
            b0 = float(np.nanmedian(x[bmask]))
            absolute[sensor] = x - b0
            corrected[sensor] = (x - b0) / max(abs(b0), 1e-9)
        corrected_by_sample[sid] = corrected
        absolute_by_sample[sid] = absolute
        records.append(
            {
                "sample_id": sid,
                "label": label,
                "period": row.get("period", ""),
                "day": row.get("day", ""),
                "day_num": int(float(row.get("day_num", -1))) if str(row.get("day_num", "")).strip() else -1,
                "day_label": row.get("day_label", ""),
                "record_date": row.get("record_date", ""),
                "batch": row.get("batch", "unassigned"),
                "source_kind": row.get("source_kind", ""),
                "target_file": row.get("target_file", ""),
                "duration_s": duration,
                "t": t,
                "baseline_mask": bmask,
                "exposure_mask": emask,
                "recovery_mask": rmask,
            }
        )
    return pd.DataFrame(records), corrected_by_sample, absolute_by_sample


def finite_rows(arr: np.ndarray) -> np.ndarray:
    return np.isfinite(arr).all(axis=1)


def scaled_points(values: np.ndarray, mode: str) -> tuple[np.ndarray, pd.DataFrame]:
    if mode == "clip":
        low = np.nanpercentile(values, 1, axis=0)
        high = np.nanpercentile(values, 99, axis=0)
        clipped = np.clip(values, low, high)
        display_min = low
        display_max = high
        scaled_source = clipped
        method = "1st-99th percentile clipping, then min/max scaling"
    else:
        raw_min = np.nanmin(values, axis=0)
        raw_max = np.nanmax(values, axis=0)
        span0 = np.where(raw_max - raw_min == 0, 1.0, raw_max - raw_min)
        padding = span0 * 0.03
        display_min = raw_min - padding
        display_max = raw_max + padding
        scaled_source = values
        method = "full min/max scaling with 3% padding; no clipping"
    span = np.where(display_max - display_min == 0, 1.0, display_max - display_min)
    scaled = (scaled_source - display_min) / span
    scale = pd.DataFrame(
        {
            "axis": ["x", "y", "z"],
            "display_min": display_min,
            "display_max": display_max,
            "method": method,
        }
    )
    return scaled, scale


def apply_scale(values: np.ndarray, scale: pd.DataFrame) -> np.ndarray:
    low = scale["display_min"].to_numpy(float)
    high = scale["display_max"].to_numpy(float)
    span = np.where(high - low == 0, 1.0, high - low)
    return (np.nan_to_num(values) - low) / span


def scale_payload(scale: pd.DataFrame) -> list[dict[str, float]]:
    return [
        {
            "axis": str(row.axis),
            "displayMin": float(row.display_min),
            "displayMax": float(row.display_max),
        }
        for row in scale.itertuples(index=False)
    ]


def zero_started_series(corrected_by_sample: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    zeroed: dict[str, dict[str, np.ndarray]] = {}
    for sample, sensors in corrected_by_sample.items():
        zeroed[sample] = {}
        for sensor, values in sensors.items():
            arr = np.asarray(values, dtype=float)
            finite = np.flatnonzero(np.isfinite(arr))
            start = float(arr[finite[0]]) if len(finite) else 0.0
            zeroed[sample][sensor] = arr - start
    return zeroed


def resample_indices(t: np.ndarray, max_points: int = 181) -> np.ndarray:
    if len(t) <= max_points:
        return np.arange(len(t))
    return np.linspace(0, len(t) - 1, max_points).astype(int)


def raw_trajectory_payload(
    trials: pd.DataFrame,
    corrected_by_sample: dict[str, dict[str, np.ndarray]],
    axes: list[str],
    scale: pd.DataFrame,
) -> dict[str, list[list[float]]]:
    trajectories: dict[str, list[list[float]]] = {}
    for row in trials.itertuples(index=False):
        values = np.column_stack([corrected_by_sample[row.sample_id][a] for a in axes])
        idx = resample_indices(row.t)
        scaled = apply_scale(values[idx], scale)
        t = np.nan_to_num(row.t[idx], nan=0.0)
        trajectories[row.sample_id] = [
            [round(float(ti), 3), round(float(x), 5), round(float(y), 5), round(float(z), 5)]
            for ti, (x, y, z) in zip(t, scaled)
            if 0.0 <= float(ti) <= 180.0
        ]
    return trajectories


def uci_trajectory_payload(
    trials: pd.DataFrame,
    corrected_by_sample: dict[str, dict[str, np.ndarray]],
    axes: list[str],
    scale: pd.DataFrame,
) -> dict[str, list[list[float]]]:
    trajectories: dict[str, list[list[float]]] = {}
    for row in trials.itertuples(index=False):
        series = np.column_stack([corrected_by_sample[row.sample_id][a] for a in axes])
        idx = resample_indices(row.t)
        cumulative = []
        for i in idx:
            prefix = series[: max(1, i + 1)]
            cumulative.append(np.nanmax(prefix, axis=0) - np.nanmin(prefix, axis=0))
        scaled = apply_scale(np.asarray(cumulative), scale)
        t = np.nan_to_num(row.t[idx], nan=0.0)
        trajectories[row.sample_id] = [
            [round(float(ti), 3), round(float(x), 5), round(float(y), 5), round(float(z), 5)]
            for ti, (x, y, z) in zip(t, scaled)
            if 0.0 <= float(ti) <= 180.0
        ]
    return trajectories


def sample_options(trials: pd.DataFrame) -> list[dict[str, object]]:
    cols = ["sample_id", "label", "day_label", "batch", "target_file"]
    return [
        {k: ("" if pd.isna(v) else v) for k, v in row.items()}
        for row in trials.sort_values(["batch", "label", "period", "day_num", "sample_id"])[cols].to_dict("records")
    ]


def domain_options(trials: pd.DataFrame) -> list[str]:
    available = set(str(v) for v in trials["batch"].dropna().unique())
    ordered = [label for label in BATCH_LABELS if label in available]
    extras = sorted(available.difference(ordered))
    return ordered + extras


def fit_lda(coords: np.ndarray, labels: np.ndarray, sample_ids: np.ndarray) -> dict[str, object]:
    mask = np.isin(labels, ["alcohol", "acetone"])
    x = coords[mask]
    y = labels[mask]
    ids = sample_ids[mask]
    x0 = x[y == "alcohol"]
    x1 = x[y == "acetone"]
    if len(x0) < 2 or len(x1) < 2:
        return {"available": False, "predictions": pd.DataFrame()}
    mu0 = x0.mean(axis=0)
    mu1 = x1.mean(axis=0)
    scatter = (x0 - mu0).T @ (x0 - mu0) + (x1 - mu1).T @ (x1 - mu1)
    pooled = scatter / max(1, len(x0) + len(x1) - 2)
    reg = 1e-3 * np.trace(pooled) / max(1, pooled.shape[0])
    w = np.linalg.solve(pooled + reg * np.eye(pooled.shape[0]), mu1 - mu0)
    b = -0.5 * float(w @ (mu0 + mu1)) + math.log(len(x1) / len(x0))
    scores = x @ w + b
    pred = np.where(scores >= 0, "acetone", "alcohol")
    alcohol_total = int((y == "alcohol").sum())
    acetone_total = int((y == "acetone").sum())
    alcohol_correct = int(((y == "alcohol") & (pred == "alcohol")).sum())
    acetone_correct = int(((y == "acetone") & (pred == "acetone")).sum())
    alcohol_recall = alcohol_correct / alcohol_total if alcohol_total else 0.0
    acetone_recall = acetone_correct / acetone_total if acetone_total else 0.0
    return {
        "available": True,
        "w": w.tolist(),
        "b": float(b),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": 0.5 * (alcohol_recall + acetone_recall),
        "alcohol_total": alcohol_total,
        "acetone_total": acetone_total,
        "alcohol_correct": alcohol_correct,
        "acetone_correct": acetone_correct,
        "alcohol_recall": alcohol_recall,
        "acetone_recall": acetone_recall,
        "alcohol_as_acetone": alcohol_total - alcohol_correct,
        "acetone_as_alcohol": acetone_total - acetone_correct,
        "predictions": pd.DataFrame(
            {
                "sample_id": ids,
                "label": y,
                "score": scores,
                "predicted_label": pred,
                "correct": pred == y,
            }
        ),
    }


def plane_cells(w: list[float], b: float, n: int = 22) -> list[list[list[float]]]:
    normal = np.asarray(w, dtype=float)
    solve_axis = int(np.argmax(np.abs(normal)))
    axes = [0, 1, 2]
    axes.remove(solve_axis)
    vals = np.linspace(0, 1, n + 1)
    cells = []
    for i in range(n):
        for j in range(n):
            quad = []
            ok = True
            for a, c in [(vals[i], vals[j]), (vals[i + 1], vals[j]), (vals[i + 1], vals[j + 1]), (vals[i], vals[j + 1])]:
                p = [0.0, 0.0, 0.0]
                p[axes[0]] = float(a)
                p[axes[1]] = float(c)
                p[solve_axis] = float(-(b + normal[axes[0]] * a + normal[axes[1]] * c) / normal[solve_axis])
                if p[solve_axis] < 0 or p[solve_axis] > 1:
                    ok = False
                    break
                quad.append([round(v, 5) for v in p])
            if ok:
                cells.append(quad)
    return cells


def pca_3d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std < 1e-9] = 1.0
    z = np.nan_to_num((x - mean) / std)
    _, s, vt = np.linalg.svd(z, full_matrices=False)
    coords = z @ vt[:3].T
    explained = (s**2) / max(1, z.shape[0] - 1)
    ratio = explained / explained.sum()
    return coords, vt[:3], ratio[:3]


def sensor_feature_columns(sensors: list[str]) -> list[str]:
    return [pattern.format(sensor=sensor) for sensor in sensors for pattern in FEATURE_FAMILIES]


def make_uci_features(trials: pd.DataFrame, corrected_by_sample: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for row in trials.itertuples(index=False):
        corr = corrected_by_sample[row.sample_id]
        out = {
            "sample_id": row.sample_id,
            "label": row.label,
            "day_label": row.day_label,
            "day_num": row.day_num,
            "period": row.period,
            "batch": row.batch,
            "source_kind": row.source_kind,
        }
        emask = row.exposure_mask
        rmask = row.recovery_mask
        for sensor in SENSORS:
            z = corr[sensor]
            exp = z[emask]
            rec = z[rmask]
            delta = float(np.nanmax(exp) - np.nanmin(exp))
            out[f"deltaR_{sensor}"] = delta
            out[f"normDeltaR_{sensor}"] = delta
            for alpha in ALPHAS:
                a = f"{alpha:g}"
                out[f"riseEMA_a{a}_{sensor}"] = ema_diff(exp, alpha, "max")
                out[f"recEMA_a{a}_{sensor}"] = ema_diff(rec, alpha, "min")
        rows.append(out)
    return pd.DataFrame(rows)


def raw_point_payload(
    trials: pd.DataFrame,
    corrected_by_sample: dict[str, dict[str, np.ndarray]],
    axes: list[str],
    title: str,
    mode: str,
    axis_label_by_sample: dict[str, dict[str, np.ndarray]] | None = None,
    max_points_per_sample: int = 450,
) -> tuple[dict, pd.DataFrame]:
    all_values = []
    all_axis_label_values = []
    metas = []
    for row in trials.itertuples(index=False):
        corr = corrected_by_sample[row.sample_id]
        values = np.column_stack([corr[a] for a in axes])
        ok = finite_rows(values)
        values = values[ok]
        label_values = None
        if axis_label_by_sample is not None:
            label_values = np.column_stack([axis_label_by_sample[row.sample_id][a] for a in axes])[ok]
        if len(values) > max_points_per_sample:
            idx = np.linspace(0, len(values) - 1, max_points_per_sample).astype(int)
            values = values[idx]
            if label_values is not None:
                label_values = label_values[idx]
        all_values.append(values)
        if label_values is not None:
            all_axis_label_values.append(label_values)
        metas.extend([(row.sample_id, row.label, row.day_label, row.batch)] * len(values))
    values = np.vstack(all_values)
    scaled, scale = scaled_points(values, mode)
    axis_label_scale = scale
    if all_axis_label_values:
        _, axis_label_scale = scaled_points(np.vstack(all_axis_label_values), mode)
    points = [
        [round(float(x), 5), round(float(y), 5), round(float(z), 5), GASES.index(label), sample, day, "", 0, 0.0, batch]
        for (x, y, z), (sample, label, day, batch) in zip(scaled, metas)
    ]
    counts = trials["label"].value_counts().reindex(GASES, fill_value=0).to_dict()
    payload = {
        "title": title,
        "axes": [a.upper() for a in axes],
        "points": points,
        "samples": sample_options(trials),
        "trajectories": raw_trajectory_payload(trials, corrected_by_sample, axes, scale),
        "trajectoryMode": "baseline-corrected sensor trajectory",
        "scale": scale_payload(axis_label_scale),
        "gases": GASES,
        "colors": COLORS,
        "counts": counts,
        "days": sorted(trials["day_label"].unique(), key=day_sort_key),
        "domains": domain_options(trials),
        "mode": mode,
        "note": "baseline/reference are treated as air for color; source=Period I, target batches follow Day1-2, Day3-9, Day10-17, Day18-21, Day22-24",
    }
    if axis_label_by_sample is not None:
        payload["note"] += "; point positions are unchanged, axis tick labels show absolute baseline-corrected values"
    return payload, axis_label_scale


def group_payload(
    trials: pd.DataFrame,
    corrected_by_sample: dict[str, dict[str, np.ndarray]],
    axes: list[str],
    title: str,
    mode: str,
    axis_label_by_sample: dict[str, dict[str, np.ndarray]] | None = None,
) -> tuple[dict, pd.DataFrame, dict[str, object]]:
    rows = []
    axis_label_rows = []
    for row in trials.itertuples(index=False):
        corr = corrected_by_sample[row.sample_id]
        mask = row.exposure_mask | row.recovery_mask
        vals = [float(np.nanmean(corr[a][mask])) for a in axes]
        if axis_label_by_sample is not None:
            axis_corr = axis_label_by_sample[row.sample_id]
            axis_vals = [float(np.nanmean(axis_corr[a][mask])) for a in axes]
            axis_label_rows.append(axis_vals)
        rows.append(
            {
                "sample_id": row.sample_id,
                "label": row.label,
                "day_label": row.day_label,
                "batch": row.batch,
                "x": vals[0],
                "y": vals[1],
                "z": vals[2],
            }
        )
    df = pd.DataFrame(rows)
    values = df[["x", "y", "z"]].to_numpy(float)
    scaled, scale = scaled_points(values, mode)
    axis_label_scale = scale
    if axis_label_rows:
        _, axis_label_scale = scaled_points(np.asarray(axis_label_rows, dtype=float), mode)
    hp = fit_lda(scaled, df["label"].to_numpy(str), df["sample_id"].to_numpy(str))
    pred_map = {}
    if hp.get("available"):
        pred_map = {
            r.sample_id: (r.predicted_label, bool(r.correct), float(r.score))
            for r in hp["predictions"].itertuples(index=False)
        }
    points = []
    for i, row in enumerate(df.itertuples(index=False)):
        pred, correct, score = pred_map.get(row.sample_id, ("", True, 0.0))
        points.append(
            [
                round(float(scaled[i, 0]), 5),
                round(float(scaled[i, 1]), 5),
                round(float(scaled[i, 2]), 5),
                GASES.index(row.label),
                row.sample_id,
                row.day_label,
                pred,
                0 if correct else 1,
                round(score, 5),
                row.batch,
            ]
        )
    payload = {
        "title": title,
        "axes": [a.upper() for a in axes],
        "points": points,
        "samples": sample_options(trials),
        "trajectories": raw_trajectory_payload(trials, corrected_by_sample, axes, scale),
        "trajectoryMode": "baseline-corrected sensor trajectory",
        "scale": scale_payload(axis_label_scale),
        "gases": GASES,
        "colors": COLORS,
        "counts": trials["label"].value_counts().reindex(GASES, fill_value=0).to_dict(),
        "days": sorted(trials["day_label"].unique(), key=day_sort_key),
        "domains": domain_options(trials),
        "mode": mode,
        "note": "one point per sample using post-baseline exposure/recovery mean; baseline/reference are treated as air",
        "plane": {
            "available": bool(hp.get("available")),
            "w": hp.get("w", []),
            "b": hp.get("b", 0.0),
            "cells": plane_cells(hp["w"], float(hp["b"])) if hp.get("available") else [],
            "accuracy": hp.get("accuracy", None),
            "balancedAccuracy": hp.get("balanced_accuracy", None),
            "alcoholCorrect": hp.get("alcohol_correct", 0),
            "alcoholTotal": hp.get("alcohol_total", 0),
            "acetoneCorrect": hp.get("acetone_correct", 0),
            "acetoneTotal": hp.get("acetone_total", 0),
        },
    }
    if axis_label_by_sample is not None:
        payload["note"] += "; point positions are unchanged, axis tick labels show absolute baseline-corrected values"
    return payload, axis_label_scale, hp


def uci_payload(
    trials: pd.DataFrame,
    corrected_by_sample: dict[str, dict[str, np.ndarray]],
    features: pd.DataFrame,
    sensors: list[str],
    title: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    cols = [f"normDeltaR_{sensor}" for sensor in sensors]
    coords = features[cols].to_numpy(float)
    scaled, scale = scaled_points(coords, "no_clip")
    hp = fit_lda(scaled, features["label"].to_numpy(str), features["sample_id"].to_numpy(str))
    pred_map = {}
    if hp.get("available"):
        pred_map = {
            r.sample_id: (r.predicted_label, bool(r.correct), float(r.score))
            for r in hp["predictions"].itertuples(index=False)
        }
    points = []
    for i, row in enumerate(features.itertuples(index=False)):
        pred, correct, score = pred_map.get(row.sample_id, ("", True, 0.0))
        points.append(
            [
                round(float(scaled[i, 0]), 5),
                round(float(scaled[i, 1]), 5),
                round(float(scaled[i, 2]), 5),
                GASES.index(row.label),
                row.sample_id,
                row.day_label,
                pred,
                0 if correct else 1,
                round(score, 5),
                row.batch,
            ]
        )
    loading_rows = [
        {"space": "_".join(sensors), "axis": axis, "feature": col}
        for axis, col in zip(["x", "y", "z"], cols)
    ]
    payload = {
        "title": title,
        "axes": [f"normDeltaR {s.upper()}" for s in sensors],
        "points": points,
        "samples": sample_options(trials),
        "trajectories": uci_trajectory_payload(trials, corrected_by_sample, sensors, scale),
        "trajectoryMode": "progressive cumulative normDeltaR trajectory",
        "scale": scale_payload(scale),
        "gases": GASES,
        "colors": COLORS,
        "counts": features["label"].value_counts().reindex(GASES, fill_value=0).to_dict(),
        "days": sorted(features["day_label"].unique(), key=day_sort_key),
        "domains": domain_options(trials),
        "mode": "uci_norm_delta",
        "note": f"UCI-style normDeltaR feature space on sensors {'/'.join(s.upper() for s in sensors)}; animation shows progressive cumulative normDeltaR over 0-180s",
        "plane": {
            "available": bool(hp.get("available")),
            "w": hp.get("w", []),
            "b": hp.get("b", 0.0),
            "cells": plane_cells(hp["w"], float(hp["b"])) if hp.get("available") else [],
            "accuracy": hp.get("accuracy", None),
            "balancedAccuracy": hp.get("balanced_accuracy", None),
            "alcoholCorrect": hp.get("alcohol_correct", 0),
            "alcoholTotal": hp.get("alcohol_total", 0),
            "acetoneCorrect": hp.get("acetone_correct", 0),
            "acetoneTotal": hp.get("acetone_total", 0),
        },
    }
    return payload, scale, pd.DataFrame(loading_rows), hp


def write_html(path: Path, payload: dict) -> None:
    data_json = json.dumps(payload, separators=(",", ":"))
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{payload["title"]}</title>
<style>
html,body{{margin:0;height:100%;overflow:hidden;background:#fff;color:#22272e;font-family:Arial,sans-serif}}
#wrap{{position:fixed;inset:0;display:grid;grid-template-rows:auto 1fr}}
header{{padding:15px 22px 9px;border-bottom:1px solid #e6e8eb;background:#fff}}
h1{{margin:0;font-size:21px;font-weight:700;letter-spacing:0}}
p{{margin:5px 0 0;color:#5d6673;font-size:13px}}
#stage{{position:relative;min-height:0;background:#fafbfc}}
canvas{{position:absolute;inset:0;width:100%;height:100%;cursor:grab}}
canvas:active{{cursor:grabbing}}
#legend,#panel{{position:absolute;background:rgba(255,255,255,.93);border:1px solid #e2e8f0;border-radius:8px;box-shadow:0 8px 24px rgba(15,23,42,.08);font-size:13px}}
#legend{{left:18px;top:18px;padding:12px 14px}}
#panel{{right:18px;top:18px;width:340px;padding:13px;max-height:calc(100% - 70px);overflow:auto}}
.row{{display:flex;gap:8px;align-items:center;margin:5px 0}}
.dot{{width:10px;height:10px;border-radius:50%;display:inline-block}}
.section{{border-top:1px solid #e2e8f0;margin-top:10px;padding-top:10px}}
label{{display:block;font-size:12px;font-weight:700;margin:8px 0 4px;color:#44515f}}
select,input{{width:100%;box-sizing:border-box;border:1px solid #cbd5e1;border-radius:6px;padding:7px;background:#fff;font-size:13px}}
button{{border:1px solid #cbd5e1;border-radius:6px;padding:7px 10px;background:#f8fafc;cursor:pointer;font-size:13px;margin-top:8px}}
button:hover{{background:#eef2f7}}
#status,#planeStats{{font-size:12px;line-height:1.4;color:#44515f;margin-top:8px}}
.toggle{{display:flex;align-items:center;gap:8px;margin-top:8px}}
.toggle input{{width:auto}}
#hint{{position:absolute;right:18px;bottom:14px;color:#667085;font-size:12px;background:rgba(255,255,255,.82);padding:7px 9px;border-radius:6px}}
.btnrow{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}}
</style>
</head>
<body>
<div id="wrap">
<header>
<h1>{payload["title"]}</h1>
<p>{payload["note"]}</p>
</header>
<div id="stage">
<canvas id="canvas"></canvas>
<div id="legend"></div>
<div id="panel">
<strong>Feature space controls</strong>
<div id="planeStats"></div>
<div class="toggle"><input id="showPlane" type="checkbox" checked><span>show alcohol/acetone hyperplane when available</span></div>
<div class="toggle"><input id="showPred" type="checkbox"><span>color alcohol/acetone by predicted side</span></div>
<div class="toggle"><input id="showErrors" type="checkbox" checked><span>emphasize wrong alcohol/acetone groups</span></div>
<div class="section">
<label for="domainSelect">View source / target batch</label><select id="domainSelect"></select>
<div class="btnrow"><button id="domainBtn">View selected</button><button id="allDomainBtn">All domains</button><button id="domainStartBtn">Starts only</button></div>
<div id="domainStatus">Showing all source/target batches.</div>
</div>
<div class="section">
<label for="sampleSelect">Select sample trajectory</label><select id="sampleSelect"></select>
<div class="btnrow"><button id="loadSampleBtn">Load</button><button id="playBtn">Play</button><button id="pauseBtn">Pause</button></div>
<div id="trajStatus">No trajectory loaded.</div>
</div>
<div class="section">
<button id="startPointsBtn">Show all trial start points</button>
</div>
<div class="section">
<label for="gasSelect">Highlight gas</label><select id="gasSelect"></select>
<button id="gasBtn">Selected gas, all dates</button>
</div>
<div class="section">
<label for="daySelect">Highlight date/domain</label><select id="daySelect"></select>
<button id="dayBtn">Selected date, all gases</button>
<button id="gasDayBtn">Selected gas on selected date</button>
</div>
<div class="section">
<label for="sampleText">Highlight sample id</label><input id="sampleText" placeholder="e.g. baseline_o1, alcohol_o2">
<button id="sampleBtn">Apply sample highlight</button>
</div>
<div id="status">No highlight selected.</div>
</div>
<div id="hint">Drag to rotate; wheel to zoom; double click to reset</div>
</div>
</div>
<script>
const DATA={data_json};
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
let yaw=-0.66,pitch=0.39,zoom=1.08,dragging=false,lastX=0,lastY=0,pending=false;
let highlight={{mode:'none',day:'',sample:''}};
let viewDomain='';
let activeSample='',activeTrajectory=[],trajFrame=0,playing=false,lastAnim=0;
const gasNames=DATA.gases,gasIndex=Object.fromEntries(DATA.gases.map((g,i)=>[g,i]));
const groupHighlight='#ff8c00',predAlcohol='#1f9d55',predAcetone='#d94841',errorColor='#111827';
function pointDomain(p){{return p.length>9?p[9]:''}}
function sampleVisible(s){{return !viewDomain || s.batch===viewDomain}}
function pointVisible(p){{return !viewDomain || pointDomain(p)===viewDomain}}
function startPoints(){{return (DATA.samples||[]).filter(sampleVisible).map(s=>{{const tr=DATA.trajectories[s.sample_id]||[];if(!tr.length)return null;const p=tr[0];return[p[1],p[2],p[3],gasIndex[s.label]??0,s.sample_id,s.day_label,s.batch]}}).filter(Boolean)}}
function pct(v){{return v==null?'NA':(v*100).toFixed(1)+'%'}}
function requestDraw(){{if(pending)return;pending=true;requestAnimationFrame(()=>{{pending=false;draw()}})}}
function resize(){{const dpr=Math.max(1,window.devicePixelRatio||1),r=canvas.getBoundingClientRect();canvas.width=Math.round(r.width*dpr);canvas.height=Math.round(r.height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);requestDraw()}}
document.getElementById('legend').innerHTML=DATA.gases.map(g=>`<div class="row"><span class="dot" style="background:${{DATA.colors[g]}}"></span><span>${{g}} samples=${{DATA.counts[g]||0}}</span></div>`).join('')+
`<div class="section"><div class="row"><span class="dot" style="background:${{groupHighlight}}"></span><span>highlighted sample / trajectory</span></div><div class="row"><span class="dot" style="background:rgba(9,105,218,.45)"></span><span>hyperplane</span></div><div class="row"><span class="dot" style="background:${{errorColor}}"></span><span>wrong alcohol/acetone</span></div></div>`;
document.getElementById('daySelect').innerHTML=DATA.days.map(d=>`<option value="${{d}}">${{d}}</option>`).join('');
document.getElementById('gasSelect').innerHTML=DATA.gases.map(g=>`<option value="${{g}}">${{g}}</option>`).join('');
document.getElementById('domainSelect').innerHTML=(DATA.domains||[]).map(d=>`<option value="${{d}}">${{d}}</option>`).join('');
function refreshSampleSelect(){{document.getElementById('sampleSelect').innerHTML=DATA.samples.filter(sampleVisible).map(s=>`<option value="${{s.sample_id}}">${{s.sample_id}} | ${{s.label}} | ${{s.batch}} | ${{s.day_label}}</option>`).join('');}}
refreshSampleSelect();
if(DATA.plane&&DATA.plane.available){{
document.getElementById('planeStats').innerHTML=`<div>Axes: <strong>${{DATA.axes.join(' / ')}}</strong></div><div>Plane BAcc: <strong>${{pct(DATA.plane.balancedAccuracy)}}</strong></div><div>Alcohol: ${{DATA.plane.alcoholCorrect}}/${{DATA.plane.alcoholTotal}}; Acetone: ${{DATA.plane.acetoneCorrect}}/${{DATA.plane.acetoneTotal}}</div>`;
}} else {{
document.getElementById('planeStats').innerHTML=`<div>Axes: <strong>${{DATA.axes.join(' / ')}}</strong></div><div>No hyperplane for dense raw point cloud.</div>`;
}}
for(const id of ['showPlane','showPred','showErrors'])document.getElementById(id).addEventListener('change',requestDraw);
document.getElementById('domainBtn').addEventListener('click',()=>{{viewDomain=document.getElementById('domainSelect').value;playing=false;activeSample='';activeTrajectory=[];highlight={{mode:'domain',domain:viewDomain,gas:'',day:'',sample:''}};refreshSampleSelect();document.getElementById('domainStatus').textContent='Showing '+viewDomain+' only.';document.getElementById('status').textContent='Filtered to '+viewDomain;requestDraw()}});
document.getElementById('allDomainBtn').addEventListener('click',()=>{{viewDomain='';highlight={{mode:'none',gas:'',day:'',sample:''}};refreshSampleSelect();document.getElementById('domainStatus').textContent='Showing all source/target batches.';document.getElementById('status').textContent='Showing all domains';requestDraw()}});
document.getElementById('domainStartBtn').addEventListener('click',()=>{{viewDomain=document.getElementById('domainSelect').value;playing=false;activeSample='';activeTrajectory=[];highlight={{mode:'starts',gas:'',day:'',sample:''}};refreshSampleSelect();document.getElementById('domainStatus').textContent='Showing start points for '+viewDomain+'.';document.getElementById('status').textContent='Showing '+viewDomain+' start points: '+startPoints().length+' trials';requestDraw()}});
document.getElementById('gasBtn').addEventListener('click',()=>{{const gas=document.getElementById('gasSelect').value;highlight={{mode:'gas',gas,day:'',sample:''}};document.getElementById('status').textContent='Highlighting all '+gas+' samples across all dates';requestDraw()}});
document.getElementById('dayBtn').addEventListener('click',()=>{{const day=document.getElementById('daySelect').value;highlight={{mode:'day',gas:'',day,sample:''}};document.getElementById('status').textContent='Highlighting '+day+' across all gases';requestDraw()}});
document.getElementById('gasDayBtn').addEventListener('click',()=>{{const gas=document.getElementById('gasSelect').value,day=document.getElementById('daySelect').value;highlight={{mode:'gas_day',gas,day,sample:''}};document.getElementById('status').textContent='Highlighting '+gas+' on '+day;requestDraw()}});
document.getElementById('sampleBtn').addEventListener('click',()=>{{highlight={{mode:'sample',gas:'',day:'',sample:document.getElementById('sampleText').value.trim()}};document.getElementById('status').textContent='Highlighting '+highlight.sample;requestDraw()}});
document.getElementById('startPointsBtn').addEventListener('click',()=>{{playing=false;activeSample='';activeTrajectory=[];highlight={{mode:'starts',gas:'',day:'',sample:''}};document.getElementById('trajStatus').textContent='Showing first point of every trial trajectory.';document.getElementById('status').textContent='Showing all trial start points: '+startPoints().length+' trials';requestDraw()}});
function loadTrajectory(sample){{activeSample=sample;activeTrajectory=DATA.trajectories[sample]||[];trajFrame=0;playing=false;highlight={{mode:'sample',gas:'',day:'',sample}};document.getElementById('sampleText').value=sample;const meta=(DATA.samples||[]).find(s=>s.sample_id===sample);document.getElementById('trajStatus').textContent=meta?`Loaded ${{sample}} (${{meta.label}}, ${{meta.day_label}}), mode: ${{DATA.trajectoryMode}}, points=${{activeTrajectory.length}}`:`Loaded ${{sample}}, points=${{activeTrajectory.length}}`;document.getElementById('status').textContent='Highlighting '+sample;requestDraw()}}
document.getElementById('loadSampleBtn').addEventListener('click',()=>loadTrajectory(document.getElementById('sampleSelect').value));
document.getElementById('playBtn').addEventListener('click',()=>{{if(!activeTrajectory.length)loadTrajectory(document.getElementById('sampleSelect').value);playing=true;lastAnim=0;requestAnimationFrame(animate)}});
document.getElementById('pauseBtn').addEventListener('click',()=>{{playing=false;requestDraw()}});
function rotate(x,y,z){{x-=.5;y-=.5;z-=.5;const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);const x1=x*cy-y*sy,y1=x*sy+y*cy,z1=z,y2=y1*cp-z1*sp,z2=y1*sp+z1*cp;return[x1,y2,z2]}}
function project(p,r){{const s=Math.min(r.width,r.height)*.78*zoom;return[r.width*.49+p[0]*s,r.height*.56-p[1]*s,p[2]]}}
function axisValue(axisIndex,v){{const s=(DATA.scale||[])[axisIndex];if(!s)return v.toFixed(1);const raw=s.displayMin+v*(s.displayMax-s.displayMin);const abs=Math.abs(raw);return abs>=10?raw.toFixed(1):(abs>=1?raw.toFixed(2):raw.toFixed(3))}}
function drawTick(axisIndex,v,r){{let p=[0,0,0],a=[0,0,0],b=[0,0,0];p[axisIndex]=v;a[axisIndex]=v;b[axisIndex]=v;const o1=(axisIndex+1)%3,o2=(axisIndex+2)%3;a[o1]=-.018;b[o1]=.018;const pp=project(rotate(...p),r),pa=project(rotate(...a),r),pb=project(rotate(...b),r);ctx.strokeStyle='#7a8491';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(pa[0],pa[1]);ctx.lineTo(pb[0],pb[1]);ctx.stroke();ctx.fillStyle='#344054';ctx.font='10px Arial';ctx.fillText(axisValue(axisIndex,v),pp[0]+4,pp[1]+12)}}
function drawAxes(r){{const cs=[[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],es=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]],pts=cs.map(c=>project(rotate(...c),r));ctx.lineWidth=1;ctx.strokeStyle='#d9dee3';for(const [a,b] of es){{ctx.beginPath();ctx.moveTo(pts[a][0],pts[a][1]);ctx.lineTo(pts[b][0],pts[b][1]);ctx.stroke()}}ctx.font='13px Arial';ctx.fillStyle='#4f565e';[[[0,0,0],[1.1,0,0],DATA.axes[0]],[[0,0,0],[0,1.1,0],DATA.axes[1]],[[0,0,0],[0,0,1.1],DATA.axes[2]]].forEach(([a,b,t],idx)=>{{const pa=project(rotate(...a),r),pb=project(rotate(...b),r);ctx.strokeStyle='#4f565e';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(pa[0],pa[1]);ctx.lineTo(pb[0],pb[1]);ctx.stroke();ctx.fillText(t,pb[0]+5,pb[1]-5);[0,.5,1].forEach(v=>drawTick(idx,v,r))}})}}
function drawPlane(r){{if(!DATA.plane||!DATA.plane.available||!document.getElementById('showPlane').checked)return;ctx.globalAlpha=.2;ctx.fillStyle='#0969da';ctx.strokeStyle='#0969da';ctx.lineWidth=.7;for(const cell of DATA.plane.cells){{const q=cell.map(p=>project(rotate(p[0],p[1],p[2]),r));ctx.beginPath();ctx.moveTo(q[0][0],q[0][1]);for(let i=1;i<q.length;i++)ctx.lineTo(q[i][0],q[i][1]);ctx.closePath();ctx.fill();ctx.stroke()}}ctx.globalAlpha=1}}
function isHL(p){{const gas=gasNames[p[3]];return (highlight.mode==='domain'&&pointDomain(p)===highlight.domain)||(highlight.mode==='gas'&&gas===highlight.gas)||(highlight.mode==='day'&&p[5]===highlight.day)||(highlight.mode==='gas_day'&&gas===highlight.gas&&p[5]===highlight.day)||(highlight.mode==='sample'&&p[4]===highlight.sample)}}
function color(p){{if(isHL(p))return groupHighlight;if(document.getElementById('showPred').checked&&p[6])return p[6]==='acetone'?predAcetone:predAlcohol;return DATA.colors[gasNames[p[3]]]}}
function drawStartPoints(r){{const starts=startPoints().map(p=>{{const pp=project(rotate(p[0],p[1],p[2]),r);return[pp[0],pp[1],pp[2],p]}}).sort((a,b)=>a[2]-b[2]);ctx.globalAlpha=.92;for(const [x,y,z,p] of starts){{if(x<-10||x>r.width+10||y<-10||y>r.height+10)continue;ctx.fillStyle=DATA.colors[gasNames[p[3]]];ctx.strokeStyle='#111827';ctx.lineWidth=.8;ctx.beginPath();ctx.arc(x,y,5.6,0,Math.PI*2);ctx.fill();ctx.stroke()}}ctx.globalAlpha=1;ctx.fillStyle='#111827';ctx.font='13px Arial';ctx.fillText('All trial start points: '+starts.length,18,r.height-18)}}
function drawTrajectory(r){{if(!activeTrajectory.length)return;const end=Math.min(activeTrajectory.length-1,trajFrame);ctx.lineWidth=3;ctx.strokeStyle=groupHighlight;ctx.globalAlpha=.92;ctx.beginPath();for(let i=0;i<=end;i++){{const p=activeTrajectory[i],pp=project(rotate(p[1],p[2],p[3]),r);if(i===0)ctx.moveTo(pp[0],pp[1]);else ctx.lineTo(pp[0],pp[1]);}}ctx.stroke();const p=activeTrajectory[end],pp=project(rotate(p[1],p[2],p[3]),r);ctx.fillStyle=groupHighlight;ctx.beginPath();ctx.arc(pp[0],pp[1],8,0,Math.PI*2);ctx.fill();ctx.fillStyle='#111827';ctx.font='12px Arial';ctx.fillText(`${{activeSample}}  t=${{p[0].toFixed(1)}}s`,pp[0]+10,pp[1]-10);ctx.globalAlpha=1}}
function draw(){{const r=canvas.getBoundingClientRect();ctx.clearRect(0,0,r.width,r.height);ctx.fillStyle='#fafbfc';ctx.fillRect(0,0,r.width,r.height);drawAxes(r);drawPlane(r);if(highlight.mode==='starts'){{drawStartPoints(r);return}}const pts=[];for(const p of DATA.points){{if(!pointVisible(p))continue;const pp=project(rotate(p[0],p[1],p[2]),r);pts.push([pp[0],pp[1],pp[2],p])}}pts.sort((a,b)=>a[2]-b[2]);for(const [x,y,z,p] of pts){{if(x<-8||x>r.width+8||y<-8||y>r.height+8)continue;ctx.globalAlpha=DATA.mode==='raw_points'||DATA.mode==='raw_points_clip'?.42:.86;ctx.fillStyle=(document.getElementById('showErrors').checked&&p[7]===1)?errorColor:color(p);ctx.beginPath();ctx.arc(x,y,isHL(p)?7:(DATA.mode.startsWith('raw_points')?2.1:5.2),0,Math.PI*2);ctx.fill();}}ctx.globalAlpha=1;drawTrajectory(r)}}
function animate(ts){{if(!playing)return;if(!lastAnim)lastAnim=ts;if(ts-lastAnim>24){{trajFrame=(trajFrame+2)%Math.max(1,activeTrajectory.length);lastAnim=ts;requestDraw()}}requestAnimationFrame(animate)}}
canvas.addEventListener('pointerdown',e=>{{dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId)}});
canvas.addEventListener('pointermove',e=>{{if(!dragging)return;const dx=e.clientX-lastX,dy=e.clientY-lastY;lastX=e.clientX;lastY=e.clientY;yaw+=dx*.008;pitch=Math.max(-1.25,Math.min(1.25,pitch+dy*.008));requestDraw()}});
canvas.addEventListener('pointerup',()=>dragging=false);
canvas.addEventListener('wheel',e=>{{e.preventDefault();zoom=Math.max(.4,Math.min(4,zoom*(e.deltaY>0?.92:1.08)));requestDraw()}},{{passive:false}});
canvas.addEventListener('dblclick',()=>{{yaw=-.66;pitch=.39;zoom=1.08;requestDraw()}});
window.addEventListener('resize',resize);resize();
</script>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")


def clean_output_dirs() -> None:
    for directory in [RAW_DIR, RAW_CUT_DIR, UCI_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        for html_path in directory.glob("*.html"):
            html_path.unlink()


def main() -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    clean_output_dirs()
    trials, corrected, absolute = load_trials()
    zero_started = zero_started_series(corrected)
    absolute_zero_started = zero_started_series(absolute)
    trials.drop(columns=["t", "baseline_mask", "exposure_mask", "recovery_mask"]).to_csv(
        TABLE_ROOT / "feature_space_current_manifest_samples.csv", index=False
    )
    summary_rows = []
    scale_rows = []

    for slug, axes in RAW_AXES:
        payload, scale = raw_point_payload(
            trials,
            corrected,
            axes,
            f"Baseline-corrected raw point cloud {slug.upper()}",
            "no_clip",
            axis_label_by_sample=absolute,
        )
        payload["mode"] = "raw_points"
        out = RAW_DIR / f"interactive_bc_no_clip_{slug}_feature_space.html"
        write_html(out, payload)
        scale["space"] = out.name
        scale_rows.append(scale)
        summary_rows.append({"space": out.name, "folder": "feature_space_raw", "points": len(payload["points"]), "kind": "raw point cloud"})

        payload, scale, hp = group_payload(
            trials,
            corrected,
            axes,
            f"Post-baseline group hyperplane {slug.upper()}",
            "no_clip",
            axis_label_by_sample=absolute,
        )
        payload["mode"] = "group_hyperplane"
        out = RAW_DIR / f"interactive_bc_no_clip_post60_group_hyperplane_{slug}.html"
        write_html(out, payload)
        scale["space"] = out.name
        scale_rows.append(scale)
        summary_rows.append({"space": out.name, "folder": "feature_space_raw", "points": len(payload["points"]), "kind": "group hyperplane", "balanced_accuracy": hp.get("balanced_accuracy")})

        payload, scale = raw_point_payload(
            trials,
            corrected,
            axes,
            f"Baseline-corrected raw point cloud {slug.upper()} with percentile cut",
            "clip",
            axis_label_by_sample=absolute,
        )
        payload["mode"] = "raw_points_clip"
        out = RAW_CUT_DIR / f"interactive_bc_{slug}_feature_space.html"
        write_html(out, payload)
        scale["space"] = out.name
        scale_rows.append(scale)
        summary_rows.append({"space": out.name, "folder": "feature_space_raw_cut", "points": len(payload["points"]), "kind": "raw point cloud with 1-99 percentile display cut"})

        payload, scale = raw_point_payload(
            trials,
            zero_started,
            axes,
            f"Zero-started baseline-corrected raw point cloud {slug.upper()} with percentile cut",
            "clip",
            axis_label_by_sample=absolute_zero_started,
        )
        payload["mode"] = "raw_points_clip"
        payload["note"] = "each sample is shifted so its first valid baseline-corrected point is at the origin; baseline/reference are treated as air"
        out = RAW_CUT_DIR / f"interactive_bc_zero_start_{slug}_feature_space.html"
        write_html(out, payload)
        scale["space"] = out.name
        scale_rows.append(scale)
        summary_rows.append({"space": out.name, "folder": "feature_space_raw_cut", "points": len(payload["points"]), "kind": "zero-started raw point cloud with 1-99 percentile display cut"})

        payload, scale, hp = group_payload(
            trials,
            corrected,
            axes,
            f"Post-baseline group hyperplane {slug.upper()} with percentile cut",
            "clip",
            axis_label_by_sample=absolute,
        )
        payload["mode"] = "group_hyperplane"
        out = RAW_CUT_DIR / f"interactive_bc_post60_group_hyperplane_{slug}.html"
        write_html(out, payload)
        scale["space"] = out.name
        scale_rows.append(scale)
        summary_rows.append({"space": out.name, "folder": "feature_space_raw_cut", "points": len(payload["points"]), "kind": "group hyperplane with 1-99 percentile display cut", "balanced_accuracy": hp.get("balanced_accuracy")})

    uci = make_uci_features(trials, corrected)
    uci.to_csv(TABLE_ROOT / "feature_space_current_uci_features.csv", index=False)
    loading_rows = []
    for slug, sensors in UCI_SPACES:
        payload, scale, loadings, hp = uci_payload(
            trials,
            corrected,
            uci,
            sensors,
            f"UCI normDeltaR feature space no-clip {slug.upper()}",
        )
        payload["mode"] = "uci_norm_delta"
        out = UCI_DIR / f"uci_norm_delta_no_clip_{slug}.html"
        write_html(out, payload)
        scale["space"] = out.name
        scale_rows.append(scale)
        loading_rows.append(loadings)
        summary_rows.append({"space": out.name, "folder": "feature_space_uci", "points": len(payload["points"]), "kind": "UCI normDeltaR sensor feature space", "balanced_accuracy": hp.get("balanced_accuracy")})

    pd.DataFrame(summary_rows).to_csv(TABLE_ROOT / "feature_space_current_summary.csv", index=False)
    pd.concat(scale_rows, ignore_index=True).to_csv(TABLE_ROOT / "feature_space_current_scales.csv", index=False)
    if loading_rows:
        pd.concat(loading_rows, ignore_index=True).to_csv(TABLE_ROOT / "feature_space_current_uci_pca_loadings.csv", index=False)
    print("Updated feature-space figures")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
