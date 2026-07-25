from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path(r"D:\datasets\enose_export\enose_data")
RECORD_PATH = Path(r"C:\Users\yaoli\Desktop\record_enose.xlsx")
OUT_ROOT = Path(r"D:\thesis")
SENSORS = [f"s{i}" for i in range(1, 7)]
GASES = ["air", "alcohol", "acetone"]
UCI_FEATURE_TABLE = OUT_ROOT / "tables" / "uci_style_trial_level_features.csv"
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


def natural_key(path: Path) -> tuple:
    chunks: list[object] = []
    current = ""
    is_digit = None
    for char in path.stem:
        char_is_digit = char.isdigit()
        if is_digit is None or char_is_digit == is_digit:
            current += char
        else:
            chunks.append(int(current) if is_digit else current)
            current = char
        is_digit = char_is_digit
    if current:
        chunks.append(int(current) if is_digit else current)
    return tuple(chunks)


def day_num(day: str) -> int:
    match = re.search(r"\d+", str(day))
    return int(match.group(0)) if match else 10**9


def normalize_day_label(value: str) -> str | None:
    match = re.fullmatch(r"\s*day\s*(\d+)\s*", value, re.IGNORECASE)
    if not match:
        return None
    return f"Day{int(match.group(1))}"


def sample_parts(sample_name: str) -> tuple[str, int] | None:
    match = re.fullmatch(r"\s*(air|alcohol|acetone)_(\d+)\s*", sample_name, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower(), int(match.group(2))


def load_record_mapping() -> dict[str, dict[str, object]]:
    raw = pd.read_excel(RECORD_PATH, header=None)
    current_day: str | None = None
    current_date = ""
    mapping: dict[str, dict[str, object]] = {}
    for _, row in raw.iterrows():
        first = "" if pd.isna(row.iloc[0]) else str(row.iloc[0]).strip()
        day = normalize_day_label(first)
        if day:
            current_day = day
            current_date = "" if pd.isna(row.iloc[2]) else str(row.iloc[2]).strip()
            continue
        parts = sample_parts(first)
        if not parts or current_day is None:
            continue
        gas, group = parts
        source_file = f"{gas}_{group}.csv"
        mapping[source_file] = {
            "source_file": source_file,
            "gas": gas,
            "group": group,
            "day": current_day,
            "date": current_date,
        }
    return mapping


def elapsed_seconds(df: pd.DataFrame) -> pd.Series:
    if "arduino_time" in df.columns:
        t = pd.to_numeric(df["arduino_time"], errors="coerce")
        elapsed = (t - t.iloc[0]) / 1000.0
        if elapsed.notna().sum() > 10 and elapsed.max() > 30:
            return elapsed
    t = pd.to_numeric(df["time_s"], errors="coerce")
    return t - t.iloc[0]


def load_raw6_post60_trial_centroids(record_map: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for gas in GASES:
        for path in sorted((DATA_ROOT / gas).glob("*.csv"), key=natural_key):
            raw = pd.read_csv(path)
            elapsed = elapsed_seconds(raw)
            baseline_mask = elapsed.between(0, 60, inclusive="left")
            post60_mask = elapsed >= 60
            if baseline_mask.sum() < 5 or post60_mask.sum() < 5:
                continue
            sensors = raw[SENSORS].apply(pd.to_numeric, errors="coerce")
            corrected = sensors - sensors.loc[baseline_mask].mean()
            values = corrected.loc[post60_mask, SENSORS].mean(axis=0)
            meta = record_map.get(path.name, {})
            group_match = re.search(r"_(\d+)\.csv$", path.name)
            row = {
                "source_file": path.name,
                "gas": gas,
                "group": int(meta.get("group", int(group_match.group(1)) if group_match else -1)),
                "day": str(meta.get("day", "Unknown")),
                "post60_points": int(post60_mask.sum()),
            }
            for sensor in SENSORS:
                row[f"rawMean_{sensor}"] = float(values[sensor])
            rows.append(row)
    return pd.DataFrame(rows)


def uci48_columns() -> list[str]:
    return [pattern.format(sensor=sensor) for sensor in SENSORS for pattern in FEATURE_FAMILIES]


def zscore(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    x = out[columns].to_numpy(float)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(std == 0, 1.0, std)
    out[columns] = (x - mean) / std
    return out


def day_gas_centroids(df: pd.DataFrame, columns: list[str], representation: str) -> pd.DataFrame:
    rows = []
    for (day, gas), group in df.groupby(["day", "gas"], dropna=False):
        row = {
            "representation": representation,
            "day": str(day),
            "day_num": day_num(str(day)),
            "gas": gas,
            "n_trials": len(group),
        }
        vals = group[columns].to_numpy(float).mean(axis=0)
        for col, val in zip(columns, vals):
            row[col] = val
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["day_num", "gas"]).reset_index(drop=True)


def vector_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an == 0 or bn == 0:
        return float("nan")
    cosv = float(np.clip(np.dot(a, b) / (an * bn), -1, 1))
    return float(np.degrees(np.arccos(cosv)))


def compute_drift_metrics(centroids: pd.DataFrame, columns: list[str], representation: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available_days = sorted(centroids["day"].unique().tolist(), key=day_num)
    centroid_map = {
        (str(row["day"]), str(row["gas"])): row[columns].to_numpy(float)
        for _, row in centroids.iterrows()
    }
    sep_rows = []
    drift_rows = []
    risk_rows = []

    for day in available_days:
        if (day, "alcohol") in centroid_map and (day, "acetone") in centroid_map:
            sep = centroid_map[(day, "acetone")] - centroid_map[(day, "alcohol")]
            sep_rows.append({
                "representation": representation,
                "day": day,
                "day_num": day_num(day),
                "separation": "acetone_minus_alcohol",
                "distance": vector_norm(sep),
            })
        if (day, "air") in centroid_map:
            for gas in ["alcohol", "acetone"]:
                if (day, gas) in centroid_map:
                    sep = centroid_map[(day, gas)] - centroid_map[(day, "air")]
                    sep_rows.append({
                        "representation": representation,
                        "day": day,
                        "day_num": day_num(day),
                        "separation": f"{gas}_minus_air",
                        "distance": vector_norm(sep),
                    })

    for d0, d1 in zip(available_days[:-1], available_days[1:]):
        sep0 = None
        if (d0, "alcohol") in centroid_map and (d0, "acetone") in centroid_map:
            sep0 = centroid_map[(d0, "acetone")] - centroid_map[(d0, "alcohol")]
            sep_unit = sep0 / max(np.linalg.norm(sep0), 1e-12)
        else:
            sep_unit = None
        shared_gases = [gas for gas in GASES if (d0, gas) in centroid_map and (d1, gas) in centroid_map]
        gas_drifts = []
        for gas in shared_gases:
            drift = centroid_map[(d1, gas)] - centroid_map[(d0, gas)]
            gas_drifts.append(drift)
            projection_on_sep = float(np.dot(drift, sep_unit)) if sep_unit is not None else float("nan")
            drift_rows.append({
                "representation": representation,
                "from_day": d0,
                "to_day": d1,
                "from_day_num": day_num(d0),
                "to_day_num": day_num(d1),
                "gas": gas,
                "drift_distance": vector_norm(drift),
                "angle_to_acetone_minus_alcohol_deg": angle_deg(drift, sep0) if sep0 is not None else float("nan"),
                "projection_on_acetone_minus_alcohol": projection_on_sep,
                "interpretation": (
                    "positive means movement toward acetone side" if gas == "alcohol"
                    else "negative means movement toward alcohol side" if gas == "acetone"
                    else "projection relative to acetone-minus-alcohol axis"
                ),
            })
        if gas_drifts:
            balanced_drift = np.mean(np.vstack(gas_drifts), axis=0)
            drift_rows.append({
                "representation": representation,
                "from_day": d0,
                "to_day": d1,
                "from_day_num": day_num(d0),
                "to_day_num": day_num(d1),
                "gas": "balanced_all_available_gases",
                "drift_distance": vector_norm(balanced_drift),
                "angle_to_acetone_minus_alcohol_deg": angle_deg(balanced_drift, sep0) if sep0 is not None else float("nan"),
                "projection_on_acetone_minus_alcohol": float(np.dot(balanced_drift, sep_unit)) if sep_unit is not None else float("nan"),
                "interpretation": "overall balanced day shift relative to acetone-minus-alcohol axis",
            })

    # Linear trend per gas and simple nearest-approach risk in the same standardized space.
    for gas in ["alcohol", "acetone", "air"]:
        gas_cent = centroids[centroids["gas"] == gas].sort_values("day_num")
        if len(gas_cent) < 3:
            continue
        t = gas_cent["day_num"].to_numpy(float)
        x = gas_cent[columns].to_numpy(float)
        t_center = t - t.mean()
        denom = float(np.dot(t_center, t_center))
        slope = (t_center[:, None] * x).sum(axis=0) / max(denom, 1e-12)
        risk_rows.append({
            "representation": representation,
            "gas": gas,
            "n_days": len(gas_cent),
            "linear_drift_speed_per_day": vector_norm(slope),
            "first_to_last_distance": vector_norm(x[-1] - x[0]),
            "mean_step_distance": float(np.mean(np.linalg.norm(np.diff(x, axis=0), axis=1))) if len(x) > 1 else float("nan"),
        })

    return pd.DataFrame(sep_rows), pd.DataFrame(drift_rows), pd.DataFrame(risk_rows)


def nearest_cross_gas_distances(centroids: pd.DataFrame, columns: list[str], representation: str) -> pd.DataFrame:
    rows = []
    alcohol = centroids[centroids["gas"] == "alcohol"].sort_values("day_num")
    acetone = centroids[centroids["gas"] == "acetone"].sort_values("day_num")
    for _, a in alcohol.iterrows():
        av = a[columns].to_numpy(float)
        best = None
        for _, c in acetone.iterrows():
            cv = c[columns].to_numpy(float)
            dist = vector_norm(av - cv)
            if best is None or dist < best[0]:
                best = (dist, c["day"])
        if best:
            rows.append({
                "representation": representation,
                "query_gas": "alcohol",
                "query_day": a["day"],
                "nearest_other_gas": "acetone",
                "nearest_other_day": best[1],
                "distance": best[0],
            })
    for _, c in acetone.iterrows():
        cv = c[columns].to_numpy(float)
        best = None
        for _, a in alcohol.iterrows():
            av = a[columns].to_numpy(float)
            dist = vector_norm(cv - av)
            if best is None or dist < best[0]:
                best = (dist, a["day"])
        if best:
            rows.append({
                "representation": representation,
                "query_gas": "acetone",
                "query_day": c["day"],
                "nearest_other_gas": "alcohol",
                "nearest_other_day": best[1],
                "distance": best[0],
            })
    return pd.DataFrame(rows)


def main() -> None:
    (OUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    record_map = load_record_mapping()

    raw = load_raw6_post60_trial_centroids(record_map)
    raw_cols = [f"rawMean_{s}" for s in SENSORS]
    raw_z = zscore(raw, raw_cols)
    raw_centroids = day_gas_centroids(raw_z, raw_cols, "raw6_post60_centroid_zscore")

    uci = pd.read_csv(UCI_FEATURE_TABLE)
    uci_cols = uci48_columns()
    uci_z = zscore(uci, uci_cols)
    uci_centroids = day_gas_centroids(uci_z, uci_cols, "uci48_features_zscore")

    all_centroids = pd.concat([raw_centroids, uci_centroids], ignore_index=True, sort=False)
    all_centroids.to_csv(OUT_ROOT / "tables" / "drift_day_gas_centroids_raw6_vs_uci48.csv", index=False, encoding="utf-8-sig")

    tables = []
    for centroids, cols, rep in [
        (raw_centroids, raw_cols, "raw6_post60_centroid_zscore"),
        (uci_centroids, uci_cols, "uci48_features_zscore"),
    ]:
        sep, drift, trend = compute_drift_metrics(centroids, cols, rep)
        nearest = nearest_cross_gas_distances(centroids, cols, rep)
        tables.append((sep, drift, trend, nearest))

    pd.concat([x[0] for x in tables], ignore_index=True).to_csv(OUT_ROOT / "tables" / "drift_separation_distances_raw6_vs_uci48.csv", index=False, encoding="utf-8-sig")
    pd.concat([x[1] for x in tables], ignore_index=True).to_csv(OUT_ROOT / "tables" / "drift_day_to_day_vectors_raw6_vs_uci48.csv", index=False, encoding="utf-8-sig")
    pd.concat([x[2] for x in tables], ignore_index=True).to_csv(OUT_ROOT / "tables" / "drift_linear_trend_summary_raw6_vs_uci48.csv", index=False, encoding="utf-8-sig")
    pd.concat([x[3] for x in tables], ignore_index=True).to_csv(OUT_ROOT / "tables" / "drift_nearest_cross_gas_days_raw6_vs_uci48.csv", index=False, encoding="utf-8-sig")

    # Compact human-readable summary.
    summary_rows = []
    sep_all = pd.concat([x[0] for x in tables], ignore_index=True)
    drift_all = pd.concat([x[1] for x in tables], ignore_index=True)
    nearest_all = pd.concat([x[3] for x in tables], ignore_index=True)
    for rep in sep_all["representation"].unique():
        sep_rep = sep_all[(sep_all["representation"] == rep) & (sep_all["separation"] == "acetone_minus_alcohol")]
        drift_rep = drift_all[(drift_all["representation"] == rep) & (drift_all["gas"].isin(["alcohol", "acetone"]))]
        nearest_rep = nearest_all[nearest_all["representation"] == rep]
        risky = drift_rep.dropna(subset=["angle_to_acetone_minus_alcohol_deg"]).copy()
        if not risky.empty:
            risky["axis_alignment"] = np.abs(np.cos(np.radians(risky["angle_to_acetone_minus_alcohol_deg"].astype(float))))
        summary_rows.append({
            "representation": rep,
            "mean_alcohol_acetone_distance": float(sep_rep["distance"].mean()) if not sep_rep.empty else float("nan"),
            "min_alcohol_acetone_distance": float(sep_rep["distance"].min()) if not sep_rep.empty else float("nan"),
            "mean_day_to_day_gas_drift": float(drift_rep["drift_distance"].mean()) if not drift_rep.empty else float("nan"),
            "max_day_to_day_gas_drift": float(drift_rep["drift_distance"].max()) if not drift_rep.empty else float("nan"),
            "mean_abs_alignment_with_alcohol_acetone_axis": float(risky["axis_alignment"].mean()) if not risky.empty else float("nan"),
            "nearest_cross_gas_day_distance_min": float(nearest_rep["distance"].min()) if not nearest_rep.empty else float("nan"),
        })
    pd.DataFrame(summary_rows).to_csv(OUT_ROOT / "tables" / "drift_compact_summary_raw6_vs_uci48.csv", index=False, encoding="utf-8-sig")

    print(OUT_ROOT / "tables" / "drift_compact_summary_raw6_vs_uci48.csv")


if __name__ == "__main__":
    main()
