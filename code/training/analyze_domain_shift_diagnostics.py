from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
FEATURE_PATH = ROOT / "thesis_out" / "processed" / "sample_features_baseline_corrected.csv"
TABLE_DIR = ROOT / "thesis_out" / "tables"
REPORT_DIR = ROOT / "thesis_out" / "reports"

META_COLS = {"sample_id", "label", "duration_s", "collection_date", "distance_cm", "wind_level"}
GASES = ["air", "alcohol", "acetone"]


def zscore_from_source(xs: np.ndarray, xt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(xs, axis=0)
    std = np.nanstd(xs, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (xs - mean) / std, (xt - mean) / std, mean, std


def zscore_global(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (x - mean) / std, mean, std


def covariance(x: np.ndarray) -> np.ndarray:
    if len(x) <= 1:
        return np.eye(x.shape[1])
    return np.cov(x, rowvar=False)


def coral_distance(xs: np.ndarray, xt: np.ndarray) -> float:
    cs = covariance(xs)
    ct = covariance(xt)
    return float(np.linalg.norm(cs - ct, ord="fro") / (4.0 * xs.shape[1] ** 2))


def sqrtm_spd(cov: np.ndarray, inverse: bool = False) -> np.ndarray:
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-8, None)
    if inverse:
        vals = 1.0 / np.sqrt(vals)
    else:
        vals = np.sqrt(vals)
    return (vecs * vals) @ vecs.T


def coral_align_source_to_target(xs: np.ndarray, xt: np.ndarray) -> np.ndarray:
    ms = xs.mean(axis=0, keepdims=True)
    mt = xt.mean(axis=0, keepdims=True)
    cs = covariance(xs - ms) + np.eye(xs.shape[1]) * 1e-4
    ct = covariance(xt - mt) + np.eye(xt.shape[1]) * 1e-4
    return (xs - ms) @ sqrtm_spd(cs, inverse=True) @ sqrtm_spd(ct, inverse=False) + mt


def rbf_mmd2(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    xy = np.vstack([x, y])
    d2 = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=2)
    nonzero = d2[d2 > 0]
    gamma = 1.0 / (2.0 * np.median(nonzero)) if nonzero.size else 1.0
    kxx = np.exp(-gamma * np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2))
    kyy = np.exp(-gamma * np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=2))
    kxy = np.exp(-gamma * np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2))
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def nearest_centroid_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    labels = np.array([label for label in GASES if np.any(y_train == label)])
    centroids = np.vstack([x_train[y_train == label].mean(axis=0) for label in labels])
    d2 = np.sum((x_test[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return labels[np.argmin(d2, axis=1)]


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for label in GASES:
        mask = y_true == label
        if mask.any():
            recalls.append(float((y_pred[mask] == label).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def class_balance_string(labels: pd.Series) -> str:
    counts = labels.value_counts().reindex(GASES, fill_value=0)
    total = counts.sum()
    return ", ".join(f"{name}:{int(value)}({value / total:.2f})" for name, value in counts.items())


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    text = frame.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            text[col] = text[col].map(lambda value: "" if pd.isna(value) else str(value))
    headers = list(text.columns)
    rows = text.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def anova_f(values: np.ndarray, groups: np.ndarray) -> float:
    grand = np.nanmean(values)
    ss_between = 0.0
    ss_within = 0.0
    n_groups = 0
    for group in pd.unique(groups):
        vals = values[groups == group]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        n_groups += 1
        mean = vals.mean()
        ss_between += len(vals) * (mean - grand) ** 2
        ss_within += np.sum((vals - mean) ** 2)
    df_between = max(n_groups - 1, 1)
    df_within = max(len(values) - n_groups, 1)
    return float((ss_between / df_between) / max(ss_within / df_within, 1e-12))


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURE_PATH)
    df["collection_date"] = pd.to_datetime(df["collection_date"]).dt.strftime("%Y-%m-%d")

    numeric_cols = [
        col
        for col in df.columns
        if col not in META_COLS and pd.api.types.is_numeric_dtype(df[col])
    ]
    baseline_cols = [col for col in numeric_cols if col.endswith("_baseline")]
    response_cols = [col for col in numeric_cols if col not in baseline_cols]

    x_response = df[response_cols].to_numpy(float)
    xz, _, _ = zscore_global(x_response)

    source_mask = df["collection_date"] <= "2026-06-01"
    target_mask = df["collection_date"].isin(["2026-06-04", "2026-06-05"])
    source = df[source_mask].copy()
    target = df[target_mask].copy()
    xs_raw = source[response_cols].to_numpy(float)
    xt_raw = target[response_cols].to_numpy(float)
    xs, xt, _, _ = zscore_from_source(xs_raw, xt_raw)

    source_centroid = xs.mean(axis=0)
    rows = []
    cond_rows = []
    for date in sorted(df["collection_date"].unique()):
        date_mask = df["collection_date"] == date
        subset = df[date_mask]
        xd_raw = subset[response_cols].to_numpy(float)
        _, xd, _, _ = zscore_from_source(xs_raw, xd_raw)
        centroid_dist = float(np.linalg.norm(xd.mean(axis=0) - source_centroid))
        rows.append(
            {
                "collection_date": date,
                "n": int(len(subset)),
                "class_balance": class_balance_string(subset["label"]),
                "mmd2_to_source": rbf_mmd2(xs, xd),
                "coral_distance_to_source": coral_distance(xs, xd),
                "centroid_distance_to_source": centroid_dist,
                "has_all_three_classes": bool(set(GASES).issubset(set(subset["label"]))),
            }
        )
        for gas in GASES:
            date_gas = subset["label"] == gas
            src_gas = source["label"] == gas
            if not date_gas.any() or not src_gas.any():
                continue
            xdg_raw = subset.loc[date_gas, response_cols].to_numpy(float)
            xsg_raw = source.loc[src_gas, response_cols].to_numpy(float)
            xsg, xdg, _, _ = zscore_from_source(xsg_raw, xdg_raw)
            cond_rows.append(
                {
                    "collection_date": date,
                    "label": gas,
                    "n": int(date_gas.sum()),
                    "conditional_centroid_distance_to_source_class": float(
                        np.linalg.norm(xdg.mean(axis=0) - xsg.mean(axis=0))
                    ),
                    "conditional_mmd2_to_source_class": rbf_mmd2(xsg, xdg),
                }
            )

    date_summary = pd.DataFrame(rows)
    cond_summary = pd.DataFrame(cond_rows)

    # New-bottle domain-adaptation probe with labels hidden during adaptation.
    y_source = source["label"].to_numpy(str)
    y_target = target["label"].to_numpy(str)
    pred_raw = nearest_centroid_predict(xs, y_source, xt)
    xs_coral = coral_align_source_to_target(xs, xt)
    pred_coral = nearest_centroid_predict(xs_coral, y_source, xt)
    eval_df = pd.DataFrame(
        [
            {
                "setting": "source<=2026-06-01_to_target=2026-06-04/05_new_bottle",
                "method": "no_domain_adaptation_nearest_centroid",
                "target_balanced_accuracy": balanced_accuracy(y_target, pred_raw),
                "target_accuracy": float((pred_raw == y_target).mean()),
                "source_n": int(len(source)),
                "target_n": int(len(target)),
                "source_class_balance": class_balance_string(source["label"]),
                "target_class_balance": class_balance_string(target["label"]),
            },
            {
                "setting": "source<=2026-06-01_to_target=2026-06-04/05_new_bottle",
                "method": "unsupervised_CORAL_then_nearest_centroid",
                "target_balanced_accuracy": balanced_accuracy(y_target, pred_coral),
                "target_accuracy": float((pred_coral == y_target).mean()),
                "source_n": int(len(source)),
                "target_n": int(len(target)),
                "source_class_balance": class_balance_string(source["label"]),
                "target_class_balance": class_balance_string(target["label"]),
            },
        ]
    )

    # PCA drift view.
    u, s, vt = np.linalg.svd(xz, full_matrices=False)
    coords = xz @ vt[:3].T
    explained = (s**2) / max(1, len(df) - 1)
    explained_ratio = explained / explained.sum()
    pca_df = df[["sample_id", "label", "collection_date"]].copy()
    pca_df["pc1"] = coords[:, 0]
    pca_df["pc2"] = coords[:, 1]
    pca_df["pc3"] = coords[:, 2]

    # Feature screening: large class F, small date F is a domain-invariant candidate.
    feat_rows = []
    for col in response_cols:
        vals = df[col].to_numpy(float)
        class_f = anova_f(vals, df["label"].to_numpy(str))
        date_f = anova_f(vals, df["collection_date"].to_numpy(str))
        feat_rows.append(
            {
                "feature": col,
                "class_f": class_f,
                "date_f": date_f,
                "class_to_date_ratio": class_f / max(date_f, 1e-12),
            }
        )
    feature_scores = pd.DataFrame(feat_rows).sort_values(
        ["class_to_date_ratio", "class_f"], ascending=[False, False]
    )

    # Baseline drift is analyzed separately because response features are already baseline-corrected.
    baseline_rows = []
    for col in baseline_cols:
        vals = df[col].to_numpy(float)
        baseline_rows.append(
            {
                "feature": col,
                "date_f": anova_f(vals, df["collection_date"].to_numpy(str)),
                "class_f": anova_f(vals, df["label"].to_numpy(str)),
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
                "range": float(np.nanmax(vals) - np.nanmin(vals)),
            }
        )
    baseline_scores = pd.DataFrame(baseline_rows).sort_values("date_f", ascending=False)

    date_summary.to_csv(TABLE_DIR / "domain_shift_date_summary.csv", index=False)
    cond_summary.to_csv(TABLE_DIR / "domain_shift_class_conditional.csv", index=False)
    eval_df.to_csv(TABLE_DIR / "domain_shift_new_bottle_da_eval.csv", index=False)
    feature_scores.to_csv(TABLE_DIR / "domain_shift_invariant_feature_candidates.csv", index=False)
    baseline_scores.to_csv(TABLE_DIR / "domain_shift_baseline_drift_by_sensor.csv", index=False)
    pca_df.to_csv(TABLE_DIR / "domain_shift_pca_scores.csv", index=False)

    target_cond = cond_summary[cond_summary["collection_date"].isin(["2026-06-04", "2026-06-05"])]
    date_top = date_summary.sort_values("centroid_distance_to_source", ascending=False).head(5)
    top_features = feature_scores.head(8)
    baseline_top = baseline_scores.head(6)

    report = f"""# Domain Adaptation 漂移诊断

输入特征表：`{FEATURE_PATH}`

主分析特征集：仅使用 baseline-corrected response features，排除 `sample_id`、`label`、`collection_date`、`duration_s`、`distance_cm`、`wind_level` 和原始 `*_baseline` 列。

## 数据集划分

- 总样本数：{len(df)}
- 日期数：{df['collection_date'].nunique()}
- 新瓶实验 source domain：日期 <= 2026-06-01，n={len(source)}，类别比例：{class_balance_string(source['label'])}
- 新瓶实验 target domain：2026-06-04/2026-06-05，n={len(target)}，类别比例：{class_balance_string(target['label'])}
- 2026-06-02 只有 air，因此可用于观察 air baseline drift，但不适合作为完整三分类 target domain。

## 主要漂移观察

- PCA 解释方差：PC1={explained_ratio[0]:.3f}, PC2={explained_ratio[1]:.3f}, PC3={explained_ratio[2]:.3f}。
- 新瓶 target 相对 source 的 marginal MMD2：{rbf_mmd2(xs, xt):.4f}。
- 新瓶 target 相对 source 的 CORAL covariance distance：{coral_distance(xs, xt):.6f}。
- 新瓶 target 相对 source 的 centroid distance：{float(np.linalg.norm(xt.mean(axis=0) - xs.mean(axis=0))):.3f}。

## 新瓶 domain adaptation 小实验

{markdown_table(eval_df)}

## 与 source period 差异最大的日期

{markdown_table(date_top[['collection_date', 'n', 'class_balance', 'mmd2_to_source', 'coral_distance_to_source', 'centroid_distance_to_source']])}

## 新瓶 target 日期上的 class-conditional drift

{markdown_table(target_cond[['collection_date', 'label', 'n', 'conditional_centroid_distance_to_source_class', 'conditional_mmd2_to_source_class']])}

## 候选 domain-invariant response features

class-to-date ratio 越高，说明该特征携带的气体类别信息相对多、日期/domain 信息相对少，更适合作为共同特征。

{markdown_table(top_features)}

## 各传感器 baseline drift

原始 baseline 列单独报告，因为上面的 response features 已经做过 baseline correction。

{markdown_table(baseline_top)}

## 结论解释

该数据同时存在 marginal domain shift 和 class-conditional shift。marginal shift 体现在不同日期的 centroid、covariance 和 MMD 距离发生变化；conditional shift 体现在同一种气体在 target 日期上的 class-conditional centroid/MMD 仍然明显偏离 source。由于不同日期的类别比例也不完全一致，完整的 joint distribution P(X,Y) 也发生了变化。

对你的数据来说，单纯做 marginal distribution alignment 是可行的，但如果单独使用会有风险，因为类别比例变化可能让 marginal alignment 把不同气体类别拉近。更适合的论文表述是 conditional 或 joint adaptation：在对齐不同日期/domain 的同时，保留气体类别结构。实现上可以使用 source labels，并用 target pseudo-labels 或 class-aware loss 辅助目标域对齐。
"""
    report_path = REPORT_DIR / "domain_adaptation_drift_diagnosis_cn.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"Wrote {TABLE_DIR / 'domain_shift_date_summary.csv'}")
    print(f"Wrote {TABLE_DIR / 'domain_shift_class_conditional.csv'}")
    print(f"Wrote {TABLE_DIR / 'domain_shift_new_bottle_da_eval.csv'}")
    print(f"Wrote {TABLE_DIR / 'domain_shift_invariant_feature_candidates.csv'}")
    print(f"Wrote {TABLE_DIR / 'domain_shift_baseline_drift_by_sensor.csv'}")
    print(f"Wrote {REPORT_DIR / 'domain_adaptation_drift_diagnosis_cn.md'}")


if __name__ == "__main__":
    main()
