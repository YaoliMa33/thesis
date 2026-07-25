from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_manifest_da_baseline_splits as ml  # noqa: E402


OUT_ROOT = Path(r"D:\thesis")
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures" / "concentration"
LEGACY_FEATURE_PATH = TABLE_DIR / "legacy9_manifest_baseline_as_air_features.csv"
FULL_ML_PATH = TABLE_DIR / "legacy9_manifest_da_baseline_split_results.csv"

EXPERIMENTS = ["amplitude_only", "shape_normalized"]


def legacy_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c[:2] in {f"s{i}" for i in range(1, 7)} and c[2:3] == "_"]


def add_control_features(df: pd.DataFrame, cols: list[str], experiment: str) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    x = out[cols].to_numpy(float)
    if experiment == "amplitude_only":
        out["global_response_energy"] = np.sqrt(np.nanmean(x * x, axis=1))
        return out, ["global_response_energy"]
    if experiment == "shape_normalized":
        norm = np.linalg.norm(np.nan_to_num(x), axis=1)
        norm[norm < 1e-12] = 1.0
        shape = x / norm[:, None]
        shape_cols = [f"shape_{c}" for c in cols]
        for i, col in enumerate(shape_cols):
            out[col] = shape[:, i]
        return out, shape_cols
    raise ValueError(experiment)


def evaluate_split_with_cols(df: pd.DataFrame, cols: list[str], split: dict, task_name: str, classes: list[str], experiment: str) -> tuple[list[dict], list[dict]]:
    task_df = df[df["label"].isin(classes)].copy().reset_index(drop=True)
    train_mask = split["train"](task_df)
    val_mask = split["val"](task_df)
    test_mask = split["test"](task_df)
    train_df = task_df[train_mask].copy()
    val_df = task_df[val_mask].copy()
    test_df = task_df[test_mask].copy()
    if len(train_df) == 0 or len(test_df) == 0:
        return [], []
    if not set(classes).issubset(set(train_df["label"])) or not set(classes).issubset(set(test_df["label"])):
        return [], []
    if len(val_df) > 0 and not set(classes).issubset(set(val_df["label"])):
        return [], []

    x_train_raw = train_df[cols].to_numpy(float)
    x_val_raw = val_df[cols].to_numpy(float)
    x_test_raw = test_df[cols].to_numpy(float)
    x_train, x_val, x_test = ml.standardize(x_train_raw, x_train_raw, x_val_raw, x_test_raw)
    y_train = ml.encode(train_df["label"].to_numpy(str), classes)
    y_val = ml.encode(val_df["label"].to_numpy(str), classes) if len(val_df) else np.array([], dtype=int)
    y_test = ml.encode(test_df["label"].to_numpy(str), classes)

    rows, split_rows = [], []
    for name, part_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        split_rows.append(
            {
                "experiment": experiment,
                "split": split["split"],
                "task": task_name,
                "partition": name,
                "n": len(part_df),
                "class_counts": ml.class_counts(part_df, classes) if len(part_df) else "",
                "period_days": ml.describe_period_days(part_df) if len(part_df) else "",
            }
        )

    for model in ml.MODELS:
        if len(val_df) == 0:
            best_params = ml.DEFAULT_PARAMS[model]
            selection_method = "fixed_default_no_validation"
        else:
            best_params, best_score = None, None
            selection_method = "validation_accuracy_macro_f1_balanced_accuracy"
            for params in ml.GRIDS[model]:
                pred_val = ml.predict(model, x_train, y_train, x_val, params, len(classes))
                val_metrics = ml.metric_dict(y_val, pred_val, len(classes))
                score = (val_metrics["accuracy"], val_metrics["macro_f1"], val_metrics["balanced_accuracy"])
                if best_score is None or score > best_score:
                    best_score = score
                    best_params = params

        pred_train = ml.predict(model, x_train, y_train, x_train, best_params, len(classes))
        pred_val = ml.predict(model, x_train, y_train, x_val, best_params, len(classes)) if len(val_df) else np.array([], dtype=int)
        pred_test = ml.predict(model, x_train, y_train, x_test, best_params, len(classes))
        train_metrics = ml.metric_dict(y_train, pred_train, len(classes))
        val_metrics = ml.metric_dict(y_val, pred_val, len(classes))
        test_metrics = ml.metric_dict(y_test, pred_test, len(classes))

        rows.append(
            {
                "experiment": experiment,
                "feature_dim": len(cols),
                "split": split["split"],
                "task": task_name,
                "model": model,
                "best_params": ";".join(f"{k}={v}" for k, v in best_params.items()),
                "selection_method": selection_method,
                "train_n": len(train_df),
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
                "train_test_accuracy_gap": train_metrics["accuracy"] - test_metrics["accuracy"],
                "fit_assessment": ml.assess_fit(train_metrics["accuracy"], val_metrics["accuracy"], test_metrics["accuracy"]),
            }
        )
    return rows, split_rows


def best_by_split(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, split, task), sub in results.groupby(["experiment", "split", "task"], sort=False):
        r = sub.sort_values(["test_accuracy", "test_macro_f1"], ascending=[False, False]).iloc[0]
        rows.append(r)
    return pd.DataFrame(rows).reset_index(drop=True)


def compare_with_full(control_best: pd.DataFrame, full_results: pd.DataFrame) -> pd.DataFrame:
    full_rows = []
    for (split, task), sub in full_results.groupby(["split", "task"], sort=False):
        r = sub.sort_values(["test_accuracy", "test_macro_f1"], ascending=[False, False]).iloc[0]
        full_rows.append(
            {
                "split": split,
                "task": task,
                "full54_best_model": r["model"],
                "full54_test_accuracy": r["test_accuracy"],
                "full54_test_macro_f1": r["test_macro_f1"],
            }
        )
    full_best = pd.DataFrame(full_rows)
    merged = control_best.merge(full_best, on=["split", "task"], how="left")
    merged["control_minus_full54_accuracy"] = merged["test_accuracy"] - merged["full54_test_accuracy"]
    return merged


def fmt(v: float) -> str:
    return "NA" if pd.isna(v) else f"{float(v):.3f}"


def short_split(split: str) -> str:
    return split.split("_", 1)[0]


def task_label(task: str) -> str:
    return "3-class" if task == "three_class_with_air" else "binary"


def write_svg(comparison: pd.DataFrame, path: Path) -> None:
    ordered_splits = [s["split"] for s in ml.SPLITS]
    rows = []
    for split in ordered_splits:
        for task in ["three_class_with_air", "binary_without_air"]:
            sub = comparison[comparison["split"].eq(split) & comparison["task"].eq(task)]
            if sub.empty:
                continue
            full_acc = float(sub["full54_test_accuracy"].iloc[0])
            amp = sub[sub["experiment"].eq("amplitude_only")].iloc[0]
            shape = sub[sub["experiment"].eq("shape_normalized")].iloc[0]
            rows.append((split, task, full_acc, amp, shape))

    width = 1540
    row_h = 38
    header_h = 46
    height = 170 + header_h + row_h * len(rows) + 150
    cols = [
        ("Split", 64),
        ("Task", 78),
        ("Full 54 best", 112),
        ("Full 54 Acc", 82),
        ("Amplitude best", 128),
        ("Amp Acc", 72),
        ("Amp F1", 72),
        ("Shape best", 128),
        ("Shape Acc", 78),
        ("Shape F1", 72),
        ("Shape-Full", 86),
        ("Interpretation", 475),
    ]
    x0, y0 = 24, 92
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="34" font-family="Arial" font-size="23" font-weight="700">Concentration / Magnitude Control Experiments</text>',
        '<text x="24" y="58" font-family="Arial" font-size="13" fill="#555">Amplitude-only uses one global response-energy feature. Shape-normalized uses the 54-dimensional vector divided by each trial L2 norm. Same splits and five ML models are used.</text>',
    ]
    x = x0
    for label, w in cols:
        lines.append(f'<rect x="{x}" y="{y0}" width="{w}" height="{header_h}" fill="#f1f4f8" stroke="#d7dde7"/>')
        lines.append(f'<text x="{x + 6}" y="{y0 + 27}" font-family="Arial" font-size="11" font-weight="700">{label}</text>')
        x += w
    y = y0 + header_h
    for split, task, full_acc, amp, shape in rows:
        shape_delta = float(shape["control_minus_full54_accuracy"])
        amp_acc = float(amp["test_accuracy"])
        shape_acc = float(shape["test_accuracy"])
        if amp_acc >= full_acc - 0.03:
            interp = "Magnitude alone is close to full model; strong concentration-risk."
            fill = "#fff1dc"
        elif shape_acc >= amp_acc + 0.10:
            interp = "Shape-normalized features outperform amplitude-only; response pattern matters."
            fill = "#e8f5eb"
        else:
            interp = "Amplitude explains part of the result; shape evidence is moderate."
            fill = "#ffffff"
        vals = [
            short_split(split),
            task_label(task),
            str(shape["full54_best_model"]),
            fmt(full_acc),
            str(amp["model"]),
            fmt(amp_acc),
            fmt(amp["test_macro_f1"]),
            str(shape["model"]),
            fmt(shape_acc),
            fmt(shape["test_macro_f1"]),
            fmt(shape_delta),
            interp,
        ]
        x = x0
        for (label, w), val in zip(cols, vals):
            cell_fill = fill if label == "Interpretation" else "white"
            lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{row_h}" fill="{cell_fill}" stroke="#d7dde7"/>')
            lines.append(f'<text x="{x + 6}" y="{y + 23}" font-family="Arial" font-size="10">{str(val)[:68]}</text>')
            x += w
        y += row_h

    summary = comparison.groupby(["experiment", "task"])["test_accuracy"].mean().reset_index()
    y += 26
    lines.append(f'<text x="24" y="{y}" font-family="Arial" font-size="15" font-weight="700">Mean Test Accuracy</text>')
    y += 24
    for _, r in summary.iterrows():
        lines.append(
            f'<text x="24" y="{y}" font-family="Arial" font-size="12" fill="#444">{r["experiment"]} / {task_label(r["task"])}: {float(r["test_accuracy"]):.3f}</text>'
        )
        y += 18
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(LEGACY_FEATURE_PATH)
    original_cols = legacy_feature_columns(df)
    if len(original_cols) != 54:
        raise RuntimeError(f"Expected 54 legacy features, found {len(original_cols)}")

    all_rows, all_splits = [], []
    for experiment in EXPERIMENTS:
        exp_df, exp_cols = add_control_features(df, original_cols, experiment)
        for split in ml.SPLITS:
            for task_name, classes in ml.TASKS:
                rows, split_rows = evaluate_split_with_cols(exp_df, exp_cols, split, task_name, classes, experiment)
                all_rows.extend(rows)
                all_splits.extend(split_rows)

    results = pd.DataFrame(all_rows)
    split_info = pd.DataFrame(all_splits)
    best = best_by_split(results)
    full_results = pd.read_csv(FULL_ML_PATH)
    comparison = compare_with_full(best, full_results)

    results.to_csv(TABLE_DIR / "concentration_control_all_results.csv", index=False)
    best.to_csv(TABLE_DIR / "concentration_control_best_results.csv", index=False)
    comparison.to_csv(TABLE_DIR / "concentration_control_comparison.csv", index=False)
    split_info.to_csv(TABLE_DIR / "concentration_control_splits.csv", index=False)
    write_svg(comparison, FIG_DIR / "concentration_control_summary.svg")

    print("Feature columns:", len(original_cols))
    print("\nBest comparison")
    print(
        comparison[
            [
                "experiment",
                "split",
                "task",
                "model",
                "test_accuracy",
                "test_macro_f1",
                "full54_best_model",
                "full54_test_accuracy",
                "control_minus_full54_accuracy",
            ]
        ].to_string(index=False)
    )
    print("\nMean accuracy")
    print(comparison.groupby(["experiment", "task"])["test_accuracy"].mean().to_string())
    print(FIG_DIR / "concentration_control_summary.svg")


if __name__ == "__main__":
    main()
