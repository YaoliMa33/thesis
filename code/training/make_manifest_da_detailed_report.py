from __future__ import annotations

import html
from pathlib import Path

import pandas as pd


OUT_ROOT = Path(r"D:\thesis")
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
ML_FIG_DIR = FIG_DIR / "ml_learning"

RESULTS_PATH = TABLE_DIR / "manifest_da_baseline_split_results.csv"
SPLITS_PATH = TABLE_DIR / "manifest_da_baseline_split_splits.csv"

SPLIT_ORDER = [
    "S1_train_I_test_all",
    "S2_train_I_internal_val_test_all",
    "S3_train_I_internal_20_10_test_all",
    "S4_train_I_val_D1_2",
    "S5_train_I_D1_val_D2_3",
    "S6_train_I_D1_2_val_D3_5",
    "S7_train_I_D1_3_val_D4_5",
]

TASK_ORDER = ["three_class_with_air", "binary_without_air"]

MODEL_ORDER = ["LDA", "Logistic Regression", "kNN", "Random Forest", "RBF LS-SVM", "MLP"]

SPLIT_TITLES = {
    "S1_train_I_test_all": "Split 1: Train Period I only; no validation; test all Period II/III",
    "S2_train_I_internal_val_test_all": "Split 2: Period I holdout validation; train 25 / val 5 for three-class; test all Period II/III",
    "S3_train_I_internal_20_10_test_all": "Split 3: Period I holdout validation; train 20 / val 10 for three-class; test all Period II/III",
    "S4_train_I_val_D1_2": "Split 4: Train Period I; validation Day1-2; test Day3-19",
    "S5_train_I_D1_val_D2_3": "Split 5: Train Period I + Day1; validation Day2-3; test Day4-19",
    "S6_train_I_D1_2_val_D3_5": "Split 6: Train Period I + Day1-2; validation Day3-5; test Day6-19",
    "S7_train_I_D1_3_val_D4_5": "Split 7: Train Period I + Day1-3; validation Day4-5; test Day6-19",
}

TASK_TITLES = {
    "three_class_with_air": "Three-class classification: air / alcohol / acetone",
    "binary_without_air": "Binary classification: alcohol / acetone only",
}


def fmt(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def compact_params(params: str) -> str:
    return (
        str(params)
        .replace("gamma_mult", "gamma")
        .replace("c_value", "C")
        .replace("subspace_mult", "subspace")
        .replace("hidden", "h")
    )


def make_judgment(row: pd.Series, is_best: bool) -> str:
    assessment = str(row["fit_assessment"])
    if assessment == "Possible overfitting or domain shift":
        label = "Overfitting/domain shift"
    elif assessment == "Possible underfitting":
        label = "Underfitting"
    elif assessment == "High train-test gap":
        label = "High gap"
    elif assessment == "Validation may be over-optimistic":
        label = "Optimistic validation"
    else:
        label = "Acceptable"
    if is_best:
        return f"Best; {label}"
    return label


def build_detailed_table(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in TASK_ORDER:
        task_df = results[results["task"].eq(task)].copy()
        for split in SPLIT_ORDER:
            split_df = task_df[task_df["split"].eq(split)].copy()
            if split_df.empty:
                continue
            best_acc = split_df["test_accuracy"].max()
            for model in MODEL_ORDER:
                sub = split_df[split_df["model"].eq(model)]
                if sub.empty:
                    continue
                row = sub.iloc[0]
                is_best = abs(float(row["test_accuracy"]) - float(best_acc)) < 1e-12
                rows.append(
                    {
                        "Task": TASK_TITLES[task],
                        "Split": SPLIT_TITLES[split],
                        "Model": row["model"],
                        "Final training": row.get("final_training", "train_only"),
                        "Best hyperparameters": compact_params(row["best_params"]),
                        "Selection": row["selection_method"],
                        "Train n": int(row["train_n"]),
                        "Validation n": int(row["validation_n"]),
                        "Test n": int(row["test_n"]),
                        "Train Acc": float(row["train_accuracy"]),
                        "Val Acc": row["validation_accuracy"],
                        "Test Acc": float(row["test_accuracy"]),
                        "Train BA": float(row["train_balanced_accuracy"]),
                        "Val BA": row["validation_balanced_accuracy"],
                        "Test BA": float(row["test_balanced_accuracy"]),
                        "Train Macro-F1": float(row["train_macro_f1"]),
                        "Val Macro-F1": row["validation_macro_f1"],
                        "Test Macro-F1": float(row["test_macro_f1"]),
                        "Train-Test Gap": float(row["train_test_accuracy_gap"]),
                        "Val-Test Gap": row["validation_test_accuracy_gap"],
                        "Judgment": make_judgment(row, is_best),
                    }
                )
    return pd.DataFrame(rows)


def cell(lines: list[str], x: float, y: float, w: float, h: float, text: str, *, fill: str = "white", color: str = "#222", bold: bool = False, size: int = 11, anchor: str = "start") -> None:
    lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="#d6dbe3" stroke-width="1"/>')
    tx = x + 7 if anchor == "start" else x + w / 2
    weight = "700" if bold else "400"
    safe = html.escape(str(text))
    lines.append(
        f'<text x="{tx}" y="{y + h / 2 + size / 3 - 1}" text-anchor="{anchor}" '
        f'font-family="Arial" font-size="{size}" font-weight="{weight}" fill="{color}">{safe}</text>'
    )


def judgment_fill(text: str) -> str:
    if "Best" in text:
        return "#dff3e4"
    if "Overfitting" in text or "High gap" in text:
        return "#fde7d7"
    if "Underfitting" in text:
        return "#fff1c7"
    return "#f4f7fb"


def write_task_svg(table: pd.DataFrame, task_title: str, path: Path) -> None:
    sub = table[table["Task"].eq(task_title)].copy()
    final_training = str(sub["Final training"].iloc[0]) if "Final training" in sub.columns and not sub.empty else "train_only"
    final_note = (
        "Method B: after validation-based hyperparameter selection, final models are refit on train+validation when validation exists."
        if final_training == "train_val_refit"
        else "Method A: after validation-based hyperparameter selection, final models are trained on train only."
    )
    split_groups = [s for s in [SPLIT_TITLES[k] for k in SPLIT_ORDER] if not sub[sub["Split"].eq(s)].empty]
    cols = [
        ("Model", 155),
        ("Best hyperparameters", 180),
        ("Train n", 70),
        ("Val n", 60),
        ("Test n", 65),
        ("Train Acc", 78),
        ("Val Acc", 72),
        ("Test Acc", 76),
        ("Train BA", 76),
        ("Val BA", 70),
        ("Test BA", 74),
        ("Train F1", 76),
        ("Val F1", 68),
        ("Test F1", 74),
        ("T-Test Gap", 78),
        ("V-Test Gap", 78),
        ("Judgment", 185),
    ]
    margin_x = 24
    header_h = 34
    row_h = 31
    split_h = 34
    title_h = 86
    width = margin_x * 2 + sum(w for _, w in cols)
    height = title_h + len(split_groups) * (split_h + header_h + len(MODEL_ORDER) * row_h + 14) + 42
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700">{html.escape(task_title)}</text>',
        '<text x="24" y="57" font-family="Arial" font-size="13" fill="#555">Baseline/reference files are included as air. Hyperparameters are selected on validation accuracy, with macro-F1 and balanced accuracy as tie-breakers; Split 1 uses fixed defaults.</text>',
        f'<text x="24" y="76" font-family="Arial" font-size="13" fill="#555">{html.escape(final_note)}</text>',
    ]
    y = title_h
    for split_title in split_groups:
        split_df = sub[sub["Split"].eq(split_title)].copy()
        cell(lines, margin_x, y, width - 2 * margin_x, split_h, split_title, fill="#edf2f7", bold=True, size=13)
        y += split_h
        x = margin_x
        for label, w in cols:
            cell(lines, x, y, w, header_h, label, fill="#f8fafc", bold=True, size=10, anchor="start")
            x += w
        y += header_h
        for _, row in split_df.iterrows():
            row_values = [
                row["Model"],
                row["Best hyperparameters"],
                row["Train n"],
                row["Validation n"],
                row["Test n"],
                fmt(row["Train Acc"]),
                fmt(row["Val Acc"]),
                fmt(row["Test Acc"]),
                fmt(row["Train BA"]),
                fmt(row["Val BA"]),
                fmt(row["Test BA"]),
                fmt(row["Train Macro-F1"]),
                fmt(row["Val Macro-F1"]),
                fmt(row["Test Macro-F1"]),
                fmt(row["Train-Test Gap"]),
                fmt(row["Val-Test Gap"]),
                row["Judgment"],
            ]
            x = margin_x
            for (label, w), value in zip(cols, row_values):
                fill = judgment_fill(str(value)) if label == "Judgment" else "white"
                bold = label == "Judgment" and "Best" in str(value)
                cell(lines, x, y, w, row_h, value, fill=fill, bold=bold, size=9 if label == "Best hyperparameters" else 10)
                x += w
            y += row_h
        y += 14
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ML_FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(RESULTS_PATH)
    detailed = build_detailed_table(results)
    detailed.to_csv(TABLE_DIR / "manifest_da_baseline_detailed_report.csv", index=False)
    write_task_svg(detailed, TASK_TITLES["three_class_with_air"], ML_FIG_DIR / "manifest_da_baseline_detailed_three_class.svg")
    write_task_svg(detailed, TASK_TITLES["binary_without_air"], ML_FIG_DIR / "manifest_da_baseline_detailed_binary.svg")
    print("Wrote detailed report CSV and SVG figures.")
    print(detailed[["Task", "Split", "Model", "Train Acc", "Val Acc", "Test Acc", "Test BA", "Test Macro-F1", "Train-Test Gap", "Judgment"]].to_string(index=False))


if __name__ == "__main__":
    main()
