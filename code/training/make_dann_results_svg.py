from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\thesis")
TABLE_DIR = ROOT / "tables"
OUT_PATH = ROOT / "figures" / "dann_learning" / "dann_three_class_selected_results.svg"

SUMMARY_PATHS = [
    TABLE_DIR / "dann_multi_split_selected_summary.csv",
    TABLE_DIR / "dann_conditional_selected_summary.csv",
    TABLE_DIR / "cdan_hybrid_selected_summary.csv",
]

SPLIT_ORDER = [
    "S1_train_I_test_all",
    "S2_train_I_internal_val_test_all",
    "S3_train_I_internal_20_10_test_all",
    "S4_train_I_val_D1_2",
    "S5_train_I_D1_val_D2_3",
    "S6_train_I_D1_2_val_D3_5",
    "S7_train_I_D1_3_val_D4_5",
]

VARIANT_ORDER = [
    "MLP source only",
    "MLP source + target labels",
    "DANN-UDA",
    "DANN-semi",
    "C-DANN-UDA",
    "C-DANN-semi",
    "CDAN-UDA",
    "CDAN-semi",
    "CDAN+C-DANN-UDA",
    "CDAN+C-DANN-semi",
]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Summary table is missing required columns: {missing}")


def load_results() -> pd.DataFrame:
    missing = [path for path in SUMMARY_PATHS if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing summary CSV files: {missing}")

    df = pd.concat([pd.read_csv(path) for path in SUMMARY_PATHS], ignore_index=True, sort=False)
    require_columns(
        df,
        [
            "split",
            "task",
            "variant",
            "lambda",
            "seed",
            "train_accuracy",
            "validation_accuracy",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_macro_f1",
            "domain_discriminator_accuracy",
        ],
    )
    df = df[df["task"].eq("three_class_with_air")].copy()
    if df.empty:
        raise ValueError("No three_class_with_air rows found in selected summaries.")

    df["alpha_lc"] = df["conditional_alignment_strength"] if "conditional_alignment_strength" in df.columns else np.nan
    df["split_order"] = df["split"].map({name: i for i, name in enumerate(SPLIT_ORDER)})
    if df["split_order"].isna().any():
        unknown = sorted(df.loc[df["split_order"].isna(), "split"].unique())
        raise ValueError(f"Unexpected split names in result table: {unknown}")

    df["variant_order"] = df["variant"].map({name: i for i, name in enumerate(VARIANT_ORDER)})
    if df["variant_order"].isna().any():
        unknown = sorted(df.loc[df["variant_order"].isna(), "variant"].unique())
        raise ValueError(f"Unexpected variant names in result table: {unknown}")

    return df.sort_values(["split_order", "variant_order", "variant"]).reset_index(drop=True)


def fmt(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def fmt_mean_std(row: pd.Series, col: str, digits: int = 3) -> str:
    if col not in row or pd.isna(row[col]):
        return "NA"
    if f"{col}_std" in row and pd.notna(row[f"{col}_std"]):
        return f"{float(row[col]):.{digits}f}+/-{float(row[f'{col}_std']):.{digits}f}"
    return fmt(row[col], digits)


def fmt_seed(row: pd.Series) -> str:
    if "seed_values" in row and pd.notna(row["seed_values"]):
        return str(row["seed_values"])
    return str(int(row["seed"]))


def split_label(split: str) -> str:
    tag = split.split("_")[0]
    return f"{tag} - {split.removeprefix(tag + '_')}"


def text(x: float, y: float, value: object, size: int = 12, weight: int = 400, anchor: str = "middle", color: str = "#111827") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}">{esc(value)}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "#cbd5e1", sw: float = 1) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def write_svg(df: pd.DataFrame) -> None:
    margin_x = 50
    top = 42
    title_h = 82
    row_h = 28
    split_h = 34
    header_h = 30
    bottom = 40
    columns = [
        ("Split", 70, "middle"),
        ("Variant", 270, "start"),
        ("lambda", 86, "middle"),
        ("alpha Lc", 86, "middle"),
        ("Seeds", 70, "middle"),
        ("Train acc", 105, "middle"),
        ("Val acc", 105, "middle"),
        ("Test acc", 105, "middle"),
        ("Test BA", 105, "middle"),
        ("Macro-F1", 105, "middle"),
        ("Domain acc", 115, "middle"),
    ]
    table_w = sum(col[1] for col in columns)
    split_count = sum(1 for split in SPLIT_ORDER if not df[df["split"].eq(split)].empty)
    data_rows = len(df)
    width = table_w + margin_x * 2
    height = top + title_h + header_h + split_count * split_h + data_rows * row_h + bottom

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text(margin_x, top, "DANN Learning - Three-class Selected Results", 27, 700, "start", "#111827"),
        text(
            margin_x,
            top + 38,
            "Task: air / alcohol / acetone only. Rows are grouped by split; the highest test accuracy within each split is bold. Validation accuracy is NA for splits without validation-based model selection.",
            13,
            400,
            "start",
            "#4b5563",
        ),
    ]

    x_positions = [margin_x]
    for _, w, _ in columns[:-1]:
        x_positions.append(x_positions[-1] + w)

    y = top + title_h
    parts.append(rect(margin_x, y, table_w, header_h, "#e5e7eb"))
    for (label, w, anchor), x in zip(columns, x_positions):
        parts.append(rect(x, y, w, header_h, "#e5e7eb"))
        tx = x + (6 if anchor == "start" else w / 2)
        parts.append(text(tx, y + 20, label, 12, 700, anchor, "#111827"))
    y += header_h

    for split in SPLIT_ORDER:
        split_df = df[df["split"].eq(split)]
        if split_df.empty:
            continue

        parts.append(rect(margin_x, y, table_w, split_h, "#eef2f7", "#cbd5e1"))
        parts.append(text(margin_x + 8, y + 22, split_label(split), 14, 700, "start", "#1f2937"))
        y += split_h

        best_test = split_df["test_accuracy"].max()
        for _, row in split_df.iterrows():
            values = [
                str(row["split"]).split("_")[0],
                str(row["variant"]),
                fmt(row["lambda"], 2),
                fmt(row["alpha_lc"], 2),
                fmt_seed(row),
                fmt_mean_std(row, "train_accuracy"),
                fmt_mean_std(row, "validation_accuracy"),
                fmt_mean_std(row, "test_accuracy"),
                fmt_mean_std(row, "test_balanced_accuracy"),
                fmt_mean_std(row, "test_macro_f1"),
                fmt_mean_std(row, "domain_discriminator_accuracy"),
            ]
            for i, ((_, w, anchor), x) in enumerate(zip(columns, x_positions)):
                parts.append(rect(x, y, w, row_h, "#ffffff", "#d1d9e6", 0.8))
                tx = x + (6 if anchor == "start" else w / 2)
                weight = 700 if i == 7 and row["test_accuracy"] == best_test else 400
                parts.append(text(tx, y + 18, values[i], 12, weight, anchor, "#111827"))
            y += row_h

    parts.append("</svg>")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    df = load_results()
    write_svg(df)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
