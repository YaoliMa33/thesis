from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(r"D:\thesis")
TABLE_DIR = ROOT / "tables"
OUT_PATH = ROOT / "figures" / "dann_learning" / "dann_three_class_selected_results.pdf"

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


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Summary table is missing required columns: {missing}")


def load_results() -> pd.DataFrame:
    missing = [path for path in SUMMARY_PATHS if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing summary CSV files: {missing}")

    frames = [pd.read_csv(path) for path in SUMMARY_PATHS]
    df = pd.concat(frames, ignore_index=True, sort=False)
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

    if "conditional_alignment_strength" in df.columns:
        df["alpha_lc"] = df["conditional_alignment_strength"]
    else:
        df["alpha_lc"] = np.nan

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


def split_label(split: str) -> str:
    tag = split.split("_")[0]
    return f"{tag} - {split.removeprefix(tag + '_')}"


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def build_table(df: pd.DataFrame) -> Table:
    styles = getSampleStyleSheet()
    cell = ParagraphStyle(
        "Cell",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=6.9,
        leading=8.4,
        alignment=TA_CENTER,
    )
    left_cell = ParagraphStyle(
        "LeftCell",
        parent=cell,
        alignment=TA_LEFT,
    )
    split_style = ParagraphStyle(
        "SplitHeader",
        parent=cell,
        fontName="Helvetica-Bold",
        fontSize=8.2,
        leading=10,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#1f2937"),
    )

    headers = [
        "Split",
        "Variant",
        "lambda",
        "alpha Lc",
        "Seed",
        "Train acc",
        "Val acc",
        "Test acc",
        "Test BA",
        "Macro-F1",
        "Domain acc",
    ]
    rows: list[list[Paragraph | str]] = [[paragraph(h, cell) for h in headers]]
    row_styles: list[tuple] = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]

    current_row = 1
    for split in SPLIT_ORDER:
        split_df = df[df["split"].eq(split)]
        if split_df.empty:
            continue

        rows.append([paragraph(split_label(split), split_style)] + [""] * (len(headers) - 1))
        row_styles.extend(
            [
                ("SPAN", (0, current_row), (-1, current_row)),
                ("BACKGROUND", (0, current_row), (-1, current_row), colors.HexColor("#eef2f7")),
                ("LINEABOVE", (0, current_row), (-1, current_row), 1.2, colors.white),
                ("BOTTOMPADDING", (0, current_row), (-1, current_row), 5),
                ("TOPPADDING", (0, current_row), (-1, current_row), 6),
            ]
        )
        current_row += 1

        best_test = split_df["test_accuracy"].max()
        for _, row in split_df.iterrows():
            test_text = fmt(row["test_accuracy"])
            if row["test_accuracy"] == best_test:
                test_text = f"<b>{test_text}</b>"
            rows.append(
                [
                    paragraph(str(row["split"]).split("_")[0], cell),
                    paragraph(str(row["variant"]), left_cell),
                    paragraph(fmt(row["lambda"], 2), cell),
                    paragraph(fmt(row["alpha_lc"], 2), cell),
                    paragraph(str(int(row["seed"])), cell),
                    paragraph(fmt(row["train_accuracy"]), cell),
                    paragraph(fmt(row["validation_accuracy"]), cell),
                    paragraph(test_text, cell),
                    paragraph(fmt(row["test_balanced_accuracy"]), cell),
                    paragraph(fmt(row["test_macro_f1"]), cell),
                    paragraph(fmt(row["domain_discriminator_accuracy"]), cell),
                ]
            )
            current_row += 1

    col_widths = [15 * mm, 43 * mm, 16 * mm, 16 * mm, 13 * mm, 20 * mm, 19 * mm, 20 * mm, 19 * mm, 19 * mm, 22 * mm]
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle(row_styles))
    return table


def build_pdf(df: pd.DataFrame) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT_PATH),
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title="DANN Three-class Selected Results",
        author="Codex",
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#111827"),
    )
    note = ParagraphStyle(
        "Note",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.4,
        leading=10.5,
        textColor=colors.HexColor("#4b5563"),
    )

    story = [
        Paragraph("DANN Learning - Three-class Selected Results", title),
        Spacer(1, 4 * mm),
        Paragraph(
            "Task: air / alcohol / acetone only. Rows are grouped by split; the highest test accuracy within each split is bold. "
            "Validation accuracy is NA for splits without validation-based model selection.",
            note,
        ),
        Spacer(1, 5 * mm),
        build_table(df),
    ]
    doc.build(story)


def main() -> None:
    df = load_results()
    build_pdf(df)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
