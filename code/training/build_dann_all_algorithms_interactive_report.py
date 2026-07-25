"""Assemble the interactive DANN-family dashboard from existing experiment artifacts.

This script does not train models. Run build_dann_interactive_report.py first to
create representative-seed curves and before/after PCA rows for one-/two-layer
MLP and DANN. Existing C-DANN/CDAN artifacts are then merged into the same page.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import build_dann_interactive_report as report


ROOT = Path(r"D:\thesis")
TABLE_DIR = ROOT / "tables"
OUTPUT_PATH = ROOT / "figures" / "dann_learning" / "dann_9feature_cdann_report.html"

SUMMARY_PATHS = [
    TABLE_DIR / "dann_multi_split_selected_summary.csv",
    TABLE_DIR / "dann_one_layer" / "dann_multi_split_selected_summary.csv",
    TABLE_DIR / "dann_conditional_selected_summary.csv",
    TABLE_DIR / "cdan_hybrid_selected_summary.csv",
]
CURVE_PATHS = [
    TABLE_DIR / "dann_interactive_training_curves.csv",
    TABLE_DIR / "dann_conditional_training_curves.csv",
    TABLE_DIR / "cdan_hybrid_training_curves.csv",
]
FEATURE_PATHS = [
    TABLE_DIR / "dann_interactive_feature_space.csv",
    TABLE_DIR / "dann_conditional_feature_space.csv",
    TABLE_DIR / "cdan_hybrid_feature_space.csv",
]


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required report artifact is missing: {path}")
    return pd.read_csv(path)


def with_architecture(frame: pd.DataFrame, architecture: str) -> pd.DataFrame:
    frame = frame.copy()
    if "architecture" not in frame:
        frame["architecture"] = architecture
    else:
        frame["architecture"] = frame["architecture"].fillna(architecture)
    return frame


def normalize_space_names(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "space" not in frame:
        raise ValueError("Feature-space artifact lacks the 'space' column.")
    latent = frame["space"].astype(str).str.contains("latent", case=False, na=False)
    raw = frame["space"].astype(str).str.contains("raw|54D|Before DA", case=False, na=False)
    frame.loc[latent, "space"] = "After feature extractor: latent 16D PCA"
    frame.loc[raw & ~latent, "space"] = "Before DA: standardized 54D PCA"
    return frame


def add_missing_before_rows(summary: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    before = features[features["space"].eq("Before DA: standardized 54D PCA")]
    rows = [features]
    for _, model in summary.iterrows():
        key_mask = (
            features["architecture"].eq(model["architecture"])
            & features["split"].eq(model["split"])
            & features["task"].eq(model["task"])
            & features["variant"].eq(model["variant"])
            & features["space"].eq("Before DA: standardized 54D PCA")
        )
        if key_mask.any():
            continue
        template = before[
            before["architecture"].eq(model["architecture"])
            & before["split"].eq(model["split"])
            & before["task"].eq(model["task"])
        ]
        if template.empty:
            template = before[
                before["architecture"].eq(report.TWO_LAYER)
                & before["split"].eq(model["split"])
                & before["task"].eq(model["task"])
            ]
        if template.empty:
            raise ValueError(f"No raw PCA template for {model['split']} / {model['task']}.")
        copied = template.copy()
        copied["architecture"] = model["architecture"]
        copied["variant"] = model["variant"]
        rows.append(copied)
    return pd.concat(rows, ignore_index=True, sort=False).drop_duplicates(
        ["architecture", "split", "task", "variant", "space", "target_file", "partition"]
    )


def main() -> None:
    two_summary = with_architecture(read_required(SUMMARY_PATHS[0]), report.TWO_LAYER)
    one_summary = with_architecture(read_required(SUMMARY_PATHS[1]), report.ONE_LAYER)
    conditional_summary = with_architecture(read_required(SUMMARY_PATHS[2]), report.TWO_LAYER)
    cdan_summary = with_architecture(read_required(SUMMARY_PATHS[3]), report.TWO_LAYER)
    summary = pd.concat(
        [two_summary, one_summary, conditional_summary, cdan_summary],
        ignore_index=True,
        sort=False,
    )
    summary = summary[summary["task"].eq("three_class_with_air")].reset_index(drop=True)

    curves = pd.concat(
        [with_architecture(read_required(path), report.TWO_LAYER) for path in CURVE_PATHS],
        ignore_index=True,
        sort=False,
    )
    if not curves["architecture"].eq(report.ONE_LAYER).any():
        raise ValueError(
            "One-layer curves are missing. Run build_dann_interactive_report.py after the one-layer training."
        )

    feature_frames = []
    for path in FEATURE_PATHS:
        feature_frames.append(normalize_space_names(with_architecture(read_required(path), report.TWO_LAYER)))
    features = pd.concat(feature_frames, ignore_index=True, sort=False)
    if not features["architecture"].eq(report.ONE_LAYER).any():
        raise ValueError(
            "One-layer PCA rows are missing. Run build_dann_interactive_report.py after the one-layer training."
        )
    features = add_missing_before_rows(summary, features)

    partitions = read_required(TABLE_DIR / "dann_multi_split_partitions.csv")
    report.HTML_PATH = OUTPUT_PATH
    report.write_html(summary, partitions, curves, features)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
