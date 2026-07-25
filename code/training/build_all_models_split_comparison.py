from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


THESIS_ROOT = Path(r"D:\thesis")
TABLE_DIR = THESIS_ROOT / "tables"
FIGURE_PATH = THESIS_ROOT / "figures" / "ml_learning" / "all_models_s1_s7_ba_macro_f1.svg"

SPLITS = [
    ("S1", "S1_train_I_test_all"),
    ("S2", "S2_train_I_internal_val_test_all"),
    ("S3", "S3_train_I_internal_20_10_test_all"),
    ("S4", "S4_train_I_val_D1_2"),
    ("S5", "S5_train_I_D1_val_D2_3"),
    ("S6", "S6_train_I_D1_2_val_D3_5"),
    ("S7", "S7_train_I_D1_3_val_D4_5"),
]
MODELS = [
    "LDA",
    "Logistic Regression",
    "kNN",
    "Random Forest",
    "RBF SVM",
    "MLP",
    "DANN",
    "CDAN",
    "C-DANN",
    "Deep CORAL",
]
MODEL_COLORS = {
    "LDA": "#0072B2",
    "Logistic Regression": "#D55E00",
    "kNN": "#009E73",
    "Random Forest": "#CC79A7",
    "RBF SVM": "#E69F00",
    "MLP": "#56B4E9",
    "DANN": "#6F42C1",
    "CDAN": "#008B8B",
    "C-DANN": "#9C4A1A",
    "Deep CORAL": "#34495E",
}


def require_one(frame: pd.DataFrame, mask: pd.Series, context: str) -> pd.Series:
    selected = frame[mask]
    if len(selected) != 1:
        raise ValueError(f"{context}: expected exactly one row, found {len(selected)}.")
    return selected.iloc[0]


def load_metrics() -> pd.DataFrame:
    rows: list[dict] = []
    ml = pd.read_csv(TABLE_DIR / "manifest_da_baseline_split_results.csv")
    for split_short, split_full in SPLITS:
        for model in MODELS[:6]:
            row = require_one(
                ml,
                ml["split"].eq(split_full)
                & ml["task"].eq("three_class_with_air")
                & ml["model"].eq(model),
                f"ML {split_short}/{model}",
            )
            rows.append(
                {
                    "split": split_short,
                    "model": model,
                    "test_balanced_accuracy": float(row["test_balanced_accuracy"]),
                    "test_macro_f1": float(row["test_macro_f1"]),
                }
            )

    def add_family(filename: str, model: str, uda_variant: str, semi_variant: str | None = None) -> None:
        frame = pd.read_csv(TABLE_DIR / filename)
        for split_index, (split_short, split_full) in enumerate(SPLITS):
            variant = semi_variant if semi_variant is not None and split_index >= 4 else uda_variant
            row = require_one(
                frame,
                frame["split"].eq(split_full)
                & frame["task"].eq("three_class_with_air")
                & frame["variant"].eq(variant),
                f"{filename} {split_short}/{variant}",
            )
            rows.append(
                {
                    "split": split_short,
                    "model": model,
                    "test_balanced_accuracy": float(row["test_balanced_accuracy"]),
                    "test_macro_f1": float(row["test_macro_f1"]),
                }
            )

    add_family("dann_multi_split_selected_summary.csv", "DANN", "DANN-UDA", "DANN-semi")
    add_family("cdan_multi_split_selected_summary.csv", "CDAN", "CDAN-UDA", "CDAN-semi")
    add_family("dann_conditional_selected_summary.csv", "C-DANN", "C-DANN-UDA", "C-DANN-semi")
    add_family("deep_coral_multi_split_selected_summary.csv", "Deep CORAL", "Deep CORAL-UDA")

    result = pd.DataFrame(rows)
    expected = pd.MultiIndex.from_product(
        [[short for short, _full in SPLITS], MODELS], names=["split", "model"]
    )
    indexed = result.set_index(["split", "model"])
    if indexed.index.has_duplicates:
        raise ValueError("Combined comparison contains duplicate split/model rows.")
    missing = expected.difference(indexed.index)
    if len(missing):
        raise ValueError(f"Combined comparison is incomplete: {list(missing)}")
    ordered = indexed.loc[expected].reset_index()
    metric_values = ordered[["test_balanced_accuracy", "test_macro_f1"]].to_numpy(float)
    if not np.isfinite(metric_values).all():
        raise ValueError("Combined comparison contains missing or infinite test metrics.")
    if (metric_values < 0.5).any() or (metric_values > 1.0).any():
        raise ValueError("The requested 0.5–1.0 axis would clip one or more test metrics.")
    return ordered


def lighten(color: str, strength: float) -> tuple[float, float, float]:
    base = np.asarray(matplotlib.colors.to_rgb(color), dtype=float)
    return tuple(1.0 - (1.0 - base) * strength)


def draw(metrics: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    split_names = [short for short, _full in SPLITS]
    strengths = np.linspace(0.32, 1.0, len(split_names))
    group_centres = np.arange(len(MODELS), dtype=float)
    width = 0.105
    offsets = (np.arange(len(split_names)) - 3.0) * width
    panels = [
        ("test_balanced_accuracy", "Test balanced accuracy (BA)"),
        ("test_macro_f1", "Test macro-F1"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(20.0, 11.2), sharex=True, sharey=True, constrained_layout=True)

    indexed = metrics.set_index(["split", "model"])
    for ax, (metric, title) in zip(axes, panels):
        for model_index, model in enumerate(MODELS):
            for split_index, split in enumerate(split_names):
                value = float(indexed.loc[(split, model), metric])
                x = group_centres[model_index] + offsets[split_index]
                color = lighten(MODEL_COLORS[model], float(strengths[split_index]))
                ax.bar(
                    x,
                    value - 0.5,
                    bottom=0.5,
                    width=width * 0.88,
                    color=color,
                    edgecolor=MODEL_COLORS[model],
                    linewidth=0.5,
                    zorder=3,
                )
                if value >= 0.96:
                    y, va = value - 0.006, "top"
                else:
                    y, va = value + 0.006, "bottom"
                ax.text(
                    x,
                    y,
                    f"{value:.3f}",
                    ha="center",
                    va=va,
                    rotation=90,
                    fontsize=5.8,
                    color="#172033",
                    clip_on=True,
                    zorder=4,
                )
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylim(0.5, 1.0)
        ax.set_yticks(np.arange(0.5, 1.01, 0.1))
        ax.set_ylabel("Test metric value")
        ax.grid(axis="y", color="#D9DEE7", linewidth=0.65)
        ax.set_axisbelow(True)

    labels = [
        "LDA",
        "Logistic\nRegression",
        "kNN",
        "Random\nForest",
        "RBF SVM",
        "MLP",
        "DANN",
        "CDAN",
        "C-DANN",
        "Deep\nCORAL",
    ]
    axes[-1].set_xticks(group_centres, labels=labels)
    axes[-1].set_xlabel("Model; S1–S7 are ordered left-to-right and light-to-dark within each model")
    neutral = "#334155"
    legend = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=lighten(neutral, float(strength)),
            markeredgecolor=neutral,
            label=split,
            markersize=8,
        )
        for split, strength in zip(split_names, strengths)
    ]
    axes[0].legend(
        handles=legend,
        title="Split shade within every model",
        ncol=7,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )
    fig.suptitle("S1–S7 comparison across all core models", fontsize=15, fontweight="bold", y=1.015)
    fig.text(
        0.5,
        -0.01,
        "S1–S4 use UDA variants; S5–S7 use semi variants for DANN, CDAN, and C-DANN. Deep CORAL uses the available UDA result.",
        ha="center",
        fontsize=8,
        color="#4F5D73",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metrics = load_metrics()
    draw(metrics)
    print(FIGURE_PATH)


if __name__ == "__main__":
    main()
