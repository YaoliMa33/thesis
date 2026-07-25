"""Build thesis-ready comparison figures for the 48D/54D open-source benchmark.

The figure set is intentionally data-derived only: no smoothing, imputation, or
post-hoc corrections are applied. Missing model/split/feature-set combinations
raise explicit errors.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


ROOT = Path(r"D:\thesis")
TABLE_ROOT = ROOT / "tables" / "opensource_da_48_54"
SUMMARY_PATH = TABLE_ROOT / "summary.csv"
MLPLR_PATH = TABLE_ROOT / "posthoc_lr_bottleneck16" / "summary.csv"
OUT_DIR = ROOT / "figures" / "opensource_model_comparison"

FEATURE_SETS = ("48D", "54D")
SPLITS = tuple(f"S{i}" for i in range(1, 8))

MODEL_ORDER = [
    "LR",
    "sklearn MLP",
    "MLPLR",
    "No-DA NN",
    "DANN",
    "CDAN",
    "C-DANN",
    "Deep CORAL",
    "MK-MMD",
    "CDAN+C-DANN",
    "MK-MMD+CDAN",
    "MK-MMD+C-DANN",
    "MK-MMD+CDAN+C-DANN",
]

DA_MODELS = [
    "DANN",
    "CDAN",
    "C-DANN",
    "Deep CORAL",
    "MK-MMD",
    "CDAN+C-DANN",
    "MK-MMD+CDAN",
    "MK-MMD+C-DANN",
    "MK-MMD+CDAN+C-DANN",
]

METHOD_LABELS = {
    "source-only": "No-DA NN",
    "dann": "DANN",
    "cdan": "CDAN",
    "c-dann": "C-DANN",
    "deep-coral": "Deep CORAL",
    "mk-mmd": "MK-MMD",
    "cdan+c-dann": "CDAN+C-DANN",
    "mk-mmd+cdan": "MK-MMD+CDAN",
    "mk-mmd+c-dann": "MK-MMD+C-DANN",
    "mk-mmd+cdan+c-dann": "MK-MMD+CDAN+C-DANN",
}

GROUP_COLORS = {
    "classical ML": "#d55e00",
    "MLPLR probe": "#0072b2",
    "neural baseline": "#009e73",
    "domain adaptation": "#6a3d9a",
}


def split_short(value: str) -> str:
    return str(value).split("_", 1)[0]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def read_results() -> pd.DataFrame:
    require_file(SUMMARY_PATH)
    require_file(MLPLR_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    mlplr = pd.read_csv(MLPLR_PATH)

    records: list[dict] = []
    for row in summary.to_dict("records"):
        feature_set = row["feature_set"]
        short = split_short(row["split"])
        if feature_set not in FEATURE_SETS or short not in SPLITS:
            continue
        label = None
        group = None
        if row["family"] == "classical-ml" and row["method"] == "logistic-regression":
            label, group = "LR", "classical ML"
        elif row["family"] == "classical-ml" and row["method"] == "mlp":
            label, group = "sklearn MLP", "classical ML"
        elif (
            row["family"] == "domain-adaptation"
            and row["representation"] == "bottleneck16"
            and row["head"] == "mlp"
            and row["method"] in METHOD_LABELS
        ):
            label = METHOD_LABELS[row["method"]]
            group = "neural baseline" if label == "No-DA NN" else "domain adaptation"
        if label is None:
            continue
        records.append(
            {
                "feature_set": feature_set,
                "split": short,
                "model": label,
                "group": group,
                "accuracy": float(row["test_accuracy"]),
                "balanced_accuracy": float(row["test_balanced_accuracy"]),
                "macro_f1": float(row["test_macro_f1"]),
            }
        )

    for row in mlplr.to_dict("records"):
        feature_set = row["feature_set"]
        short = split_short(row["split"])
        if feature_set not in FEATURE_SETS or short not in SPLITS:
            continue
        if row["method"] != "source-only":
            continue
        records.append(
            {
                "feature_set": feature_set,
                "split": short,
                "model": "MLPLR",
                "group": "MLPLR probe",
                "accuracy": float(row["test_accuracy"]),
                "balanced_accuracy": float(row["test_balanced_accuracy"]),
                "macro_f1": float(row["test_macro_f1"]),
            }
        )

    data = pd.DataFrame(records)
    expected = len(FEATURE_SETS) * len(SPLITS) * len(MODEL_ORDER)
    keys = ["feature_set", "split", "model"]
    duplicates = data[data.duplicated(keys, keep=False)]
    if not duplicates.empty:
        raise ValueError(f"Duplicate result rows:\n{duplicates[keys].to_string(index=False)}")
    missing = []
    for feature_set in FEATURE_SETS:
        for split in SPLITS:
            for model in MODEL_ORDER:
                if not ((data["feature_set"] == feature_set) & (data["split"] == split) & (data["model"] == model)).any():
                    missing.append((feature_set, split, model))
    if missing:
        raise ValueError(f"Missing {len(missing)} required result rows: {missing[:20]}")
    if len(data) != expected:
        raise ValueError(f"Expected {expected} rows after filtering, found {len(data)}.")
    return data


def pivot_metric(data: pd.DataFrame, feature_set: str, metric: str = "balanced_accuracy") -> pd.DataFrame:
    sub = data[data["feature_set"] == feature_set]
    table = sub.pivot(index="model", columns="split", values=metric)
    return table.loc[MODEL_ORDER, list(SPLITS)]


def save_current(name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        path = OUT_DIR / f"{name}.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()


def annotate_heatmap(ax: plt.Axes, values: np.ndarray, fmt: str = "{:.1f}", threshold: float | None = None) -> None:
    if threshold is None:
        threshold = float(np.nanmean(values))
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            color = "white" if value > threshold else "#111111"
            ax.text(x, y, fmt.format(value), ha="center", va="center", fontsize=8, color=color)


def draw_performance_heatmap(data: pd.DataFrame, feature_set: str) -> None:
    table = pivot_metric(data, feature_set) * 100.0
    fig, ax = plt.subplots(figsize=(9.7, 7.0))
    image = ax.imshow(table.values, cmap="viridis", vmin=60, vmax=98, aspect="auto")
    ax.set_title(f"{feature_set} model performance across chronological S1-S7 splits", fontsize=14, weight="bold")
    ax.set_xlabel("Split")
    ax.set_ylabel("Model")
    ax.set_xticks(np.arange(len(SPLITS)), SPLITS)
    ax.set_yticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER)
    annotate_heatmap(ax, table.values, threshold=80)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Balanced accuracy (%)")
    for y, model in enumerate(MODEL_ORDER):
        group = data[(data["model"] == model)]["group"].iloc[0]
        ax.get_yticklabels()[y].set_color(GROUP_COLORS[group])
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.set_xticks(np.arange(-0.5, len(SPLITS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
    save_current(f"performance_heatmap_{feature_set.lower()}_balanced_accuracy")


def draw_feature_delta_heatmap(data: pd.DataFrame) -> None:
    p48 = pivot_metric(data, "48D")
    p54 = pivot_metric(data, "54D")
    delta = (p54 - p48) * 100.0
    limit = float(np.nanmax(np.abs(delta.values)))
    fig, ax = plt.subplots(figsize=(9.7, 7.0))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    image = ax.imshow(delta.values, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_title("Feature-set effect: 54D minus 48D", fontsize=14, weight="bold")
    ax.set_xlabel("Split")
    ax.set_ylabel("Model")
    ax.set_xticks(np.arange(len(SPLITS)), SPLITS)
    ax.set_yticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER)
    for y in range(delta.shape[0]):
        for x in range(delta.shape[1]):
            value = delta.values[y, x]
            ax.text(x, y, f"{value:+.1f}", ha="center", va="center", fontsize=8, color="#111111")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Balanced accuracy change (percentage points)")
    save_current("feature_delta_heatmap_54d_minus_48d_balanced_accuracy")


def draw_da_gain_heatmap(data: pd.DataFrame, feature_set: str) -> None:
    table = pivot_metric(data, feature_set)
    gain = (table.loc[DA_MODELS] - table.loc["No-DA NN"]) * 100.0
    limit = max(1.0, float(np.nanmax(np.abs(gain.values))))
    fig, ax = plt.subplots(figsize=(9.7, 5.7))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    image = ax.imshow(gain.values, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_title(f"{feature_set} domain-adaptation gain over No-DA NN", fontsize=14, weight="bold")
    ax.set_xlabel("Split")
    ax.set_ylabel("DA method")
    ax.set_xticks(np.arange(len(SPLITS)), SPLITS)
    ax.set_yticks(np.arange(len(DA_MODELS)), DA_MODELS)
    for y in range(gain.shape[0]):
        for x in range(gain.shape[1]):
            value = gain.values[y, x]
            ax.text(x, y, f"{value:+.1f}", ha="center", va="center", fontsize=8, color="#111111")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Gain over No-DA NN (percentage points)")
    save_current(f"da_gain_over_source_only_{feature_set.lower()}_balanced_accuracy")


def draw_baseline_gain_panels(data: pd.DataFrame, feature_set: str) -> None:
    table = pivot_metric(data, feature_set)
    rows = ["MLPLR", "No-DA NN"] + DA_MODELS
    baselines = ["LR", "sklearn MLP"]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2), sharey=True)
    all_values = []
    for baseline in baselines:
        all_values.append(((table.loc[rows] - table.loc[baseline]) * 100.0).values)
    limit = max(1.0, float(np.nanmax(np.abs(np.concatenate([v.ravel() for v in all_values])))))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    for ax, baseline in zip(axes, baselines):
        values = (table.loc[rows] - table.loc[baseline]) * 100.0
        image = ax.imshow(values.values, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_title(f"minus {baseline}", fontsize=12, weight="bold")
        ax.set_xlabel("Split")
        ax.set_xticks(np.arange(len(SPLITS)), SPLITS)
        ax.set_yticks(np.arange(len(rows)), rows)
        for y in range(values.shape[0]):
            for x in range(values.shape[1]):
                ax.text(x, y, f"{values.values[y, x]:+.1f}", ha="center", va="center", fontsize=7.5)
    axes[0].set_ylabel("Compared model")
    fig.suptitle(f"{feature_set} improvement over classical ML baselines", fontsize=14, weight="bold")
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cbar.set_label("Balanced accuracy change (percentage points)")
    save_current(f"gain_over_lr_and_sklearn_mlp_{feature_set.lower()}_balanced_accuracy")


def draw_pairwise_mean_delta(data: pd.DataFrame, feature_set: str) -> None:
    table = pivot_metric(data, feature_set)
    means = table.mean(axis=1)
    delta = pd.DataFrame(index=MODEL_ORDER, columns=MODEL_ORDER, dtype=float)
    for row_model in MODEL_ORDER:
        for col_model in MODEL_ORDER:
            delta.loc[row_model, col_model] = (means.loc[row_model] - means.loc[col_model]) * 100.0
    limit = max(1.0, float(np.nanmax(np.abs(delta.values))))
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    image = ax.imshow(delta.values, cmap="RdBu_r", norm=norm, aspect="equal")
    ax.set_title(f"{feature_set} pairwise mean advantage across S1-S7", fontsize=14, weight="bold")
    ax.set_xlabel("Reference model")
    ax.set_ylabel("Compared model")
    ax.set_xticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER)
    for y in range(delta.shape[0]):
        for x in range(delta.shape[1]):
            value = delta.values[y, x]
            label = "0" if abs(value) < 0.05 else f"{value:+.1f}"
            ax.text(x, y, label, ha="center", va="center", fontsize=6.8, color="#111111")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean balanced accuracy advantage (percentage points)")
    save_current(f"pairwise_mean_advantage_{feature_set.lower()}_balanced_accuracy")


def draw_win_count_matrix(data: pd.DataFrame, feature_set: str) -> None:
    table = pivot_metric(data, feature_set)
    wins = pd.DataFrame(index=MODEL_ORDER, columns=MODEL_ORDER, dtype=float)
    for row_model in MODEL_ORDER:
        for col_model in MODEL_ORDER:
            wins.loc[row_model, col_model] = float((table.loc[row_model] > table.loc[col_model]).sum())
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    image = ax.imshow(wins.values, cmap="YlGnBu", vmin=0, vmax=len(SPLITS), aspect="equal")
    ax.set_title(f"{feature_set} split win counts across S1-S7", fontsize=14, weight="bold")
    ax.set_xlabel("Reference model")
    ax.set_ylabel("Compared model")
    ax.set_xticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER)
    for y in range(wins.shape[0]):
        for x in range(wins.shape[1]):
            value = int(wins.values[y, x])
            color = "white" if value >= 5 else "#111111"
            ax.text(x, y, str(value), ha="center", va="center", fontsize=7.5, color=color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Number of splits where row model is higher")
    save_current(f"pairwise_split_win_counts_{feature_set.lower()}_balanced_accuracy")


def draw_feature_slope(data: pd.DataFrame) -> None:
    means = (
        data.groupby(["feature_set", "model"], as_index=False)["balanced_accuracy"]
        .mean()
        .pivot(index="model", columns="feature_set", values="balanced_accuracy")
        .loc[MODEL_ORDER]
        * 100.0
    )
    means["delta"] = means["54D"] - means["48D"]
    means = means.sort_values("delta", ascending=True)
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    y = np.arange(len(means))
    for idx, (_, row) in enumerate(means.iterrows()):
        ax.plot([row["48D"], row["54D"]], [idx, idx], color="#b0b0b0", lw=1.3, zorder=1)
    ax.scatter(means["48D"], y, label="48D", color="#d55e00", s=42, zorder=3)
    ax.scatter(means["54D"], y, label="54D", color="#0072b2", s=42, zorder=3)
    for idx, (model, row) in enumerate(means.iterrows()):
        ax.text(max(row["48D"], row["54D"]) + 0.4, idx, f"{row['delta']:+.1f}", va="center", fontsize=8)
    ax.set_yticks(y, means.index)
    ax.set_xlabel("Mean balanced accuracy across S1-S7 (%)")
    ax.set_title("Feature-set mean effect by model", fontsize=14, weight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    save_current("feature_set_paired_mean_balanced_accuracy")


def draw_rank_heatmap(data: pd.DataFrame, feature_set: str) -> None:
    table = pivot_metric(data, feature_set)
    ranks = table.rank(axis=0, ascending=False, method="min")
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    image = ax.imshow(ranks.values, cmap="magma_r", vmin=1, vmax=len(MODEL_ORDER), aspect="auto")
    ax.set_title(f"{feature_set} model rank by split", fontsize=14, weight="bold")
    ax.set_xlabel("Split")
    ax.set_ylabel("Model")
    ax.set_xticks(np.arange(len(SPLITS)), SPLITS)
    ax.set_yticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER)
    for y in range(ranks.shape[0]):
        for x in range(ranks.shape[1]):
            value = int(ranks.values[y, x])
            color = "white" if value > 7 else "#111111"
            ax.text(x, y, str(value), ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Rank, 1 is best")
    save_current(f"rank_heatmap_{feature_set.lower()}_balanced_accuracy")


def draw_mlplr_context(data: pd.DataFrame, feature_set: str) -> None:
    table = pivot_metric(data, feature_set) * 100.0
    context_models = ["LR", "sklearn MLP", "MLPLR", "No-DA NN", "MK-MMD+CDAN"]
    colors = {
        "LR": "#d55e00",
        "sklearn MLP": "#cc79a7",
        "MLPLR": "#0072b2",
        "No-DA NN": "#009e73",
        "MK-MMD+CDAN": "#6a3d9a",
    }
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    x = np.arange(len(SPLITS))
    for model in context_models:
        ax.plot(x, table.loc[model].values, marker="o", lw=2.0, color=colors[model], label=model)
    ax.set_xticks(x, SPLITS)
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title(f"{feature_set} MLPLR in context", fontsize=14, weight="bold")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    save_current(f"mlplr_context_{feature_set.lower()}_balanced_accuracy")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    data = read_results()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT_DIR / "figure_source_data.csv", index=False)

    for feature_set in FEATURE_SETS:
        draw_performance_heatmap(data, feature_set)
        draw_da_gain_heatmap(data, feature_set)
        draw_baseline_gain_panels(data, feature_set)
        draw_pairwise_mean_delta(data, feature_set)
        draw_win_count_matrix(data, feature_set)
        draw_rank_heatmap(data, feature_set)
        draw_mlplr_context(data, feature_set)
    draw_feature_delta_heatmap(data)
    draw_feature_slope(data)

    figure_paths = sorted(OUT_DIR.glob("*.svg"))
    print(f"Wrote {len(figure_paths)} SVG figures and matching PNG files to {OUT_DIR}")
    for path in figure_paths:
        print(path.name)


if __name__ == "__main__":
    main()
