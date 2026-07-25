"""Build explanatory comparison figures for the e-nose thesis benchmark.

These figures emphasize model structure and pairwise visual comparison rather
than tabular heatmap-style result matrices.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(r"D:\thesis")
OPEN_ROOT = ROOT / "tables" / "opensource_da_48_54"
OPEN_SUMMARY = OPEN_ROOT / "summary.csv"
OPEN_MLPLR = OPEN_ROOT / "posthoc_lr_bottleneck16" / "summary.csv"
UNIFIED_ROOT = ROOT / "tables" / "unified_da_source_target"
OUT_DIR = ROOT / "figures" / "opensource_model_explanatory"

FEATURE_SETS = ("48D", "54D")
SPLITS = tuple(f"S{i}" for i in range(1, 8))

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

DA_METHODS = [
    "source-only",
    "dann",
    "cdan",
    "c-dann",
    "deep-coral",
    "mk-mmd",
    "cdan+c-dann",
    "mk-mmd+cdan",
    "mk-mmd+c-dann",
    "mk-mmd+cdan+c-dann",
]

ARCH_METHODS = ["source-only", "dann", "cdan", "c-dann", "deep-coral", "mk-mmd", "cdan+c-dann"]

COLORS = {
    "raw_lr": "#d55e00",
    "raw_mlp": "#cc79a7",
    "mlp_head": "#6a3d9a",
    "mlplr": "#0072b2",
    "one": "#009e73",
    "two": "#984ea3",
    "grey": "#9aa0a6",
}


def split_short(value: str) -> str:
    return str(value).split("_", 1)[0]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def save_current(name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        plt.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close()


def read_open_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_file(OPEN_SUMMARY)
    require_file(OPEN_MLPLR)
    summary = pd.read_csv(OPEN_SUMMARY)
    mlplr = pd.read_csv(OPEN_MLPLR)
    rows: list[dict] = []

    for row in summary.to_dict("records"):
        feature_set = row["feature_set"]
        split = split_short(row["split"])
        if feature_set not in FEATURE_SETS or split not in SPLITS:
            continue
        if row["family"] == "classical-ml" and row["method"] == "logistic-regression":
            rows.append(
                record(row, feature_set, split, "LR", "raw LR", "raw features -> sklearn Logistic Regression")
            )
        elif row["family"] == "classical-ml" and row["method"] == "mlp":
            rows.append(record(row, feature_set, split, "sklearn MLP", "raw MLP", "raw features -> sklearn MLP"))
        elif (
            row["family"] == "domain-adaptation"
            and row["representation"] == "bottleneck16"
            and row["head"] == "mlp"
            and row["method"] in DA_METHODS
        ):
            label = METHOD_LABELS[row["method"]]
            rows.append(record(row, feature_set, split, label, "MLP head", "bottleneck16 -> neural MLP head"))

    for row in mlplr.to_dict("records"):
        feature_set = row["feature_set"]
        split = split_short(row["split"])
        if feature_set not in FEATURE_SETS or split not in SPLITS or row["method"] not in DA_METHODS:
            continue
        label = f"{METHOD_LABELS[row['method']]} + MLPLR"
        rows.append(record(row, feature_set, split, label, "MLPLR", "frozen bottleneck16 -> sklearn LR"))

    long = pd.DataFrame(rows)
    long.to_csv(OUT_DIR / "explanatory_open_results_source_data.csv", index=False)
    neural = long[long["classifier_strategy"] == "MLP head"].copy()
    probe = long[long["classifier_strategy"] == "MLPLR"].copy()
    return long, neural, probe


def record(row: dict, feature_set: str, split: str, model: str, strategy: str, path: str) -> dict:
    method = row.get("method", model)
    return {
        "feature_set": feature_set,
        "split": split,
        "method": method,
        "model": model,
        "classifier_strategy": strategy,
        "model_path": path,
        "balanced_accuracy": float(row["test_balanced_accuracy"]),
        "balanced_accuracy_std": float(row.get("test_balanced_accuracy_std", np.nan)),
        "accuracy": float(row["test_accuracy"]),
        "macro_f1": float(row["test_macro_f1"]),
    }


def box(ax: plt.Axes, xy: tuple[float, float], text: str, color: str, width: float = 1.9, height: float = 0.58) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        linewidth=1.2,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=10, color="#111111")


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#666666") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            lw=1.2,
            color=color,
            shrinkA=7,
            shrinkB=7,
        )
    )


def draw_model_taxonomy() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Model taxonomy used in the thesis figures", fontsize=16, weight="bold", pad=12)

    box(ax, (1.2, 6.6), "Input features\n48D or 54D", "#333333")
    box(ax, (3.8, 7.1), "Classical ML\nraw feature classifier", COLORS["raw_lr"], 2.2)
    box(ax, (6.8, 7.1), "Open-source-style\nbottleneck16 MLP", COLORS["mlp_head"], 2.5)
    box(ax, (10.0, 7.1), "Adaptation objective\noptional", COLORS["mlp_head"], 2.2)

    box(ax, (3.1, 5.8), "LR\nraw -> LR", COLORS["raw_lr"], 1.6)
    box(ax, (4.7, 5.8), "sklearn MLP\nraw -> MLP", COLORS["raw_mlp"], 1.8)

    box(ax, (6.8, 5.8), "G_f feature extractor\nx -> h(16D)", COLORS["mlp_head"], 2.3)
    box(ax, (6.8, 4.6), "MLP head\nh -> gas class", COLORS["mlp_head"], 2.0)
    box(ax, (9.4, 4.6), "MLPLR\nfreeze h, then LR", COLORS["mlplr"], 2.1)

    methods = [
        ("No-DA NN", 1.8, 2.7),
        ("DANN", 3.4, 2.7),
        ("CDAN", 5.0, 2.7),
        ("C-DANN", 6.6, 2.7),
        ("Deep CORAL", 8.2, 2.7),
        ("MK-MMD", 9.8, 2.7),
        ("Hybrids", 11.0, 2.7),
    ]
    for label, x, y in methods:
        box(ax, (x, y), label, COLORS["mlp_head"], 1.45, 0.52)
        arrow(ax, (10.0, 6.82), (x, y + 0.32), "#777777")

    arrow(ax, (2.1, 6.6), (3.0, 6.9))
    arrow(ax, (2.1, 6.6), (6.0, 6.9))
    arrow(ax, (3.8, 6.8), (3.1, 6.1), COLORS["raw_lr"])
    arrow(ax, (3.8, 6.8), (4.7, 6.1), COLORS["raw_mlp"])
    arrow(ax, (6.8, 6.82), (6.8, 6.1), COLORS["mlp_head"])
    arrow(ax, (6.8, 5.5), (6.8, 4.92), COLORS["mlp_head"])
    arrow(ax, (6.8, 5.5), (9.4, 4.92), COLORS["mlplr"])

    ax.text(
        0.6,
        1.1,
        "Important distinction: DANN/CDAN/C-DANN/CORAL/MK-MMD define how h is trained. "
        "The final classifier can be the neural MLP head or the post-hoc MLPLR probe.",
        fontsize=10,
        color="#333333",
    )
    save_current("model_taxonomy_tree")


def mean_by(data: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    return (
        data[data["feature_set"] == feature_set]
        .groupby(["model", "classifier_strategy"], as_index=False)["balanced_accuracy"]
        .mean()
    )


def draw_strategy_lollipop(data: pd.DataFrame, feature_set: str) -> None:
    means = mean_by(data, feature_set)
    keep = []
    for _, row in means.iterrows():
        strategy = row["classifier_strategy"]
        model = row["model"]
        if strategy in {"raw LR", "raw MLP"}:
            keep.append(True)
        elif strategy == "MLP head" and model in [METHOD_LABELS[m] for m in DA_METHODS]:
            keep.append(True)
        elif strategy == "MLPLR" and model.endswith("+ MLPLR"):
            keep.append(True)
        else:
            keep.append(False)
    means = means[keep].copy()
    means["plot_label"] = means["model"].str.replace(" + MLPLR", "", regex=False)
    means["score"] = means["balanced_accuracy"] * 100.0
    means["strategy_order"] = means["classifier_strategy"].map(
        {"raw LR": 0, "raw MLP": 1, "MLP head": 2, "MLPLR": 3}
    )
    means = means.sort_values(["strategy_order", "score"], ascending=[True, True])

    fig, ax = plt.subplots(figsize=(9.2, 9.8))
    y = np.arange(len(means))
    color_map = {
        "raw LR": COLORS["raw_lr"],
        "raw MLP": COLORS["raw_mlp"],
        "MLP head": COLORS["mlp_head"],
        "MLPLR": COLORS["mlplr"],
    }
    ax.hlines(y, means["score"].min() - 1.0, means["score"], color="#d0d0d0", lw=1.2)
    for strategy, sub in means.groupby("classifier_strategy", sort=False):
        idx = sub.index
        ax.scatter(sub["score"], [means.index.get_loc(i) for i in idx], s=52, color=color_map[strategy], label=strategy, zorder=3)
    ax.set_yticks(y, [f"{r.plot_label}  [{r.classifier_strategy}]" for r in means.itertuples()])
    ax.set_xlabel("Mean balanced accuracy across S1-S7 (%)")
    ax.set_title(f"{feature_set} model paths as lollipop points", fontsize=14, weight="bold")
    ax.grid(axis="x", color="#dddddd")
    ax.legend(frameon=False, loc="lower right")
    save_current(f"model_path_lollipop_{feature_set.lower()}")


def draw_head_vs_mlplr_dumbbell(neural: pd.DataFrame, probe: pd.DataFrame, feature_set: str) -> None:
    n = neural[neural["feature_set"] == feature_set].copy()
    p = probe[probe["feature_set"] == feature_set].copy()
    n["base_method"] = n["method"]
    p["base_method"] = p["method"]
    n_mean = n.groupby("base_method", as_index=False)["balanced_accuracy"].mean().rename(columns={"balanced_accuracy": "mlp_head"})
    p_mean = p.groupby("base_method", as_index=False)["balanced_accuracy"].mean().rename(columns={"balanced_accuracy": "mlplr"})
    merged = n_mean.merge(p_mean, on="base_method", how="inner")
    expected = set(DA_METHODS)
    missing = expected - set(merged["base_method"])
    if missing:
        raise ValueError(f"Missing head/probe comparison rows for {feature_set}: {sorted(missing)}")
    merged["label"] = merged["base_method"].map(METHOD_LABELS)
    merged["delta"] = (merged["mlplr"] - merged["mlp_head"]) * 100.0
    merged["mlp_head"] *= 100.0
    merged["mlplr"] *= 100.0
    merged = merged.sort_values("delta", ascending=True)

    fig, ax = plt.subplots(figsize=(8.4, 7.1))
    y = np.arange(len(merged))
    for i, row in enumerate(merged.itertuples()):
        ax.plot([row.mlp_head, row.mlplr], [i, i], color="#b0b0b0", lw=1.4, zorder=1)
    ax.scatter(merged["mlp_head"], y, s=52, color=COLORS["mlp_head"], label="MLP head", zorder=3)
    ax.scatter(merged["mlplr"], y, s=52, color=COLORS["mlplr"], label="MLPLR probe", zorder=3)
    for i, row in enumerate(merged.itertuples()):
        ax.text(max(row.mlp_head, row.mlplr) + 0.35, i, f"{row.delta:+.1f}", va="center", fontsize=9)
    ax.set_yticks(y, merged["label"])
    ax.set_xlabel("Mean balanced accuracy across S1-S7 (%)")
    ax.set_title(f"{feature_set}: neural MLP head vs MLPLR probe", fontsize=14, weight="bold")
    ax.grid(axis="x", color="#dddddd")
    ax.legend(frameon=False, loc="lower right")
    save_current(f"head_vs_mlplr_dumbbell_{feature_set.lower()}")


def draw_head_vs_mlplr_delta(neural: pd.DataFrame, probe: pd.DataFrame, feature_set: str) -> None:
    n = neural[neural["feature_set"] == feature_set].copy()
    p = probe[probe["feature_set"] == feature_set].copy()
    n_table = n.pivot_table(index="method", columns="split", values="balanced_accuracy", aggfunc="first")
    p_table = p.pivot_table(index="method", columns="split", values="balanced_accuracy", aggfunc="first")
    delta = (p_table.loc[DA_METHODS, SPLITS] - n_table.loc[DA_METHODS, SPLITS]) * 100.0
    labels = [METHOD_LABELS[m] for m in DA_METHODS]
    limit = max(1.0, float(np.nanmax(np.abs(delta.values))))
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    image = ax.imshow(delta.values, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    ax.set_xticks(np.arange(len(SPLITS)), SPLITS)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Split")
    ax.set_ylabel("Training method")
    ax.set_title(f"{feature_set}: MLPLR probe minus neural MLP head", fontsize=14, weight="bold")
    for y in range(delta.shape[0]):
        for x in range(delta.shape[1]):
            ax.text(x, y, f"{delta.values[y, x]:+.1f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Balanced accuracy change (percentage points)")
    save_current(f"mlplr_minus_mlp_head_delta_{feature_set.lower()}")


def draw_da_with_classical_lollipop(data: pd.DataFrame, feature_set: str) -> None:
    """Compare DA model paths against LR and sklearn MLP in one dot figure."""
    order = [
        ("LR", "raw LR"),
        ("sklearn MLP", "raw MLP"),
        ("No-DA NN", "MLP head"),
        ("No-DA NN + MLPLR", "MLPLR"),
        ("DANN", "MLP head"),
        ("DANN + MLPLR", "MLPLR"),
        ("CDAN", "MLP head"),
        ("CDAN + MLPLR", "MLPLR"),
        ("C-DANN", "MLP head"),
        ("C-DANN + MLPLR", "MLPLR"),
        ("Deep CORAL", "MLP head"),
        ("Deep CORAL + MLPLR", "MLPLR"),
        ("MK-MMD", "MLP head"),
        ("MK-MMD + MLPLR", "MLPLR"),
        ("CDAN+C-DANN", "MLP head"),
        ("CDAN+C-DANN + MLPLR", "MLPLR"),
        ("MK-MMD+CDAN", "MLP head"),
        ("MK-MMD+CDAN + MLPLR", "MLPLR"),
        ("MK-MMD+C-DANN", "MLP head"),
        ("MK-MMD+C-DANN + MLPLR", "MLPLR"),
        ("MK-MMD+CDAN+C-DANN", "MLP head"),
        ("MK-MMD+CDAN+C-DANN + MLPLR", "MLPLR"),
    ]
    means = (
        data[data["feature_set"] == feature_set]
        .groupby(["model", "classifier_strategy"], as_index=False)["balanced_accuracy"]
        .mean()
    )
    rows = []
    for model, strategy in order:
        match = means[(means["model"] == model) & (means["classifier_strategy"] == strategy)]
        if match.empty:
            raise ValueError(f"Missing comparison row for {feature_set}: {model} [{strategy}]")
        rows.append(
            {
                "label": f"{model.replace(' + MLPLR', '')}  [{strategy}]",
                "strategy": strategy,
                "score": float(match.iloc[0]["balanced_accuracy"]) * 100.0,
            }
        )
    frame = pd.DataFrame(rows).sort_values("score", ascending=True)
    color_map = {
        "raw LR": COLORS["raw_lr"],
        "raw MLP": COLORS["raw_mlp"],
        "MLP head": COLORS["mlp_head"],
        "MLPLR": COLORS["mlplr"],
    }

    fig, ax = plt.subplots(figsize=(9.6, 9.6))
    y = np.arange(len(frame))
    x0 = min(80.0, float(frame["score"].min()) - 1.0)
    ax.hlines(y, x0, frame["score"], color="#d0d0d0", lw=1.1)
    for strategy, sub in frame.groupby("strategy", sort=False):
        positions = [frame.index.get_loc(index) for index in sub.index]
        ax.scatter(sub["score"], positions, s=54, color=color_map[strategy], label=strategy, zorder=3)
    ax.set_yticks(y, frame["label"])
    ax.set_xlabel("Mean balanced accuracy across S1-S7 (%)")
    ax.set_title(f"{feature_set}: DA models compared with LR and sklearn MLP", fontsize=14, weight="bold")
    ax.grid(axis="x", color="#dddddd")
    ax.legend(frameon=False, loc="lower right")
    save_current(f"da_with_lr_sklearn_mlp_lollipop_{feature_set.lower()}")


def draw_da_with_classical_split_lines(data: pd.DataFrame, feature_set: str) -> None:
    """Show classical baselines and selected DA paths across S1-S7."""
    selections = [
        ("LR", "raw LR", "LR", COLORS["raw_lr"], "-"),
        ("sklearn MLP", "raw MLP", "sklearn MLP", COLORS["raw_mlp"], "-"),
        ("No-DA NN", "MLP head", "No-DA NN", "#555555", "--"),
        ("DANN", "MLP head", "DANN", "#9467bd", "-"),
        ("CDAN", "MLP head", "CDAN", "#7b3294", "-"),
        ("C-DANN", "MLP head", "C-DANN", "#542788", "-"),
        ("Deep CORAL", "MLP head", "Deep CORAL", "#1b9e77", "-"),
        ("MK-MMD", "MLP head", "MK-MMD", "#e6ab02", "-"),
        ("MK-MMD+CDAN", "MLP head", "MK-MMD+CDAN", "#a6761d", "-"),
        ("MK-MMD+CDAN + MLPLR", "MLPLR", "MK-MMD+CDAN MLPLR", COLORS["mlplr"], "-"),
    ]
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    x = np.arange(len(SPLITS))
    for model, strategy, label, color, linestyle in selections:
        sub = data[
            (data["feature_set"] == feature_set)
            & (data["model"] == model)
            & (data["classifier_strategy"] == strategy)
        ].copy()
        if len(sub) != len(SPLITS):
            raise ValueError(f"Expected {len(SPLITS)} rows for {feature_set}: {model} [{strategy}], found {len(sub)}")
        sub["split"] = pd.Categorical(sub["split"], categories=SPLITS, ordered=True)
        sub = sub.sort_values("split")
        ax.plot(
            x,
            sub["balanced_accuracy"].to_numpy(float) * 100.0,
            marker="o",
            lw=1.8,
            ms=4.5,
            color=color,
            linestyle=linestyle,
            label=label,
        )
    ax.set_xticks(x, SPLITS)
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_xlabel("Split")
    ax.set_title(f"{feature_set}: classical ML baselines inside DA comparison", fontsize=14, weight="bold")
    ax.grid(axis="y", color="#dddddd")
    ax.legend(frameon=False, ncol=2, fontsize=8.5, loc="lower right")
    save_current(f"da_with_lr_sklearn_mlp_split_lines_{feature_set.lower()}")


def read_architecture_results() -> pd.DataFrame:
    paths = [
        UNIFIED_ROOT / "summary__one-layer__source-only_dann_deep-coral_mk-mmd__uda_semi.csv",
        UNIFIED_ROOT / "summary__one-layer__c-dann_cdan_cdan+c-dann__uda_semi.csv",
        UNIFIED_ROOT / "summary__two-layer__source-only_dann_deep-coral_mk-mmd__uda_semi.csv",
        UNIFIED_ROOT / "summary__two-layer__c-dann_cdan_cdan+c-dann__uda_semi.csv",
    ]
    for path in paths:
        require_file(path)
    rows = []
    for path in paths:
        frame = pd.read_csv(path)
        rows.append(frame)
    data = pd.concat(rows, ignore_index=True)
    data["split_short"] = data["split"].map(split_short)
    data = data[data["split_short"].isin(SPLITS) & data["method"].isin(ARCH_METHODS)].copy()
    return data


def draw_architecture_dumbbell(arch: pd.DataFrame, mode: str) -> None:
    sub = arch[arch["mode"] == mode].copy()
    grouped = (
        sub.groupby(["architecture", "method"], as_index=False)["test_balanced_accuracy"]
        .mean()
        .pivot(index="method", columns="architecture", values="test_balanced_accuracy")
    )
    available = [method for method in ARCH_METHODS if method in grouped.index and {"one-layer", "two-layer"}.issubset(grouped.columns) and grouped.loc[method].notna().all()]
    if not available:
        raise ValueError(f"No complete one/two-layer rows for mode {mode}")
    grouped = grouped.loc[available].copy()
    grouped["label"] = [METHOD_LABELS[m] for m in grouped.index]
    grouped["delta"] = (grouped["two-layer"] - grouped["one-layer"]) * 100.0
    grouped["one-layer"] *= 100.0
    grouped["two-layer"] *= 100.0
    grouped = grouped.sort_values("delta", ascending=True)

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    y = np.arange(len(grouped))
    for i, row in enumerate(grouped.itertuples()):
        ax.plot([getattr(row, "_1"), getattr(row, "_2")], [i, i], color="#b0b0b0", lw=1.4, zorder=1)
    # itertuples mangles column names with hyphens, so use direct columns for scatter.
    ax.scatter(grouped["one-layer"], y, s=52, color=COLORS["one"], label="one-layer 54 -> 16", zorder=3)
    ax.scatter(grouped["two-layer"], y, s=52, color=COLORS["two"], label="two-layer 54 -> 32 -> 16", zorder=3)
    for i, (_, row) in enumerate(grouped.iterrows()):
        ax.text(max(row["one-layer"], row["two-layer"]) + 0.35, i, f"{row['delta']:+.1f}", va="center", fontsize=9)
    ax.set_yticks(y, grouped["label"])
    ax.set_xlabel("Mean balanced accuracy across S1-S7 (%)")
    ax.set_title(f"Architecture comparison ({mode.upper()}): one-layer vs two-layer", fontsize=14, weight="bold")
    ax.grid(axis="x", color="#dddddd")
    ax.legend(frameon=False, loc="lower right")
    save_current(f"one_vs_two_layer_dumbbell_{mode}")


def draw_architecture_delta_by_split(arch: pd.DataFrame, mode: str) -> None:
    sub = arch[arch["mode"] == mode].copy()
    one = sub[sub["architecture"] == "one-layer"].pivot_table(
        index="method", columns="split_short", values="test_balanced_accuracy", aggfunc="first"
    )
    two = sub[sub["architecture"] == "two-layer"].pivot_table(
        index="method", columns="split_short", values="test_balanced_accuracy", aggfunc="first"
    )
    methods = [m for m in ARCH_METHODS if m in one.index and m in two.index and set(SPLITS).issubset(one.columns) and set(SPLITS).issubset(two.columns)]
    if not methods:
        return
    delta = (two.loc[methods, SPLITS] - one.loc[methods, SPLITS]) * 100.0
    limit = max(1.0, float(np.nanmax(np.abs(delta.values))))
    fig, ax = plt.subplots(figsize=(9.3, 4.8))
    image = ax.imshow(delta.values, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    ax.set_xticks(np.arange(len(SPLITS)), SPLITS)
    ax.set_yticks(np.arange(len(methods)), [METHOD_LABELS[m] for m in methods])
    ax.set_xlabel("Split")
    ax.set_ylabel("Method")
    ax.set_title(f"{mode.upper()}: two-layer minus one-layer", fontsize=14, weight="bold")
    for y in range(delta.shape[0]):
        for x in range(delta.shape[1]):
            ax.text(x, y, f"{delta.values[y, x]:+.1f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Balanced accuracy change (percentage points)")
    save_current(f"two_minus_one_layer_delta_{mode}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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

    long, neural, probe = read_open_results()
    draw_model_taxonomy()
    for feature_set in FEATURE_SETS:
        draw_strategy_lollipop(long, feature_set)
        draw_da_with_classical_lollipop(long, feature_set)
        draw_da_with_classical_split_lines(long, feature_set)
        draw_head_vs_mlplr_dumbbell(neural, probe, feature_set)
        draw_head_vs_mlplr_delta(neural, probe, feature_set)

    arch = read_architecture_results()
    arch.to_csv(OUT_DIR / "architecture_source_data.csv", index=False)
    for mode in ("uda", "semi"):
        draw_architecture_dumbbell(arch, mode)
        draw_architecture_delta_by_split(arch, mode)

    print(f"Wrote explanatory figures to {OUT_DIR}")
    for path in sorted(OUT_DIR.glob("*.svg")):
        print(path.name)


if __name__ == "__main__":
    main()
