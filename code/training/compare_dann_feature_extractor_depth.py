"""Compare established two-layer DANN results with the one-layer ablation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\thesis")
TWO_LAYER_PATH = ROOT / "tables" / "dann_multi_split_selected_summary.csv"
ONE_LAYER_PATH = ROOT / "tables" / "dann_one_layer" / "dann_multi_split_selected_summary.csv"
OUTPUT_PATH = ROOT / "tables" / "dann_one_vs_two_layer_comparison.csv"


def prepare(path: Path, architecture: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    keep = frame[frame["variant"].isin(["MLP source only", "DANN-UDA", "MLP source + target labels", "DANN-semi"])].copy()
    keep["architecture"] = architecture
    return keep[
        [
            "split", "variant", "architecture", "lambda", "seed_count",
            "train_accuracy", "train_accuracy_std",
            "validation_accuracy", "validation_accuracy_std",
            "test_accuracy", "test_accuracy_std", "test_accuracy_min", "test_accuracy_max",
            "test_balanced_accuracy", "test_balanced_accuracy_std",
            "test_macro_f1", "test_macro_f1_std",
        ]
    ]


def main() -> None:
    two = prepare(TWO_LAYER_PATH, "two-layer G_f: 54->32->16")
    one = prepare(ONE_LAYER_PATH, "one-layer G_f: 54->16")
    combined = pd.concat([two, one], ignore_index=True)
    metrics = [
        "lambda", "seed_count",
        "train_accuracy", "train_accuracy_std",
        "validation_accuracy", "validation_accuracy_std",
        "test_accuracy", "test_accuracy_std", "test_accuracy_min", "test_accuracy_max",
        "test_balanced_accuracy", "test_balanced_accuracy_std",
        "test_macro_f1", "test_macro_f1_std",
    ]
    comparison = combined.pivot(
        index=["split", "variant"],
        columns="architecture",
        values=metrics,
    )
    comparison.columns = [f"{metric}__{architecture}" for metric, architecture in comparison.columns]
    comparison = comparison.reset_index()
    one_accuracy = "test_accuracy__one-layer G_f: 54->16"
    two_accuracy = "test_accuracy__two-layer G_f: 54->32->16"
    comparison["one_minus_two_test_accuracy"] = comparison[one_accuracy] - comparison[two_accuracy]
    comparison["better_architecture_by_mean_test_accuracy"] = comparison["one_minus_two_test_accuracy"].map(
        lambda delta: "one-layer" if delta > 0 else "two-layer" if delta < 0 else "tie"
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_PATH, index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
