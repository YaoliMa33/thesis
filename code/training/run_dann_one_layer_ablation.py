"""Run a one-hidden-layer DANN architecture ablation without modifying DANN outputs.

Everything except G_f depth is inherited from run_dann_multi_splits.py. Results
are written to separate directories so the established two-layer experiment is
preserved unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch.nn as nn

import run_dann_multi_splits as base
import shared_experiment_protocol as protocol


TABLE_DIR = Path(r"D:\thesis\tables\dann_one_layer")
FIG_DIR = Path(r"D:\thesis\figures\dann_one_layer")
ARCHITECTURE = "54 -> 16 -> ReLU -> Dropout(0.05)"


class OneLayerFeatureExtractor(nn.Module):
    """Single learned projection with the same 16D latent output as base DANN."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, protocol.NEURAL_HIDDEN_DIMS[1]),
            nn.ReLU(),
            nn.Dropout(protocol.NEURAL_DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


def add_architecture_metadata(path: Path) -> None:
    frame = pd.read_csv(path)
    frame.insert(0, "feature_extractor_layers", 1)
    frame.insert(1, "feature_extractor_architecture", ARCHITECTURE)
    frame.to_csv(path, index=False)


def main() -> None:
    original_extractor = base.FeatureExtractor
    original_table_dir = base.TABLE_DIR
    original_figure_dir = base.FIG_DIR
    try:
        base.FeatureExtractor = OneLayerFeatureExtractor
        base.TABLE_DIR = TABLE_DIR
        base.FIG_DIR = FIG_DIR
        base.main()
    finally:
        base.FeatureExtractor = original_extractor
        base.TABLE_DIR = original_table_dir
        base.FIG_DIR = original_figure_dir

    add_architecture_metadata(TABLE_DIR / "dann_multi_split_all_runs.csv")
    add_architecture_metadata(TABLE_DIR / "dann_multi_split_selected_summary.csv")
    metadata = pd.DataFrame(
        [{
            "feature_extractor_layers": 1,
            "feature_extractor_architecture": ARCHITECTURE,
            "latent_dimension": protocol.NEURAL_HIDDEN_DIMS[1],
            "epochs": base.EPOCHS,
            "learning_rate": base.LEARNING_RATE,
            "weight_decay": base.WEIGHT_DECAY,
            "seeds": ",".join(str(seed) for seed in base.SEEDS),
            "lambda_candidates": ",".join(str(value) for value in base.LAMBDA_VALUES),
            "base_implementation": str(Path(base.__file__).resolve()),
        }]
    )
    metadata.to_csv(TABLE_DIR / "architecture_metadata.csv", index=False)


if __name__ == "__main__":
    main()
