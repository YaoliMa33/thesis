from __future__ import annotations

import re

import numpy as np
import pandas as pd


PROTOCOL_VERSION = "enose_three_class_shared_v1_2026-06-29"
ML_PROTOCOL_VERSION = "enose_three_class_ml_v2_2026-07-01"
THREE_CLASSES = ["air", "alcohol", "acetone"]
BATCH_ORDER = ["batch1", "batch2", "batch3", "batch4", "batch5"]

# Trial seeds are fixed before training and are experimental repeats, never
# hyperparameters. Five repeats provide a stronger variance estimate than a
# single run while remaining practical for the complete model family.
MODEL_SEEDS = (0, 1, 2, 3, 4)
SPLIT_SEED = 42
SAMPLE_STD_DDOF = 1
DCAE_UCI_REFERENCE_SEED = 0
SEED_POLICY = (
    "fixed trial seeds 0-4; seed 0 matches the YanKe DCAE UCI reference run; "
    "seed controls stochastic training only and is never selected"
)
NO_VALIDATION_SELECTION = "no validation; fixed hyperparameters"

RF_N_ESTIMATORS = 300
RF_CRITERION = "gini"
RF_N_JOBS = -1
# This is the Random Forest bagging mechanism, not bootstrap confidence
# intervals or bootstrap-based metric uncertainty.
RF_BOOTSTRAP = True
RF_GRID = [
    {"max_features": max_features, "max_depth": max_depth, "min_samples_leaf": min_samples_leaf}
    for max_features in ["sqrt", "log2"]
    for max_depth in [2, 3, 4, None]
    for min_samples_leaf in [1, 2, 4]
]
RF_DEFAULT = {"max_features": "sqrt", "max_depth": 4, "min_samples_leaf": 2}

# Shared neural backbone used by the source-only MLP and every neural domain
# adaptation method. Method-specific losses and discriminator heads remain
# separate because they define the algorithms being compared.
NEURAL_HIDDEN_DIMS = (32, 16)
NEURAL_DROPOUT = 0.05
NEURAL_EPOCHS = 400
NEURAL_LEARNING_RATE = 0.01
NEURAL_WEIGHT_DECAY = 1e-4

# Explicit sklearn MLP settings. They are fixed method settings rather than
# hyperparameters selected from validation results.
MLP_ACTIVATION = "relu"
MLP_SOLVER = "adam"
MLP_EARLY_STOPPING = False

# After validation-only hyperparameter selection, refit the complete sklearn
# Pipeline on train + validation. Splits without validation remain train-only.
ML_FINAL_TRAINING = "train_val_refit"


def natural_key(value: str) -> tuple:
    parts = re.split(r"(\d+)", str(value))
    return tuple(int(part) if part.isdigit() else part for part in parts)


def empty_mask(df: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=df.index)


def source_period_mask(df: pd.DataFrame) -> pd.Series:
    return df["period"].eq("Period_I")


def target_period_mask(df: pd.DataFrame) -> pd.Series:
    return ~source_period_mask(df)


def target_days(df: pd.DataFrame, start: int, end: int | None = None) -> pd.Series:
    target = target_period_mask(df)
    if end is None:
        return target & df["day_num"].ge(start)
    return target & df["day_num"].between(start, end)


def period_i_holdout(df: pd.DataFrame, partition: str, scheme: str = "25_5") -> pd.Series:
    if partition not in {"source_train", "source_val"}:
        raise ValueError(f"Unknown Period I partition: {partition}")
    period_i = df[source_period_mask(df)]
    if scheme == "25_5":
        validation_counts = {"air": 1, "alcohol": 2, "acetone": 2}
    elif scheme == "20_10":
        validation_counts = {"air": 2, "alcohol": 4, "acetone": 4}
    else:
        raise ValueError(f"Unknown Period I holdout scheme: {scheme}")
    if not set(validation_counts).issubset(set(period_i["label"])):
        raise ValueError("Period I is missing one or more required three-class labels.")

    mask = empty_mask(df)
    for label, subset in period_i.groupby("label"):
        label_seed = SPLIT_SEED + sum(ord(character) for character in str(label))
        rng = np.random.default_rng(label_seed)
        ordered = sorted(subset.index, key=lambda index: natural_key(df.loc[index, "target_file"]))
        permuted = list(rng.permutation(ordered))
        validation_n = validation_counts[str(label)]
        if len(permuted) <= validation_n:
            raise ValueError(
                f"Period I class {label} has {len(permuted)} samples, insufficient for {validation_n} validation samples."
            )
        validation_indices = set(permuted[:validation_n])
        selected = (
            [index for index in permuted if index not in validation_indices]
            if partition == "source_train"
            else list(validation_indices)
        )
        mask.loc[selected] = True
    return mask


DA_SPLITS = [
    {
        "split": "S1_train_I_test_all",
        "description": "Train Period I only; no validation; test every current later-day sample.",
        "source_train": source_period_mask,
        "source_val": empty_mask,
        "target_labeled_train": empty_mask,
        "target_domain": target_period_mask,
        "target_val": empty_mask,
        "target_test": target_period_mask,
        "selection": NO_VALIDATION_SELECTION,
    },
    {
        "split": "S2_train_I_internal_val_test_all",
        "description": "Train 25 and validate 5 fixed Period I samples; test every current later-day sample.",
        "source_train": lambda df: period_i_holdout(df, "source_train", "25_5"),
        "source_val": lambda df: period_i_holdout(df, "source_val", "25_5"),
        "target_labeled_train": empty_mask,
        "target_domain": target_period_mask,
        "target_val": empty_mask,
        "target_test": target_period_mask,
        "selection": "source_validation_labels",
    },
    {
        "split": "S3_train_I_internal_20_10_test_all",
        "description": "Train 20 and validate 10 fixed Period I samples; test every current later-day sample.",
        "source_train": lambda df: period_i_holdout(df, "source_train", "20_10"),
        "source_val": lambda df: period_i_holdout(df, "source_val", "20_10"),
        "target_labeled_train": empty_mask,
        "target_domain": target_period_mask,
        "target_val": empty_mask,
        "target_test": target_period_mask,
        "selection": "source_validation_labels",
    },
    {
        "split": "S4_train_I_val_D1_2",
        "description": "Train Period I; validate target Day1-Day2; test every target sample from Day3 onward.",
        "source_train": source_period_mask,
        "source_val": empty_mask,
        "target_labeled_train": empty_mask,
        "target_domain": target_period_mask,
        "target_val": lambda df: target_days(df, 1, 2),
        "target_test": lambda df: target_days(df, 3),
        "selection": "target_validation_labels",
    },
    {
        "split": "S5_train_I_D1_val_D2_3",
        "description": "Train Period I plus labeled target Day1; validate Day2-Day3; test Day4 onward.",
        "source_train": source_period_mask,
        "source_val": empty_mask,
        "target_labeled_train": lambda df: target_days(df, 1, 1),
        "target_domain": target_period_mask,
        "target_val": lambda df: target_days(df, 2, 3),
        "target_test": lambda df: target_days(df, 4),
        "selection": "target_validation_labels",
    },
    {
        "split": "S6_train_I_D1_2_val_D3_5",
        "description": "Train Period I plus labeled target Day1-Day2; validate Day3-Day5; test Day6 onward.",
        "source_train": source_period_mask,
        "source_val": empty_mask,
        "target_labeled_train": lambda df: target_days(df, 1, 2),
        "target_domain": target_period_mask,
        "target_val": lambda df: target_days(df, 3, 5),
        "target_test": lambda df: target_days(df, 6),
        "selection": "target_validation_labels",
    },
    {
        "split": "S7_train_I_D1_3_val_D4_5",
        "description": "Train Period I plus labeled target Day1-Day3; validate Day4-Day5; test Day6 onward.",
        "source_train": source_period_mask,
        "source_val": empty_mask,
        "target_labeled_train": lambda df: target_days(df, 1, 3),
        "target_domain": target_period_mask,
        "target_val": lambda df: target_days(df, 4, 5),
        "target_test": lambda df: target_days(df, 6),
        "selection": "target_validation_labels",
    },
]


def _ml_split(da_split: dict) -> dict:
    return {
        "split": da_split["split"],
        "description": da_split["description"],
        "train": lambda df, split=da_split: (
            split["source_train"](df) | split["target_labeled_train"](df)
        ),
        "val": lambda df, split=da_split: split["source_val"](df) | split["target_val"](df),
        "test": lambda df, split=da_split: split["target_test"](df),
        "standardization_ref": lambda df, split=da_split: split["source_train"](df),
        "source_train": lambda df, split=da_split: split["source_train"](df),
        "target_labeled_train": lambda df, split=da_split: split["target_labeled_train"](df),
    }


ML_SPLITS = [_ml_split(split) for split in DA_SPLITS]


PERIOD_I_PER_BATCH_SPLITS = [
    {
        "split": f"PB{index}_train_period_i_internal_val_test_{batch}",
        "description": f"Train fixed Period I subset; validate fixed Period I holdout; test {batch} only.",
        "train": lambda df, _batch=batch: period_i_holdout(df, "source_train", "25_5"),
        "val": lambda df, _batch=batch: period_i_holdout(df, "source_val", "25_5"),
        "test": lambda df, _batch=batch: df["batch"].eq(_batch),
    }
    for index, batch in enumerate(BATCH_ORDER, start=1)
]
