from __future__ import annotations

import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import run_dann_multi_splits as dann
import run_manifest_da_baseline_splits as ml
import shared_experiment_protocol as protocol


def require_equal(left, right, message: str) -> None:
    if left != right:
        raise AssertionError(f"{message}: {left!r} != {right!r}")


def require_same_mask(left: pd.Series, right: pd.Series, message: str) -> None:
    if not left.equals(right):
        mismatch = int((left != right).sum())
        raise AssertionError(f"{message}: {mismatch} membership differences")


def audit_constants() -> None:
    require_equal(tuple(ml.RANDOM_SEEDS), protocol.MODEL_SEEDS, "ML seed policy")
    require_equal(ml.FINAL_TRAINING, protocol.ML_FINAL_TRAINING, "ML final training policy")
    require_equal(tuple(dann.SEEDS), protocol.MODEL_SEEDS, "DANN-family seed policy")
    require_equal(tuple(ml.BATCH_ORDER), tuple(protocol.BATCH_ORDER), "ML batch order")
    require_equal(ml.GRIDS["Random Forest"], protocol.RF_GRID, "ML RF search grid")
    require_equal(ml.DEFAULT_PARAMS["Random Forest"], protocol.RF_DEFAULT, "ML RF default")
    require_equal(protocol.DCAE_UCI_REFERENCE_SEED, 0, "YanKe DCAE UCI reference seed")
    expected_rf = {
        "n_estimators": protocol.RF_N_ESTIMATORS,
        "criterion": protocol.RF_CRITERION,
        **protocol.RF_DEFAULT,
        "class_weight": "balanced",
        "n_jobs": protocol.RF_N_JOBS,
    }
    ml_rf_pipeline = ml.make_estimator("Random Forest", protocol.RF_DEFAULT, protocol.MODEL_SEEDS[0])
    if not isinstance(ml_rf_pipeline.named_steps["scaler"], StandardScaler):
        raise AssertionError("ML estimator Pipeline must start with StandardScaler.")
    ml_rf = ml_rf_pipeline.named_steps["classifier"]
    for name, expected in expected_rf.items():
        require_equal(getattr(ml_rf, name), expected, f"ML RF {name}")
    require_equal(ml_rf.bootstrap, protocol.RF_BOOTSTRAP, "ML RF bootstrap bagging")
    ml_mlp = ml.make_estimator("MLP", ml.DEFAULT_PARAMS["MLP"], protocol.MODEL_SEEDS[0]).named_steps["classifier"]
    require_equal(ml_mlp.solver, protocol.MLP_SOLVER, "ML MLP solver")
    require_equal(ml_mlp.early_stopping, protocol.MLP_EARLY_STOPPING, "ML MLP early stopping")
    require_equal(dann.EPOCHS, protocol.NEURAL_EPOCHS, "neural epoch count")
    require_equal(dann.LEARNING_RATE, protocol.NEURAL_LEARNING_RATE, "neural learning rate")
    require_equal(dann.WEIGHT_DECAY, protocol.NEURAL_WEIGHT_DECAY, "neural weight decay")
    require_equal(
        ml.DEFAULT_PARAMS["MLP"]["hidden_layer_sizes"],
        protocol.NEURAL_HIDDEN_DIMS,
        "ML MLP topology",
    )

    model = dann.DANN(n_features=54, n_classes=3)
    layers = list(model.feature_extractor.net)
    require_equal(layers[0].in_features, 54, "G_f input dimension")
    require_equal(layers[0].out_features, protocol.NEURAL_HIDDEN_DIMS[0], "G_f first layer")
    require_equal(layers[2].p, protocol.NEURAL_DROPOUT, "G_f dropout")
    require_equal(layers[3].out_features, protocol.NEURAL_HIDDEN_DIMS[1], "latent dimension")
    if not isinstance(layers[1], nn.ReLU) or not isinstance(layers[4], nn.ReLU):
        raise AssertionError("G_f must use ReLU after both linear layers.")


def audit_splits(df: pd.DataFrame) -> list[dict]:
    task = df[df["label"].isin(protocol.THREE_CLASSES)].copy().reset_index(drop=True)
    if set(task["label"].unique()) != set(protocol.THREE_CLASSES):
        raise AssertionError("The feature table does not contain exactly the required three gas labels.")
    target_all = protocol.target_period_mask(task)
    rows = []
    for da_split, ml_split in zip(protocol.DA_SPLITS, protocol.ML_SPLITS, strict=True):
        require_equal(da_split["split"], ml_split["split"], "split name")
        source_train = da_split["source_train"](task)
        source_val = da_split["source_val"](task)
        target_labeled = da_split["target_labeled_train"](task)
        target_domain = da_split["target_domain"](task)
        target_val = da_split["target_val"](task)
        target_test = da_split["target_test"](task)
        ml_train = ml_split["train"](task)
        ml_val = ml_split["val"](task)
        ml_test = ml_split["test"](task)
        standardization_ref = ml_split["standardization_ref"](task)

        require_same_mask(ml_train, source_train | target_labeled, f"{da_split['split']} train")
        require_same_mask(ml_val, source_val | target_val, f"{da_split['split']} validation")
        require_same_mask(ml_test, target_test, f"{da_split['split']} test")
        require_same_mask(standardization_ref, source_train, f"{da_split['split']} scaler reference")
        require_same_mask(target_domain, target_all, f"{da_split['split']} target-domain input")

        if (ml_train & ml_val).any() or (ml_train & ml_test).any() or (ml_val & ml_test).any():
            raise AssertionError(f"{da_split['split']} has overlapping train/validation/test membership.")
        if (standardization_ref & ~ml_train).any():
            raise AssertionError(f"{da_split['split']} scaler reference is not a training subset.")
        if (target_labeled & (target_val | target_test)).any() or (target_val & target_test).any():
            raise AssertionError(f"{da_split['split']} reuses target gas labels across partitions.")

        rows.append(
            {
                "split": da_split["split"],
                "train_n": int(ml_train.sum()),
                "validation_n": int(ml_val.sum()),
                "test_n": int(ml_test.sum()),
                "source_standardization_n": int(standardization_ref.sum()),
                "target_domain_n": int(target_domain.sum()),
            }
        )
    return rows


def audit_per_batch_splits(df: pd.DataFrame) -> list[dict]:
    task = df[df["label"].isin(protocol.THREE_CLASSES)].copy().reset_index(drop=True)
    require_equal(
        len(ml.PERIOD_I_TRAINED_BATCH_TEST_SPLITS),
        len(protocol.PERIOD_I_PER_BATCH_SPLITS),
        "PB split count",
    )
    rows = []
    reference_train = None
    reference_validation = None
    for batch, shared_split, ml_split in zip(
        protocol.BATCH_ORDER,
        protocol.PERIOD_I_PER_BATCH_SPLITS,
        ml.PERIOD_I_TRAINED_BATCH_TEST_SPLITS,
        strict=True,
    ):
        require_equal(shared_split["split"], ml_split["split"], "PB split name")
        shared_train = shared_split["train"](task)
        shared_validation = shared_split["val"](task)
        shared_test = shared_split["test"](task)
        require_same_mask(shared_train, ml_split["train"](task), f"{shared_split['split']} train")
        require_same_mask(shared_validation, ml_split["val"](task), f"{shared_split['split']} validation")
        require_same_mask(shared_test, ml_split["test"](task), f"{shared_split['split']} test")
        require_same_mask(shared_test, task["batch"].eq(batch), f"{shared_split['split']} batch membership")
        if (shared_train & shared_validation).any() or (shared_train & shared_test).any() or (shared_validation & shared_test).any():
            raise AssertionError(f"{shared_split['split']} has overlapping train/validation/test membership.")
        if reference_train is None:
            reference_train = shared_train
            reference_validation = shared_validation
        else:
            require_same_mask(shared_train, reference_train, f"{shared_split['split']} fixed train")
            require_same_mask(shared_validation, reference_validation, f"{shared_split['split']} fixed validation")
        rows.append(
            {
                "split": shared_split["split"],
                "train_n": int(shared_train.sum()),
                "validation_n": int(shared_validation.sum()),
                "test_n": int(shared_test.sum()),
            }
        )
    return rows


def main() -> None:
    audit_constants()
    frame = pd.read_csv(dann.FEATURE_PATH)
    rows = audit_splits(frame)
    pb_rows = audit_per_batch_splits(frame)
    print(protocol.PROTOCOL_VERSION)
    print(f"ml_protocol={protocol.ML_PROTOCOL_VERSION}")
    print(f"ml_final_training={protocol.ML_FINAL_TRAINING}")
    print(f"rf_bootstrap_bagging={protocol.RF_BOOTSTRAP}")
    print(f"mlp_solver={protocol.MLP_SOLVER}")
    print(f"classes={','.join(protocol.THREE_CLASSES)}")
    print(f"model_seeds={','.join(str(seed) for seed in protocol.MODEL_SEEDS)}")
    print(f"split_seed={protocol.SPLIT_SEED}")
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(pb_rows).to_string(index=False))
    print("AUDIT PASSED")


if __name__ == "__main__":
    main()
