"""Fixed-C and refitted Logistic Regression on frozen DA features.

For each architecture/split/variant:
1. reconstruct the selected pre-refit neural model for seeds 0-4;
2. use the pre-specified Logistic Regression value C=1 for every experiment;
3. reconstruct the final neural model with the original refit policy;
4. fit LR with selected C on final labeled latent features and test once.

Strict label boundary: UDA may use source validation only. Target validation
labels are available to Semi/Source+Target only. Logistic Regression never
uses validation or test labels for hyperparameter selection. Final rows are
checkpointed incrementally and are resumable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_unified_da_source_target_suite as suite
import shared_experiment_protocol as protocol


ROOT = Path(r"D:\thesis")
INPUT = ROOT / "tables" / "unified_da_source_target"
OUTPUT = ROOT / "tables" / "fixed_c1_refit_da_logistic_probe"
FEATURE_PATH = ROOT / "tables" / "manifest_baseline_as_air_features.csv"
ARCHITECTURES = ["one-layer", "two-layer"]
METHODS = ["source-only", "dann", "cdan", "c-dann", "mk-mmd", "deep-coral", "cdan+c-dann"]
C_VALUES = [1.0]
RUN_FILES = {
    "one-layer": [
        INPUT / "runs__one-layer__source-only_dann_deep-coral_mk-mmd__uda_semi.csv",
        INPUT / "runs__one-layer__c-dann_cdan_cdan+c-dann__uda_semi.csv",
    ],
    "two-layer": [
        INPUT / "runs__two-layer__source-only_dann_deep-coral_mk-mmd__uda_semi.csv",
        INPUT / "runs__two-layer__c-dann_cdan_cdan+c-dann__uda_semi.csv",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=ARCHITECTURES)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    return parser.parse_args()


def tag(options: argparse.Namespace) -> str:
    return "_".join(options.architectures) + "__" + "_".join(options.methods)


def output_paths(name: str) -> dict[str, Path]:
    return {
        "tuning": OUTPUT / f"tuning__{name}.csv",
        "runs": OUTPUT / f"runs__{name}.csv",
        "predictions": OUTPUT / f"predictions__{name}.csv",
        "summary": OUTPUT / f"summary__{name}.csv",
        "protocol": OUTPUT / f"protocol__{name}.json",
    }


def load_selected(options: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for architecture in options.architectures:
        for path in RUN_FILES[architecture]:
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            frame = frame[frame["method"].isin(options.methods)].copy()
            frame["probe_architecture"] = architecture
            frames.append(frame)
    selected = pd.concat(frames, ignore_index=True, sort=False)
    if "adversarial_strength" not in selected:
        selected["adversarial_strength"] = 0.0
    selected["adversarial_strength"] = selected["adversarial_strength"].fillna(0.0)
    return selected.drop_duplicates(["probe_architecture", "split", "variant", "seed"], keep="last")


def read_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def validation_kind(mode: str, initial: dict) -> str | None:
    if mode == "uda" and len(initial["x"]["source_val"]):
        return "source_val"
    if mode == "semi" and len(initial["x"]["target_val"]):
        return "target_val"
    return None


def arrays_for_final(parts: dict, columns: list[str], mode: str, has_source_val: bool) -> tuple[dict, bool]:
    if mode == "semi":
        return suite.make_arrays(parts, columns, ("source_train", "target_labeled_train", "target_val")), True
    if has_source_val:
        return suite.make_arrays(parts, columns, ("source_train", "source_val")), True
    return suite.make_arrays(parts, columns), False


def train_model(row: pd.Series, arrays: dict, include_validation: bool):
    candidate = suite.Candidate(
        float(row["alignment_strength"]),
        float(row["centroid_strength"]),
        float(row.get("adversarial_strength", 0.0) or 0.0),
    )
    return suite.train_candidate(
        arrays=arrays,
        hidden_dims=suite.ARCHITECTURES[str(row["probe_architecture"])],
        method=str(row["method"]),
        mode=str(row["mode"]),
        candidate=candidate,
        seed=int(row["seed"]),
        include_validation=include_validation,
        fixed_epochs=int(row["best_epoch"]),
    )[0]


def latent(model, x: np.ndarray) -> np.ndarray:
    model.eval()
    with suite.torch.no_grad():
        return model.features(suite.existing_dann.to_tensor(x)).cpu().numpy()


def lr(c_value: float) -> LogisticRegression:
    return LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000, solver="lbfgs")


def tune_group(
    group: pd.DataFrame,
    raw: pd.DataFrame,
    columns: list[str],
    split: dict,
    tuning_records: list[dict],
    tuning_path: Path,
) -> float:
    return 1.0


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in runs.groupby(["architecture", "split", "variant"], sort=False):
        first = group.iloc[0]
        row = {
            "architecture": key[0], "split": key[1], "variant": key[2], "method": first["method"],
            "mode": first["mode"], "selected_C": first["selected_C"], "selection_status": first["selection_status"],
            "final_training": first["final_training"], "label_usage": first["label_usage"],
            "seed_values": ",".join(map(str, sorted(group.seed.astype(int).unique()))),
        }
        for metric in ["neural_test_accuracy", "lr_train_accuracy", "lr_test_accuracy", "lr_test_balanced_accuracy", "lr_test_macro_f1", "lr_minus_neural_accuracy"]:
            values = group[metric].astype(float)
            row[metric] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    options = parse_args()
    name = tag(options)
    paths = output_paths(name)
    selected = load_selected(options)
    raw = pd.read_csv(FEATURE_PATH)
    columns = suite.feature_columns(raw)
    split_map = {item["split"]: item for item in protocol.DA_SPLITS}
    tuning_frame, runs_frame, predictions_frame = read_or_empty(paths["tuning"]), read_or_empty(paths["runs"]), read_or_empty(paths["predictions"])
    tuning_records = tuning_frame.to_dict("records") if not tuning_frame.empty else []
    run_records = runs_frame.to_dict("records") if not runs_frame.empty else []
    prediction_records = predictions_frame.to_dict("records") if not predictions_frame.empty else []
    done = set(zip(runs_frame.architecture, runs_frame.split, runs_frame.variant, runs_frame.seed.astype(int))) if not runs_frame.empty else set()
    groups = list(selected.groupby(["probe_architecture", "split", "variant"], sort=False))
    for group_number, (_, group) in enumerate(groups, 1):
        first = group.iloc[0]
        print(f"[group {group_number}/{len(groups)}] {first['probe_architecture']} | {first['split']} | {first['variant']}", flush=True)
        split = split_map[str(first["split"])]
        selected_c = tune_group(group, raw, columns, split, tuning_records, paths["tuning"])
        parts = suite.partitions(raw, split)
        initial = suite.make_arrays(parts, columns)
        has_source_val = len(initial["x"]["source_val"]) > 0
        final_arrays, include_validation = arrays_for_final(parts, columns, str(first["mode"]), has_source_val)
        selection_status = "pre-specified fixed C=1; validation and test labels not used for LR selection"
        for row in group.itertuples(index=False):
            row_series = pd.Series(row._asdict())
            key = (row_series["probe_architecture"], row_series["split"], row_series["variant"], int(row_series["seed"]))
            if key in done:
                continue
            model = train_model(row_series, final_arrays, include_validation)
            x_train, y_train, label_usage = suite.classifier_data(final_arrays, str(row_series["mode"]), include_validation)
            x_test, y_test = final_arrays["x"]["target_test"], final_arrays["y"]["target_test"]
            h_train, h_test = latent(model, x_train), latent(model, x_test)
            classifier = lr(selected_c).fit(h_train, y_train)
            train_pred, test_pred = classifier.predict(h_train), classifier.predict(h_test)
            train_m, test_m = suite.metric(y_train, train_pred), suite.metric(y_test, test_pred)
            run_records.append({
                "architecture": row_series["probe_architecture"], "split": row_series["split"], "variant": row_series["variant"],
                "method": row_series["method"], "mode": row_series["mode"], "seed": int(row_series["seed"]),
                "selected_C": selected_c, "selection_status": selection_status, "best_epoch": int(row_series["best_epoch"]),
                "final_training": row_series["final_training"], "label_usage": label_usage,
                "neural_test_accuracy": float(row_series["test_accuracy"]), "lr_train_accuracy": train_m["accuracy"],
                "lr_test_accuracy": test_m["accuracy"], "lr_test_balanced_accuracy": test_m["balanced_accuracy"],
                "lr_test_macro_f1": test_m["macro_f1"], "lr_minus_neural_accuracy": test_m["accuracy"] - float(row_series["test_accuracy"]),
            })
            test_frame = parts["target_test"]
            for i, (index, sample) in enumerate(test_frame.iterrows()):
                prediction_records.append({
                    "architecture": row_series["probe_architecture"], "split": row_series["split"], "variant": row_series["variant"],
                    "seed": int(row_series["seed"]), "selected_C": selected_c, "sample_index": int(index),
                    "sample_id": sample.get("sample_id", index), "target_file": sample.get("source_file", sample.get("file", "")),
                    "period": sample.get("period", ""), "day_label": sample.get("day_label", ""), "batch": sample.get("batch", ""),
                    "true_label": suite.CLASSES[int(y_test[i])], "predicted_label": suite.CLASSES[int(test_pred[i])],
                    "correct": bool(y_test[i] == test_pred[i]),
                })
            save(pd.DataFrame(run_records), paths["runs"])
            save(pd.DataFrame(prediction_records), paths["predictions"])
            done.add(key)
    final_runs = pd.DataFrame(run_records)
    save(summarize(final_runs), paths["summary"])
    protocol_data = {
        "C": 1.0, "selection": "pre-specified fixed value; no LR hyperparameter selection",
        "architectures": options.architectures, "methods": options.methods, "seeds": list(protocol.MODEL_SEEDS),
        "UDA_boundary": "source validation only; target validation labels prohibited",
        "Semi_boundary": "target validation may select C and joins final Source+Target refit",
        "checkpointing": "final results saved incrementally",
        "test_policy": "evaluation-only",
    }
    paths["protocol"].write_text(json.dumps(protocol_data, indent=2), encoding="utf-8")
    print(f"Outputs written to {OUTPUT}")


if __name__ == "__main__":
    main()
