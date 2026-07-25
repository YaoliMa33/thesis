"""Build the user-requested comparison without deleting exploratory runs."""

from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\thesis\tables\opensource_da_48_54")
RUNS = ROOT / "runs.csv"
SUMMARY = ROOT / "requested_scope_summary.csv"
RUNS_FILTERED = ROOT / "requested_scope_runs.csv"


def main() -> None:
    if not RUNS.exists():
        raise FileNotFoundError(RUNS)
    runs = pd.read_csv(RUNS)
    required = {"feature_set", "family", "method", "representation", "head", "split", "seed"}
    missing = sorted(required - set(runs.columns))
    if missing:
        raise ValueError(f"runs.csv is missing required columns: {missing}")
    requested_54 = (
        (runs["feature_set"] == "54D")
        & (runs["family"] == "domain-adaptation")
        & runs["method"].isin(["source-only", "dann"])
        & runs["representation"].isin(["bottleneck16", "residual-d", "fusion16"])
        & (runs["head"] == "linear")
    )
    requested_48 = runs["feature_set"] == "48D"
    selected = runs.loc[requested_54 | requested_48].copy()
    selected.to_csv(RUNS_FILTERED, index=False)
    group_columns = ["family", "feature_set", "split", "method", "method_family", "representation", "head"]
    rows = []
    for keys, group in selected.groupby(group_columns, dropna=False, sort=False):
        row = dict(zip(group_columns, keys))
        row["seed_count"] = len(group)
        row["seeds"] = ",".join(str(int(value)) for value in group["seed"])
        for metric in ("train_accuracy", "test_accuracy", "test_balanced_accuracy", "test_macro_f1"):
            values = pd.to_numeric(group[metric], errors="raise")
            row[metric] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    pd.DataFrame(rows).to_csv(SUMMARY, index=False)
    print(f"Requested-scope runs: {RUNS_FILTERED}")
    print(f"Requested-scope summary: {SUMMARY}")


if __name__ == "__main__":
    main()
