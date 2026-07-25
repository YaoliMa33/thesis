from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


OLD_ROOT = Path(r"D:\datasets\enose_data_old")
NEW_ROOT = Path(r"D:\datasets\enose_data")
THESIS_ROOT = Path(r"D:\thesis")
TARGET_ROOT = THESIS_ROOT / "enose_data"
RECORD_PATH = TARGET_ROOT / "record_data.xlsx"
DESKTOP_RECORD = Path(r"C:\Users\yaoli\Desktop\record_data.xlsx")

LABELS = ["air", "alcohol", "acetone"]


def parse_record(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    rows = []
    period = ""
    day = ""
    day_num = -1
    date_text = ""
    for _, row in raw.iterrows():
        first = "" if pd.isna(row.iloc[0]) else str(row.iloc[0]).strip()
        if not first or first.lower() == "gas":
            continue
        if re.fullmatch(r"Period\s+[IVX]+", first, flags=re.I):
            period = first.replace("Period ", "Period_")
            day = ""
            day_num = -1
            date_text = ""
            continue
        day_match = re.match(r"Day\s*(\d+)", first, flags=re.I)
        if day_match:
            day_num = int(day_match.group(1))
            day = f"Day{day_num}"
            date_text = "" if pd.isna(row.iloc[2]) else str(row.iloc[2]).strip()
            continue
        sample_match = re.fullmatch(r"(air|alcohol|acetone|reference)_(\d+)", first, flags=re.I)
        if sample_match:
            gas = sample_match.group(1).lower()
            rows.append(
                {
                    "period": period,
                    "day": day,
                    "day_num": day_num,
                    "date_text": date_text,
                    "record_name": first,
                    "record_gas": gas,
                    "record_number": int(sample_match.group(2)),
                }
            )
    return pd.DataFrame(rows)


def rewrite_as_air(src: Path, dst: Path) -> None:
    df = pd.read_csv(src)
    if "label" in df.columns:
        df["label"] = "air"
    if "gas_type" in df.columns:
        df["gas_type"] = "air"
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)


def copy_csv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def record_lookup(record: pd.DataFrame, gas: str, number: int, period_scope: str | None = None) -> dict:
    sub = record[(record["record_gas"].eq(gas)) & (record["record_number"].eq(number))]
    if period_scope == "old":
        sub = sub[sub["period"].eq("Period_I")]
    elif period_scope == "new":
        sub = sub[sub["period"].isin(["Period_II", "Period_III"])]
    if sub.empty:
        return {"period": "", "day": "", "day_num": -1, "date_text": ""}
    return sub.iloc[0][["period", "day", "day_num", "date_text"]].to_dict()


def add_manifest_row(rows: list[dict], *, src: Path, dst: Path, label: str, source_dataset: str, source_kind: str, meta: dict) -> None:
    rows.append(
        {
            "label": label,
            "target_file": str(dst.relative_to(TARGET_ROOT)),
            "target_name": dst.name,
            "source_dataset": source_dataset,
            "source_kind": source_kind,
            "source_file": str(src),
            "source_name": src.name,
            "period": meta.get("period", ""),
            "day": meta.get("day", ""),
            "day_num": meta.get("day_num", -1),
            "date_text": meta.get("date_text", ""),
        }
    )


def backup_existing_target() -> None:
    if not TARGET_ROOT.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = THESIS_ROOT / f"enose_data_before_baseline_as_air_{stamp}"
    TARGET_ROOT.rename(backup)


def main() -> None:
    record_source = DESKTOP_RECORD if DESKTOP_RECORD.exists() else RECORD_PATH
    record = parse_record(record_source)
    backup_existing_target()
    for label in LABELS:
        (TARGET_ROOT / label).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []

    for gas in ["alcohol", "acetone"]:
        for src in sorted(OLD_ROOT.glob(f"{gas}_*.csv"), key=lambda p: int(re.search(r"_(\d+)", p.stem).group(1))):
            number = int(re.search(r"_(\d+)", src.stem).group(1))
            dst = TARGET_ROOT / gas / f"{gas}_o{number}.csv"
            copy_csv(src, dst)
            meta = record_lookup(record, gas, number, period_scope="old")
            if not meta["period"]:
                meta = {"period": "Period_I", "day": "", "day_num": -1, "date_text": ""}
            add_manifest_row(manifest_rows, src=src, dst=dst, label=gas, source_dataset="enose_data_old", source_kind=gas, meta=meta)

    for src in sorted(OLD_ROOT.glob("baseline_*.csv"), key=lambda p: int(re.search(r"_(\d+)", p.stem).group(1))):
        number = int(re.search(r"_(\d+)", src.stem).group(1))
        if number == 2:
            continue
        dst = TARGET_ROOT / "air" / f"air_ob{number}.csv"
        rewrite_as_air(src, dst)
        meta = {"period": "Period_I", "day": "", "day_num": -1, "date_text": ""}
        add_manifest_row(manifest_rows, src=src, dst=dst, label="air", source_dataset="enose_data_old", source_kind="baseline_as_air", meta=meta)

    for gas in ["air", "alcohol", "acetone"]:
        for src in sorted((NEW_ROOT / gas).glob(f"{gas}_*.csv"), key=lambda p: int(re.search(r"_(\d+)", p.stem).group(1))):
            number = int(re.search(r"_(\d+)", src.stem).group(1))
            dst = TARGET_ROOT / gas / src.name
            copy_csv(src, dst)
            meta = record_lookup(record, gas, number, period_scope="new")
            add_manifest_row(manifest_rows, src=src, dst=dst, label=gas, source_dataset="enose_data", source_kind=gas, meta=meta)

    for src in sorted((NEW_ROOT / "reference").glob("reference_*.csv"), key=lambda p: int(re.search(r"_(\d+)", p.stem).group(1))):
        number = int(re.search(r"_(\d+)", src.stem).group(1))
        dst = TARGET_ROOT / "air" / f"air_ref{number}.csv"
        rewrite_as_air(src, dst)
        meta = record_lookup(record, "reference", number, period_scope="new")
        add_manifest_row(manifest_rows, src=src, dst=dst, label="air", source_dataset="enose_data", source_kind="reference_as_air", meta=meta)

    if record_source.exists():
        copy_csv(record_source, TARGET_ROOT / "record_data.xlsx") if record_source.suffix.lower() == ".csv" else shutil.copy2(record_source, TARGET_ROOT / "record_data.xlsx")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(TARGET_ROOT / "manifest.csv", index=False)
    summary = (
        manifest.groupby(["label", "source_kind"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["label", "source_kind"])
    )
    summary.to_csv(TARGET_ROOT / "summary.csv", index=False)
    print("Wrote", TARGET_ROOT)
    print(summary.to_string(index=False))
    print("Total", len(manifest))


if __name__ == "__main__":
    main()
