from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


THESIS_ROOT = Path(r"D:\thesis")
TRAINING_DIR = THESIS_ROOT / "code" / "training"
TABLE_DIR = THESIS_ROOT / "tables"
PYTHON = Path(
    r"C:\Users\yaoli\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)
MASTER_STATUS = TABLE_DIR / "framework_run_status.json"
LOCK_PATH = TABLE_DIR / "framework_run.lock"

FRAMEWORKS = {
    "ml": {
        "commands": [["run_manifest_da_baseline_splits.py"]],
        "environment": {"ENOSE_RUN_SCOPE": "all", "ENOSE_REBUILD_FEATURES_FROM_RAW": "1"},
        "patterns": ["manifest_da_baseline_*.csv"],
        "required": ["manifest_da_baseline_selected_summary.csv"],
    },
    "dann": {
        "commands": [
            ["run_dann_multi_splits.py"],
            ["build_dann_interactive_report.py"],
            ["build_dann_conditional_report.py"],
            ["run_cdan_multi_splits.py"],
            ["run_cdan_hybrid_report.py"],
        ],
        "environment": {},
        "patterns": ["dann_*.csv", "cdan_*.csv"],
        "required": [
            "dann_multi_split_selected_summary.csv",
            "dann_conditional_selected_summary.csv",
            "cdan_multi_split_selected_summary.csv",
            "cdan_hybrid_selected_summary.csv",
        ],
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_outputs(name: str, config: dict, started_timestamp: float) -> list[dict]:
    destination = TABLE_DIR / name
    destination.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Path] = {}
    for pattern in config["patterns"]:
        for source in TABLE_DIR.glob(pattern):
            if source.is_file() and source.stat().st_mtime >= started_timestamp:
                sources[source.name] = source

    missing = [filename for filename in config["required"] if filename not in sources]
    if missing:
        raise FileNotFoundError(
            f"{name} completed without required newly generated tables: {missing}"
        )

    manifest = []
    for filename in sorted(sources):
        source = sources[filename]
        target = destination / filename
        shutil.copy2(source, target)
        manifest.append(
            {
                "file": str(target),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )
    write_json(destination / "manifest.json", {"framework": name, "files": manifest})
    return manifest


def run_framework(name: str) -> dict:
    config = FRAMEWORKS[name]
    destination = TABLE_DIR / name
    destination.mkdir(parents=True, exist_ok=True)
    log_path = destination / "training.log"
    status_path = destination / "status.json"
    started_at = utc_now()
    started_timestamp = datetime.now(timezone.utc).timestamp()
    status = {
        "framework": name,
        "status": "running",
        "pid": os.getpid(),
        "started_at": started_at,
        "finished_at": None,
        "commands": [],
        "output_files": [],
    }
    write_json(status_path, status)

    environment = os.environ.copy()
    environment.update(config["environment"])
    environment["PYTHONUNBUFFERED"] = "1"
    environment["MPLBACKEND"] = "Agg"
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            for command_args in config["commands"]:
                script = TRAINING_DIR / command_args[0]
                if not script.is_file():
                    raise FileNotFoundError(f"Missing framework entrypoint: {script}")
                command = [str(PYTHON), str(script), *command_args[1:]]
                command_record = {
                    "command": command,
                    "started_at": utc_now(),
                    "finished_at": None,
                    "exit_code": None,
                }
                status["commands"].append(command_record)
                write_json(status_path, status)
                log.write(f"\n===== START {' '.join(command_args)} =====\n")
                result = subprocess.run(
                    command,
                    cwd=TRAINING_DIR,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                command_record["finished_at"] = utc_now()
                command_record["exit_code"] = result.returncode
                write_json(status_path, status)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"{name} command failed with exit code {result.returncode}: {command_args}"
                    )
                log.write(f"===== END {' '.join(command_args)} =====\n")

        status["output_files"] = collect_outputs(name, config, started_timestamp)
        status["status"] = "completed"
    except Exception as exc:
        status["status"] = "failed"
        status["error_type"] = type(exc).__name__
        status["error"] = str(exc)
    status["finished_at"] = utc_now()
    write_json(status_path, status)
    return status


def acquire_lock() -> int:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    return descriptor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run thesis model frameworks with file-only logging.")
    parser.add_argument("--framework", choices=["all", *FRAMEWORKS], default="all")
    args = parser.parse_args()
    if not PYTHON.is_file():
        raise FileNotFoundError(f"Bundled Python runtime not found: {PYTHON}")

    lock_descriptor = acquire_lock()
    selected = list(FRAMEWORKS) if args.framework == "all" else [args.framework]
    master = {
        "status": "running",
        "pid": os.getpid(),
        "started_at": utc_now(),
        "finished_at": None,
        "framework_order": selected,
        "frameworks": {},
    }
    write_json(MASTER_STATUS, master)
    try:
        for name in selected:
            master["frameworks"][name] = run_framework(name)
            write_json(MASTER_STATUS, master)
        failed = [name for name, result in master["frameworks"].items() if result["status"] != "completed"]
        master["status"] = "completed" if not failed else "completed_with_failures"
        master["failed_frameworks"] = failed
    finally:
        master["finished_at"] = utc_now()
        write_json(MASTER_STATUS, master)
        os.close(lock_descriptor)
        LOCK_PATH.unlink()


if __name__ == "__main__":
    main()
