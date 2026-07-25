from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_dann_multi_splits as dann  # noqa: E402
import run_dann_one_layer_ablation as one_layer  # noqa: E402


OUT_ROOT = Path(r"D:\thesis")
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures" / "dann_learning"
FEATURE_PATH = TABLE_DIR / "manifest_baseline_as_air_features.csv"
SUMMARY_PATH = TABLE_DIR / "dann_multi_split_selected_summary.csv"
PARTITION_PATH = TABLE_DIR / "dann_multi_split_partitions.csv"
HTML_PATH = FIG_DIR / "dann_interactive_report.html"
CURVES_PATH = TABLE_DIR / "dann_interactive_training_curves.csv"
FEATURE_SPACE_PATH = TABLE_DIR / "dann_interactive_feature_space.csv"
ONE_LAYER_SUMMARY_PATH = TABLE_DIR / "dann_one_layer" / "dann_multi_split_selected_summary.csv"

TWO_LAYER = "Two-layer G_f: 54 -> 32 -> 16"
ONE_LAYER = "One-layer G_f: 54 -> 16"

LOG_EVERY = 5


def split_by_name(name: str) -> dict:
    for split in dann.SPLITS:
        if split["split"] == name:
            return split
    raise KeyError(name)


def task_classes(task: str) -> list[str]:
    for task_name, classes in dann.TASKS:
        if task_name == task:
            return classes
    raise KeyError(task)


def prepare_parts(df: pd.DataFrame, split: dict, classes: list[str]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    task = df[df["label"].isin(classes)].copy().reset_index(drop=True)
    task["_row_id"] = np.arange(len(task))
    masks = {
        "source_train": split["source_train"](task),
        "source_val": split["source_val"](task),
        "target_labeled_train": split["target_labeled_train"](task),
        "target_domain": split["target_domain"](task),
        "target_val": split["target_val"](task),
        "target_test": split["target_test"](task),
    }
    return task, {name: task[mask].copy() for name, mask in masks.items()}


def make_arrays(task: pd.DataFrame, parts: dict[str, pd.DataFrame], classes: list[str]) -> dict:
    cols = dann.feature_columns(task)
    raw = {name: part[cols].to_numpy(float) for name, part in parts.items()}
    x_ref = raw["source_train"]
    source_x, source_val_x, target_label_x, target_domain_x, target_val_x, target_test_x = dann.standardize(
        x_ref,
        raw["source_train"],
        raw["source_val"],
        raw["target_labeled_train"],
        raw["target_domain"],
        raw["target_val"],
        raw["target_test"],
    )
    y = {}
    for name, part in parts.items():
        y[name] = dann.encode(part["label"].to_numpy(str), classes) if len(part) else np.array([], dtype=np.int64)
    return {
        "cols": cols,
        "x": {
            "source_train": source_x,
            "source_val": source_val_x,
            "target_labeled_train": target_label_x,
            "target_domain": target_domain_x,
            "target_val": target_val_x,
            "target_test": target_test_x,
        },
        "raw": raw,
        "y": y,
    }


def classifier_data(arrays: dict, row: pd.Series) -> tuple[np.ndarray, np.ndarray, str]:
    x_source = arrays["x"]["source_train"]
    y_source = arrays["y"]["source_train"]
    x_target = arrays["x"]["target_labeled_train"]
    y_target = arrays["y"]["target_labeled_train"]
    if row["classifier_label_usage"] == "source labels + selected target labels":
        if len(x_target) == 0:
            raise ValueError("Row requests labeled target classifier data, but target_labeled_train is empty.")
        return np.vstack([x_source, x_target]), np.concatenate([y_source, y_target]), "source labels + selected target labels"
    return x_source, y_source, "source labels only"


def selection_data(arrays: dict, row: pd.Series) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    label = str(row["model_selection_labels"])
    if label == "target validation labels":
        if len(arrays["x"]["target_val"]) == 0:
            raise ValueError("Row requests target validation labels, but target_val is empty.")
        return arrays["x"]["target_val"], arrays["y"]["target_val"], "target validation labels"
    if label == "source validation labels":
        if len(arrays["x"]["source_val"]) == 0:
            raise ValueError("Row requests source validation labels, but source_val is empty.")
        return arrays["x"]["source_val"], arrays["y"]["source_val"], "source validation labels"
    if label == dann.NO_VALIDATION_SELECTION:
        return None, None, dann.NO_VALIDATION_SELECTION
    raise ValueError(f"Unknown model selection label: {label}")


def predict(model: dann.DANN, x: np.ndarray) -> np.ndarray:
    return dann.predict(model, dann.to_tensor(x))


def domain_metrics(model: dann.DANN, x_source: np.ndarray, x_target: np.ndarray) -> dict[str, float]:
    if len(x_target) == 0:
        return {"accuracy": float("nan"), "balanced_accuracy": float("nan"), "majority_baseline": float("nan")}
    x_dom = np.vstack([x_source, x_target])
    y_dom = np.concatenate([np.zeros(len(x_source), dtype=int), np.ones(len(x_target), dtype=int)])
    pred = dann.predict_domain(model, dann.to_tensor(x_dom))
    source_recall = float((pred[: len(x_source)] == 0).mean())
    target_recall = float((pred[len(x_source) :] == 1).mean())
    return {
        "accuracy": float((pred == y_dom).mean()),
        "balanced_accuracy": 0.5 * (source_recall + target_recall),
        "majority_baseline": max(len(x_source), len(x_target)) / len(y_dom),
    }


def train_with_curves(row: pd.Series, arrays: dict, classes: list[str]) -> tuple[dann.DANN, list[dict]]:
    seed = int(row["seed"])
    lambda_value = float(row["lambda"])
    use_domain_loss = str(row["variant"]) in {"DANN-UDA", "DANN-semi"}
    dann.set_random_seed(seed)

    x_cls, y_cls, _ = classifier_data(arrays, row)
    x_select, y_select, selection_label = selection_data(arrays, row)
    x_source_domain = arrays["x"]["source_train"]
    x_target_domain = arrays["x"]["target_domain"]
    x_test = arrays["x"]["target_test"]
    y_test = arrays["y"]["target_test"]

    model = dann.DANN(x_cls.shape[1], len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=dann.LEARNING_RATE, weight_decay=dann.WEIGHT_DECAY)
    x_cls_t = dann.to_tensor(x_cls)
    y_cls_t = dann.to_long(y_cls)
    y_weight = dann.class_weights(y_cls, len(classes))
    if use_domain_loss and len(x_target_domain) == 0:
        raise ValueError("DANN curve training requested domain loss, but target_domain is empty.")

    x_dom_t = torch.empty((0, x_cls.shape[1]), dtype=torch.float32)
    y_dom_t = torch.empty((0,), dtype=torch.long)
    if len(x_target_domain) > 0:
        x_dom_t = torch.cat([dann.to_tensor(x_source_domain), dann.to_tensor(x_target_domain)], dim=0)
        y_dom_t = torch.cat(
            [torch.zeros(len(x_source_domain), dtype=torch.long), torch.ones(len(x_target_domain), dtype=torch.long)],
            dim=0,
        )

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_score = (-1.0, -1.0, -1.0)
    curves = []

    for epoch in range(1, dann.EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        class_logits = model.forward_class(x_cls_t)
        L = F.cross_entropy(class_logits, y_cls_t, weight=y_weight)
        Ld = torch.tensor(0.0)
        backward_loss_scalar = L
        if use_domain_loss:
            domain_logits = model.forward_domain(x_dom_t, lambda_value)
            Ld = F.cross_entropy(domain_logits, y_dom_t)
            backward_loss_scalar = L + Ld
        backward_loss_scalar.backward()
        optimizer.step()

        if epoch % LOG_EVERY == 0 or epoch == 1 or epoch == dann.EPOCHS:
            train_m = dann.metric_dict(y_cls, predict(model, x_cls), len(classes))
            select_m = (
                {"accuracy": np.nan, "balanced_accuracy": np.nan, "macro_f1": np.nan}
                if x_select is None
                else dann.metric_dict(y_select, predict(model, x_select), len(classes))
            )
            test_m = dann.metric_dict(y_test, predict(model, x_test), len(classes))
            dom = domain_metrics(model, x_source_domain, x_target_domain)
            curves.append(
                {
                    "epoch": epoch,
                    "L": float(L.detach().cpu()),
                    "Ld": float(Ld.detach().cpu()),
                    "backward_loss_scalar": float(backward_loss_scalar.detach().cpu()),
                    "class_loss": float(L.detach().cpu()),
                    "domain_loss": float(Ld.detach().cpu()),
                    "total_loss": float(backward_loss_scalar.detach().cpu()),
                    "effective_feature_objective": float(L.detach().cpu()) - lambda_value * float(Ld.detach().cpu()),
                    "train_accuracy": train_m["accuracy"],
                    "validation_accuracy": select_m["accuracy"],
                    "test_accuracy": test_m["accuracy"],
                    "domain_discriminator_accuracy": dom["accuracy"],
                    "domain_discriminator_balanced_accuracy": dom["balanced_accuracy"],
                    "domain_majority_baseline": dom["majority_baseline"],
                }
            )
            score = (select_m["macro_f1"], select_m["balanced_accuracy"], select_m["accuracy"])
            if x_select is not None and score > best_score:
                best_score = score
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif x_select is None and epoch == dann.EPOCHS:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, curves


def pca_2d(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        raise ValueError("Cannot compute PCA coordinates for an empty feature matrix.")
    x = x.astype(float)
    if not np.isfinite(x).all():
        raise ValueError("Cannot compute PCA coordinates with NaN or infinite values.")
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] < 2:
        raise ValueError("PCA plot requires at least two samples.")
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    comp = vt[:2].T
    if comp.shape[1] < 2:
        comp = np.hstack([comp, np.zeros((comp.shape[0], 2 - comp.shape[1]))])
    return x @ comp[:, :2]


def latent_features(model: dann.DANN, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.feature(dann.to_tensor(x)).cpu().numpy()


def unique_plot_parts(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    priority = [
        ("source_train", "source"),
        ("source_val", "source"),
        ("target_labeled_train", "target"),
        ("target_domain", "target"),
        ("target_val", "target"),
        ("target_test", "target"),
    ]
    for partition, domain in priority:
        part = parts[partition].copy()
        if len(part) == 0:
            continue
        part["partition"] = partition
        part["domain"] = domain
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    joined = pd.concat(frames, ignore_index=True)
    return joined.drop_duplicates("_row_id", keep="first").reset_index(drop=True)


def x_for_plot(plot_df: pd.DataFrame, parts: dict[str, pd.DataFrame], arrays: dict, model: dann.DANN, use_latent: bool) -> np.ndarray:
    by_id = {}
    for name, part in parts.items():
        for local_idx, row_id in enumerate(part["_row_id"].to_numpy(int)):
            by_id[int(row_id)] = arrays["x"][name][local_idx]
    x_raw = np.vstack([by_id[int(row_id)] for row_id in plot_df["_row_id"].to_numpy(int)])
    return latent_features(model, x_raw) if use_latent else x_raw


def build_feature_rows(
    split_name: str,
    task_name: str,
    variant: str,
    model: dann.DANN,
    parts: dict[str, pd.DataFrame],
    arrays: dict,
    include_before_after: bool = False,
) -> list[dict]:
    plot_df = unique_plot_parts(parts)
    if len(plot_df) == 0:
        return []
    rows = []
    spaces = (
        [(False, "Before DA: standardized 54D PCA"), (True, "After feature extractor: latent 16D PCA")]
        if include_before_after
        else [
            (
                variant != "MLP source only",
                "After feature extractor: latent 16D PCA" if variant != "MLP source only" else "Before DA: standardized 54D PCA",
            )
        ]
    )
    for use_latent, space in spaces:
        x = x_for_plot(plot_df, parts, arrays, model, use_latent)
        coords = pca_2d(x)
        for i, meta in plot_df.reset_index(drop=True).iterrows():
            rows.append(
                {
                    "split": split_name,
                    "task": task_name,
                    "variant": variant,
                    "space": space,
                    "label": str(meta["label"]),
                    "domain": str(meta["domain"]),
                    "partition": str(meta["partition"]),
                    "period": str(meta["period"]),
                    "day": "" if pd.isna(meta.get("day", "")) else str(meta.get("day", "")),
                    "day_num": int(meta["day_num"]) if not pd.isna(meta["day_num"]) else -1,
                    "target_file": str(meta["target_file"]),
                    "x": float(coords[i, 0]),
                    "y": float(coords[i, 1]),
                }
            )
    return rows


def compact_records(df: pd.DataFrame) -> list[dict]:
    out = []
    for record in df.to_dict("records"):
        clean = {}
        for key, value in record.items():
            if isinstance(value, float) and np.isnan(value):
                clean[key] = None
            elif isinstance(value, np.integer):
                clean[key] = int(value)
            elif isinstance(value, np.floating):
                clean[key] = float(value)
            else:
                clean[key] = value
        out.append(clean)
    return out


def hide_pseudo_validation(summary: pd.DataFrame, curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Display NA for splits that explicitly declare no validation model selection."""
    summary = summary.copy()
    curves = curves.copy()
    validation_cols = ["validation_accuracy", "validation_balanced_accuracy", "validation_macro_f1"]
    no_val = summary["model_selection_labels"].eq(dann.NO_VALIDATION_SELECTION) if "model_selection_labels" in summary else pd.Series(False, index=summary.index)
    for col in validation_cols:
        if col in summary:
            summary.loc[no_val, col] = np.nan

    if not curves.empty and no_val.any():
        no_val_keys = set(
            summary.loc[no_val, ["split", "task", "variant"]]
            .astype(str)
            .agg("||".join, axis=1)
            .tolist()
        )
        curve_keys = curves[["split", "task", "variant"]].astype(str).agg("||".join, axis=1)
        curve_no_val = curve_keys.isin(no_val_keys)
        for col in validation_cols:
            if col in curves:
                curves.loc[curve_no_val, col] = np.nan
    return summary, curves


def write_html(summary: pd.DataFrame, partitions: pd.DataFrame, curves: pd.DataFrame, feature_space: pd.DataFrame) -> None:
    summary = summary.copy()
    curves = curves.copy()
    feature_space = feature_space.copy()
    if "architecture" not in summary:
        summary["architecture"] = TWO_LAYER
    if "architecture" not in curves:
        curves["architecture"] = TWO_LAYER
    if "architecture" not in feature_space:
        feature_space["architecture"] = TWO_LAYER
    summary, curves = hide_pseudo_validation(summary, curves)
    payload = {
        "summary": compact_records(summary),
        "partitions": compact_records(partitions),
        "curves": compact_records(curves),
        "featureSpace": compact_records(feature_space),
        "settings": {
            "model": "PyTorch MLP + DANN",
            "featureExtractor": "selectable: 54 -> 32 -> 16 or 54 -> 16",
            "classifierHead": "16 -> gas classes",
            "domainHead": "16 -> source/target",
            "epochs": dann.EPOCHS,
            "optimizer": "Adam",
            "learningRate": dann.LEARNING_RATE,
            "weightDecay": dann.WEIGHT_DECAY,
            "seeds": dann.SEEDS,
            "lambda": dann.LAMBDA_VALUES,
        },
    }
    data_json = json.dumps(payload, ensure_ascii=False)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DANN Drift Adaptation Dashboard</title>
<style>
:root {{
  --bg: #f6f7f9;
  --panel: #ffffff;
  --ink: #15202b;
  --muted: #607080;
  --line: #d9e0e8;
  --accent: #2166ac;
  --source: #2166ac;
  --target: #b2182b;
  --air: #4d9221;
  --air-light: #a6d96a;
  --alcohol: #2166ac;
  --alcohol-light: #92c5de;
  --acetone: #b2182b;
  --acetone-light: #f4a582;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: Arial, Helvetica, sans-serif;
  background: var(--bg);
  color: var(--ink);
}}
header {{
  padding: 22px 28px 14px;
  background: #ffffff;
  border-bottom: 1px solid var(--line);
}}
h1 {{ margin: 0 0 6px; font-size: 24px; }}
.sub {{ color: var(--muted); font-size: 13px; line-height: 1.45; }}
.layout {{
  display: grid;
  grid-template-columns: 280px minmax(0, 1fr);
  gap: 18px;
  padding: 18px;
}}
aside, section {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}}
aside {{ padding: 16px; align-self: start; position: sticky; top: 14px; }}
label {{ display: block; font-size: 12px; font-weight: 700; color: #334; margin: 14px 0 6px; }}
select, button {{
  width: 100%;
  padding: 9px 10px;
  border: 1px solid #b9c4d0;
  border-radius: 6px;
  background: #fff;
  color: var(--ink);
}}
.content {{ display: grid; gap: 18px; }}
section {{ padding: 16px; overflow: hidden; }}
h2 {{ margin: 0 0 12px; font-size: 18px; }}
h3 {{ margin: 8px 0 10px; font-size: 14px; }}
.metrics {{
  display: grid;
  grid-template-columns: repeat(6, minmax(120px, 1fr));
  gap: 10px;
}}
.metric {{
  border: 1px solid var(--line);
  border-radius: 7px;
  padding: 10px;
  background: #fbfcfe;
}}
.metric .k {{ color: var(--muted); font-size: 11px; }}
.metric .v {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: left; vertical-align: top; }}
th {{ background: #f2f5f8; font-weight: 700; color: #27313b; }}
.grid2 {{ display: grid; grid-template-columns: 1.1fr .9fr; gap: 14px; }}
.gas-grid {{ display: grid; grid-template-columns: repeat(3, minmax(260px, 1fr)); gap: 12px; }}
.pca-stack {{ display: grid; grid-template-columns: 1fr; gap: 12px; }}
.chart {{ border: 1px solid var(--line); border-radius: 7px; background: #fff; min-height: 260px; }}
.chart-title {{ font-size: 12px; font-weight: 700; fill: #26323d; }}
.axis {{ stroke: #9aa8b5; stroke-width: 1; }}
.gridline {{ stroke: #edf1f5; stroke-width: 1; }}
.legend {{ display: flex; gap: 14px; color: var(--muted); font-size: 12px; align-items: center; flex-wrap: wrap; }}
.dotkey {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 5px; }}
.note {{ color: var(--muted); font-size: 12px; line-height: 1.45; }}
.pill {{
  display: inline-block;
  padding: 4px 7px;
  border-radius: 999px;
  background: #eef3f8;
  color: #334;
  font-size: 11px;
  margin: 2px 4px 2px 0;
}}
@media (max-width: 1050px) {{
  .layout {{ grid-template-columns: 1fr; }}
  aside {{ position: static; }}
  .metrics {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
  .grid2, .gas-grid {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<header>
  <h1>DANN Drift Adaptation Dashboard</h1>
  <div class="sub">Interactive report for e-nose gas classification under temporal drift. Compare one-layer and two-layer feature extractors, before/after PCA, loss optimization, and seed-aggregated metrics.</div>
</header>
<div class="layout">
  <aside>
    <label for="architectureSelect">Feature extractor</label>
    <select id="architectureSelect"></select>
    <label for="splitSelect">Split</label>
    <select id="splitSelect"></select>
    <label for="taskSelect">Task</label>
    <select id="taskSelect"></select>
    <label for="variantSelect">Variant</label>
    <select id="variantSelect"></select>
    <div style="height:12px"></div>
    <div class="legend">
      <span><span class="dotkey" style="background:var(--source)"></span>Source</span>
      <span><span class="dotkey" style="background:var(--target)"></span>Target</span>
    </div>
    <p class="note">Before and after PCA are fitted separately for visualization. PCA overlap is diagnostic only and is not proof of domain invariance.</p>
  </aside>
  <main class="content">
    <section>
      <h2>Selected Model Summary</h2>
      <div id="metrics" class="metrics"></div>
      <div id="usage" style="margin-top:12px"></div>
    </section>
    <section>
      <h2>Variant Comparison in Current Split</h2>
      <div id="variantTable"></div>
    </section>
    <section>
      <h2>Partition and Label Usage</h2>
      <div id="partitionTable"></div>
    </section>
    <section>
      <h2>Feature Space: Before vs After Feature Extraction</h2>
      <p id="spaceNote" class="note"></p>
      <div class="grid2" style="margin-bottom:14px">
        <div><h3>Before: standardized 54D features</h3><div id="featureSpaceBefore" class="pca-stack"></div></div>
        <div><h3>After: learned latent 16D features</h3><div id="featureSpaceAfter" class="pca-stack"></div></div>
      </div>
    </section>
    <section>
      <h2>Training Curves</h2>
      <p class="note">Test curves are retrospective diagnostics only. They are never used for epoch, architecture, or hyperparameter selection.</p>
      <div class="grid2">
        <div id="lossChart" class="chart"></div>
        <div id="accChart" class="chart"></div>
      </div>
    </section>
    <section>
      <h2>Training Settings</h2>
      <div id="settings"></div>
    </section>
  </main>
</div>
<script>
const DATA = {data_json};

function fmt(v, d=3) {{
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "NA";
  return Number(v).toFixed(d);
}}
function fmtMeanStd(r, key, d=3) {{
  const mean = r[key];
  if (mean === null || mean === undefined || Number.isNaN(Number(mean))) return "NA";
  const std = r[`${{key}}_std`];
  const min = r[`${{key}}_min`];
  const max = r[`${{key}}_max`];
  let value = fmt(mean, d);
  if (std !== null && std !== undefined && !Number.isNaN(Number(std))) {{
    value += ` +/- ${{fmt(std, d)}}`;
  }}
  if (min !== null && min !== undefined && max !== null && max !== undefined && !Number.isNaN(Number(min)) && !Number.isNaN(Number(max))) {{
    value += ` [${{fmt(min, d)}}, ${{fmt(max, d)}}]`;
  }}
  return value;
}}
function uniq(arr) {{ return [...new Set(arr)]; }}
function labelTask(t) {{ return "Three-class: air / alcohol / acetone"; }}
function shortSplit(s) {{ return s.split("_")[0]; }}
function rowKey(r) {{ return `${{r.architecture}}||${{r.split}}||${{r.task}}||${{r.variant}}`; }}
function selected() {{
  const architecture = document.getElementById("architectureSelect").value;
  const split = document.getElementById("splitSelect").value;
  const task = document.getElementById("taskSelect").value;
  const variant = document.getElementById("variantSelect").value;
  return DATA.summary.find(r => r.architecture === architecture && r.split === split && r.task === task && r.variant === variant);
}}
function fillSelect(el, values, labels) {{
  const old = el.value;
  el.innerHTML = values.map(v => `<option value="${{htmlEscape(v)}}">${{htmlEscape(labels ? labels(v) : v)}}</option>`).join("");
  if (values.includes(old)) el.value = old;
}}
function htmlEscape(s) {{
  return String(s).replace(/[&<>"']/g, m => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[m]));
}}
function renderMetric(k, v) {{
  return `<div class="metric"><div class="k">${{htmlEscape(k)}}</div><div class="v">${{htmlEscape(v)}}</div></div>`;
}}
function renderTable(rows, cols) {{
  if (!rows.length) return "<p class='note'>No records for this selection.</p>";
  return `<table><thead><tr>${{cols.map(c => `<th>${{htmlEscape(c.label)}}</th>`).join("")}}</tr></thead><tbody>` +
    rows.map(r => `<tr>${{cols.map(c => `<td>${{htmlEscape(c.f(r))}}</td>`).join("")}}</tr>`).join("") +
    "</tbody></table>";
}}
function setupSelectors() {{
  const architectureSel = document.getElementById("architectureSelect");
  const splitSel = document.getElementById("splitSelect");
  const taskSel = document.getElementById("taskSelect");
  const variantSel = document.getElementById("variantSelect");
  fillSelect(architectureSel, uniq(DATA.summary.map(r => r.architecture)));
  function refreshSplitsAndTasks() {{
    const architecture = architectureSel.value;
    const available = DATA.summary.filter(r => r.architecture === architecture);
    fillSelect(splitSel, uniq(available.map(r => r.split)), s => `${{shortSplit(s)}} - ${{s.replace(shortSplit(s) + "_", "")}}`);
    fillSelect(taskSel, uniq(available.map(r => r.task)), labelTask);
  }}
  function refreshVariants() {{
    const architecture = architectureSel.value;
    const split = splitSel.value;
    const task = taskSel.value;
    const variants = DATA.summary.filter(r => r.architecture === architecture && r.split === split && r.task === task).map(r => r.variant);
    fillSelect(variantSel, variants);
  }}
  architectureSel.addEventListener("change", () => {{ refreshSplitsAndTasks(); refreshVariants(); renderAll(); }});
  splitSel.addEventListener("change", () => {{ refreshVariants(); renderAll(); }});
  taskSel.addEventListener("change", () => {{ refreshVariants(); renderAll(); }});
  variantSel.addEventListener("change", renderAll);
  refreshSplitsAndTasks();
  refreshVariants();
}}
function renderSummary(r) {{
  const seedText = r.seed_values ? `${{r.seed_values}} (representative ${{r.seed}} for curves)` : r.seed;
  document.getElementById("metrics").innerHTML = [
    renderMetric("Train accuracy", fmtMeanStd(r, "train_accuracy")),
    renderMetric("Validation accuracy", fmtMeanStd(r, "validation_accuracy")),
    renderMetric("Test accuracy", fmtMeanStd(r, "test_accuracy")),
    renderMetric("Test balanced accuracy", fmtMeanStd(r, "test_balanced_accuracy")),
    renderMetric("Test macro F1", fmtMeanStd(r, "test_macro_f1")),
    renderMetric("Raw domain acc (imbalanced)", fmtMeanStd(r, "domain_discriminator_accuracy")),
  ].join("");
  document.getElementById("usage").innerHTML = `
    <span class="pill">lambda=${{fmt(r.lambda, 2)}}</span>
    <span class="pill">architecture=${{htmlEscape(r.architecture)}}</span>
    <span class="pill">seeds=${{seedText}}</span>
    <span class="pill">best epoch=${{fmtMeanStd(r, "best_epoch", 1)}}</span>
    <span class="pill">classifier labels: ${{htmlEscape(r.classifier_label_usage)}}</span>
    <span class="pill">model selection: ${{htmlEscape(r.model_selection_labels)}}</span>
    <span class="pill">${{htmlEscape(r.judgment)}}</span>
    <p class="note">${{htmlEscape(r.description)}}</p>
  `;
}}
function renderVariantTable(r) {{
  const rows = DATA.summary.filter(x => x.architecture === r.architecture && x.split === r.split && x.task === r.task);
  document.getElementById("variantTable").innerHTML = renderTable(rows, [
    {{label:"Variant", f:x=>x.variant}},
    {{label:"Lambda", f:x=>fmt(x.lambda, 2)}},
    {{label:"Seeds", f:x=>x.seed_values || x.seed}},
    {{label:"Train acc", f:x=>fmtMeanStd(x, "train_accuracy")}},
    {{label:"Val acc", f:x=>fmtMeanStd(x, "validation_accuracy")}},
    {{label:"Test acc", f:x=>fmtMeanStd(x, "test_accuracy")}},
    {{label:"Test BA", f:x=>fmtMeanStd(x, "test_balanced_accuracy")}},
    {{label:"Macro F1", f:x=>fmtMeanStd(x, "test_macro_f1")}},
    {{label:"Raw domain acc", f:x=>fmtMeanStd(x, "domain_discriminator_accuracy")}},
    {{label:"Labels used in classifier", f:x=>x.classifier_label_usage}},
    {{label:"Judgment", f:x=>x.judgment}},
  ]);
}}
function renderPartitions(r) {{
  const rows = DATA.partitions.filter(x => x.split === r.split && x.task === r.task);
  document.getElementById("partitionTable").innerHTML = renderTable(rows, [
    {{label:"Partition", f:x=>x.partition}},
    {{label:"N", f:x=>x.n}},
    {{label:"Class counts", f:x=>x.class_counts || ""}},
    {{label:"Period / days", f:x=>x.period_days || ""}},
    {{label:"Label usage", f:x=>x.label_usage}},
  ]);
}}
function extent(values) {{
  let min = Math.min(...values), max = Math.max(...values);
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [-1,1];
  if (min === max) {{ min -= 1; max += 1; }}
  const pad = (max - min) * 0.08;
  return [min - pad, max + pad];
}}
function scale(v, min, max, a, b) {{ return a + (v - min) / (max - min) * (b - a); }}
function gasDomainColor(p) {{
  const colors = {{
    air: {{source:"var(--air)", target:"var(--air-light)"}},
    alcohol: {{source:"var(--alcohol)", target:"var(--alcohol-light)"}},
    acetone: {{source:"var(--acetone)", target:"var(--acetone-light)"}},
  }};
  const group = colors[p.label] || {{source:"#374151", target:"#d1d5db"}};
  return p.domain === "source" ? group.source : group.target;
}}
function scatterSvg(points, gas, width=360, height=270, colorMode="domain") {{
  const m = {{l:42, r:16, t:34, b:34}};
  const xs = points.map(p => Number(p.x)), ys = points.map(p => Number(p.y));
  const [xmin, xmax] = extent(xs), [ymin, ymax] = extent(ys);
  const ticks = [0,1,2,3,4];
  const circles = points.map(p => {{
    const cx = scale(Number(p.x), xmin, xmax, m.l, width-m.r);
    const cy = scale(Number(p.y), ymin, ymax, height-m.b, m.t);
    const fill = colorMode === "gasDomain" ? gasDomainColor(p) : (p.domain === "source" ? "var(--source)" : "var(--target)");
    const stroke = p.partition.includes("val") ? "#111" : "white";
    const tip = `${{p.label}} | ${{p.domain}} | ${{p.partition}} | ${{p.period}} Day${{p.day_num}} | ${{p.target_file}}`;
    return `<circle cx="${{cx}}" cy="${{cy}}" r="4.4" fill="${{fill}}" stroke="${{stroke}}" stroke-width="1"><title>${{htmlEscape(tip)}}</title></circle>`;
  }}).join("");
  const xgrid = ticks.map(i => {{
    const x = m.l + i * (width-m.l-m.r) / 4;
    return `<line class="gridline" x1="${{x}}" y1="${{m.t}}" x2="${{x}}" y2="${{height-m.b}}"/>`;
  }}).join("");
  const ygrid = ticks.map(i => {{
    const y = m.t + i * (height-m.t-m.b) / 4;
    return `<line class="gridline" x1="${{m.l}}" y1="${{y}}" x2="${{width-m.r}}" y2="${{y}}"/>`;
  }}).join("");
  return `<svg class="chart" viewBox="0 0 ${{width}} ${{height}}" width="100%" height="${{height}}">
    <text x="${{m.l}}" y="22" class="chart-title">${{htmlEscape(gas)}}</text>
    ${{xgrid}}${{ygrid}}
    <line class="axis" x1="${{m.l}}" y1="${{height-m.b}}" x2="${{width-m.r}}" y2="${{height-m.b}}"/>
    <line class="axis" x1="${{m.l}}" y1="${{m.t}}" x2="${{m.l}}" y2="${{height-m.b}}"/>
    ${{circles}}
  </svg>`;
}}
function combinedScatterSvg(points) {{
  const width = 820, height = 360;
  const legend = [
    ["air source","var(--air)"], ["air target","var(--air-light)"],
    ["alcohol source","var(--alcohol)"], ["alcohol target","var(--alcohol-light)"],
    ["acetone source","var(--acetone)"], ["acetone target","var(--acetone-light)"],
  ].map((item, i) => {{
    const x = 58 + (i % 3) * 210;
    const y = 330 + Math.floor(i / 3) * 18;
    return `<circle cx="${{x}}" cy="${{y - 4}}" r="5" fill="${{item[1]}}"/><text x="${{x + 12}}" y="${{y}}" font-size="11" fill="#4b5563">${{item[0]}}</text>`;
  }}).join("");
  const base = scatterSvg(points, "All gases combined", width, height, "gasDomain");
  return base.replace("</svg>", `${{legend}}</svg>`);
}}
function renderFeatureSpace(r) {{
  const points = DATA.featureSpace.filter(p => p.architecture === r.architecture && p.split === r.split && p.task === r.task && p.variant === r.variant);
  const before = points.filter(p => p.space.startsWith("Before DA"));
  const after = points.filter(p => p.space.startsWith("After feature extractor"));
  const gases = ["air", "alcohol", "acetone"].filter(g => before.some(p => p.label === g) || after.some(p => p.label === g));
  document.getElementById("spaceNote").textContent = "Each panel fits its own PCA. Dark colors are source and light colors are target. Compare domain overlap together with gas-class separation; overlap alone is not sufficient.";
  document.getElementById("featureSpaceBefore").innerHTML = gases.map(g => scatterSvg(before.filter(p => p.label === g), g)).join("");
  document.getElementById("featureSpaceAfter").innerHTML = gases.map(g => scatterSvg(after.filter(p => p.label === g), g)).join("");
}}
function lineChart(rows, series, title) {{
  const width = 520, height = 300, m = {{l:50,r:18,t:34,b:38}};
  if (!rows.length) return `<svg class="chart" viewBox="0 0 ${{width}} ${{height}}"><text x="20" y="30">No curves available.</text></svg>`;
  const xs = rows.map(r => Number(r.epoch));
  const ys = [];
  series.forEach(s => rows.forEach(r => {{ if (r[s.key] !== null && Number.isFinite(Number(r[s.key]))) ys.push(Number(r[s.key])); }}));
  const [xmin, xmax] = extent(xs), [ymin, ymax] = extent(ys);
  const lines = series.map(s => {{
    const pts = rows.filter(r => r[s.key] !== null && Number.isFinite(Number(r[s.key]))).map(r => {{
      const x = scale(Number(r.epoch), xmin, xmax, m.l, width-m.r);
      const y = scale(Number(r[s.key]), ymin, ymax, height-m.b, m.t);
      return `${{x}},${{y}}`;
    }}).join(" ");
    return `<polyline points="${{pts}}" fill="none" stroke="${{s.color}}" stroke-width="2"/>`;
  }}).join("");
  const legend = series.map((s,i) => `<text x="${{m.l + i*130}}" y="${{height-10}}" font-size="11" fill="${{s.color}}">${{s.label}}</text>`).join("");
  return `<svg class="chart" viewBox="0 0 ${{width}} ${{height}}" width="100%" height="${{height}}">
    <text x="${{m.l}}" y="22" class="chart-title">${{htmlEscape(title)}}</text>
    <line class="axis" x1="${{m.l}}" y1="${{height-m.b}}" x2="${{width-m.r}}" y2="${{height-m.b}}"/>
    <line class="axis" x1="${{m.l}}" y1="${{m.t}}" x2="${{m.l}}" y2="${{height-m.b}}"/>
    ${{lines}}${{legend}}
  </svg>`;
}}
function renderCurves(r) {{
  const rows = DATA.curves.filter(x => x.architecture === r.architecture && x.split === r.split && x.task === r.task && x.variant === r.variant);
  document.getElementById("lossChart").innerHTML = lineChart(rows, [
    {{key:"L", label:"L classification loss", color:"#2166ac"}},
    {{key:"Ld", label:"Ld DANN domain loss", color:"#b2182b"}},
    {{key:"Ld_cdan", label:"Ld_cdan CDAN domain loss", color:"#d6604d"}},
    {{key:"Lc", label:"Lc centroid loss", color:"#7b3294"}},
    {{key:"backward_loss_scalar", label:"L + Ld backprop scalar", color:"#4d9221"}},
    {{key:"effective_feature_objective", label:"L - lambda*Ld for G_f", color:"#e08214"}},
  ], "Loss curves");
  document.getElementById("accChart").innerHTML = lineChart(rows, [
    {{key:"train_accuracy", label:"train acc", color:"#2166ac"}},
    {{key:"validation_accuracy", label:"val acc", color:"#b2182b"}},
    {{key:"test_accuracy", label:"test acc", color:"#4d9221"}},
    {{key:"domain_discriminator_accuracy", label:"raw domain acc", color:"#6a3d9a"}},
    {{key:"domain_discriminator_balanced_accuracy", label:"balanced domain acc", color:"#e08214"}},
    {{key:"domain_majority_baseline", label:"domain majority baseline", color:"#737373"}},
  ], "Accuracy / domain curves");
}}
function renderSettings(r) {{
  const s = DATA.settings;
  document.getElementById("settings").innerHTML = `
    <span class="pill">Model: ${{s.model}}</span>
    <span class="pill">Feature extractor: ${{htmlEscape(r.architecture)}}</span>
    <span class="pill">Classifier head: ${{s.classifierHead}}</span>
    <span class="pill">Domain head: ${{s.domainHead}}</span>
    <span class="pill">Epochs: ${{s.epochs}}</span>
    <span class="pill">Optimizer: ${{s.optimizer}}</span>
    <span class="pill">Learning rate: ${{s.learningRate}}</span>
    <span class="pill">Weight decay: ${{s.weightDecay}}</span>
    <span class="pill">Seeds: ${{s.seeds.join(", ")}}</span>
    <span class="pill">Lambda grid: ${{s.lambda.join(", ")}}</span>
  `;
}}
function renderAll() {{
  const r = selected();
  if (!r) return;
  renderSummary(r);
  renderVariantTable(r);
  renderPartitions(r);
  renderFeatureSpace(r);
  renderCurves(r);
  renderSettings(r);
}}
setupSelectors();
renderAll();
</script>
</body>
</html>
"""
    HTML_PATH.write_text(html_text, encoding="utf-8")


def main() -> None:
    torch.set_num_threads(1)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURE_PATH)
    two_layer_summary = pd.read_csv(SUMMARY_PATH)
    two_layer_summary["architecture"] = TWO_LAYER
    summaries = [two_layer_summary]
    if ONE_LAYER_SUMMARY_PATH.exists():
        one_layer_summary = pd.read_csv(ONE_LAYER_SUMMARY_PATH)
        one_layer_summary["architecture"] = ONE_LAYER
        summaries.append(one_layer_summary)
    summary = pd.concat(summaries, ignore_index=True, sort=False)
    partitions = pd.read_csv(PARTITION_PATH)
    all_curves = []
    all_features = []

    for i, row in summary.iterrows():
        original_extractor = dann.FeatureExtractor
        dann.FeatureExtractor = one_layer.OneLayerFeatureExtractor if row["architecture"] == ONE_LAYER else original_extractor
        split = split_by_name(str(row["split"]))
        classes = task_classes(str(row["task"]))
        task, parts = prepare_parts(df, split, classes)
        arrays = make_arrays(task, parts, classes)
        try:
            model, curves = train_with_curves(row, arrays, classes)
        finally:
            dann.FeatureExtractor = original_extractor
        for curve in curves:
            curve.update(
                {
                    "split": row["split"],
                    "task": row["task"],
                    "variant": row["variant"],
                    "architecture": row["architecture"],
                    "lambda": float(row["lambda"]),
                    "seed": int(row["seed"]),
                    "seed_values": row["seed_values"] if "seed_values" in row and pd.notna(row["seed_values"]) else str(int(row["seed"])),
                }
            )
        all_curves.extend(curves)
        feature_rows = build_feature_rows(
            str(row["split"]),
            str(row["task"]),
            str(row["variant"]),
            model,
            parts,
            arrays,
            include_before_after=True,
        )
        for feature_row in feature_rows:
            feature_row["architecture"] = row["architecture"]
        all_features.extend(feature_rows)
        print(f"{i + 1:02d}/{len(summary)} {row['architecture']} | {row['split']} | {row['variant']}")

    curves_df = pd.DataFrame(all_curves)
    features_df = pd.DataFrame(all_features)
    curves_df.to_csv(CURVES_PATH, index=False)
    features_df.to_csv(FEATURE_SPACE_PATH, index=False)
    write_html(summary, partitions, curves_df, features_df)
    print(HTML_PATH)
    print(CURVES_PATH)
    print(FEATURE_SPACE_PATH)


if __name__ == "__main__":
    main()
