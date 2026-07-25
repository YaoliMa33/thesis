from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


DATA_ROOT = Path(r"D:\datasets\enose_export\enose_data")
RECORD_PATH = Path(r"C:\Users\yaoli\Desktop\record_enose.xlsx")
OUT_ROOT = Path(r"D:\thesis")
GASES = ["air", "alcohol", "acetone"]
COLORS = {"air": "#2d72b2", "alcohol": "#52a046", "acetone": "#c35359"}
DAY_COLORS = {"air": "#00a6d6", "alcohol": "#f2b705", "acetone": "#8b5cf6"}
FEATURE_SPACES = [
    ("s1_s2_s3", ["s1", "s2", "s3"], "S1/S2/S3"),
    ("s2_s3_s5", ["s2", "s3", "s5"], "S2/S3/S5"),
]


def natural_key(path: Path) -> tuple:
    chunks: list[object] = []
    current = ""
    is_digit = None
    for char in path.stem:
        char_is_digit = char.isdigit()
        if is_digit is None or char_is_digit == is_digit:
            current += char
        else:
            chunks.append(int(current) if is_digit else current)
            current = char
        is_digit = char_is_digit
    if current:
        chunks.append(int(current) if is_digit else current)
    return tuple(chunks)


def normalize_day_label(value: str) -> str | None:
    match = re.fullmatch(r"\s*day\s*(\d+)\s*", value, re.IGNORECASE)
    if not match:
        return None
    return f"Day{int(match.group(1))}"


def sample_parts(sample_name: str) -> tuple[str, int] | None:
    match = re.fullmatch(r"\s*(air|alcohol|acetone)_(\d+)\s*", sample_name, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower(), int(match.group(2))


def load_record_mapping() -> tuple[dict[str, dict[str, object]], list[str], pd.DataFrame]:
    raw = pd.read_excel(RECORD_PATH, header=None)
    current_day: str | None = None
    current_date = ""
    records = []
    mapping: dict[str, dict[str, object]] = {}
    for _, row in raw.iterrows():
        first = "" if pd.isna(row.iloc[0]) else str(row.iloc[0]).strip()
        day = normalize_day_label(first)
        if day:
            current_day = day
            current_date = "" if pd.isna(row.iloc[2]) else str(row.iloc[2]).strip()
            continue
        parts = sample_parts(first)
        if not parts or current_day is None:
            continue
        gas, group_num = parts
        sample_name = f"{gas}_{group_num}"
        meta = {
            "sample_name": sample_name,
            "source_file": f"{sample_name}.csv",
            "gas": gas,
            "group": group_num,
            "day": current_day,
            "date": current_date,
        }
        mapping[meta["source_file"]] = meta
        records.append(meta)

    def day_num(label: str) -> int:
        return int(re.search(r"\d+", label).group(0))

    days = sorted({str(record["day"]) for record in records}, key=day_num)
    return mapping, days, pd.DataFrame(records)


def elapsed_seconds(df: pd.DataFrame) -> pd.Series:
    if "arduino_time" in df.columns:
        t = pd.to_numeric(df["arduino_time"], errors="coerce")
        elapsed = (t - t.iloc[0]) / 1000.0
        if elapsed.notna().sum() > 10 and elapsed.max() > 30:
            return elapsed
    if "time_s" not in df.columns:
        raise ValueError("CSV is missing both arduino_time and time_s columns.")
    t = pd.to_numeric(df["time_s"], errors="coerce")
    return t - t.iloc[0]


def load_baseline_corrected_points(required_sensors: list[str], record_map: dict[str, dict[str, object]]) -> pd.DataFrame:
    frames = []
    for gas in GASES:
        gas_dir = DATA_ROOT / gas
        if not gas_dir.exists():
            raise FileNotFoundError(f"Missing gas directory: {gas_dir}")
        for path in sorted(gas_dir.glob("*.csv"), key=natural_key):
            raw = pd.read_csv(path)
            missing = [sensor for sensor in required_sensors if sensor not in raw.columns]
            if missing:
                raise ValueError(f"{path} is missing columns: {missing}")
            elapsed = elapsed_seconds(raw)
            sensors = raw[required_sensors].apply(pd.to_numeric, errors="coerce")
            baseline_mask = elapsed.between(0, 60, inclusive="left")
            if baseline_mask.sum() < 5:
                continue
            corrected = sensors - sensors.loc[baseline_mask].mean()
            meta = record_map.get(path.name, {})
            group_match = re.search(r"_(\d+)\.csv$", path.name)
            corrected["gas"] = gas
            corrected["group"] = int(meta.get("group", int(group_match.group(1)) if group_match else -1))
            corrected["day"] = str(meta.get("day", "Unknown"))
            corrected["elapsed_s"] = elapsed
            corrected["source_file"] = path.name
            frames.append(corrected.dropna(subset=required_sensors))
    if not frames:
        raise RuntimeError(f"No usable CSV files found under {DATA_ROOT}")
    return pd.concat(frames, ignore_index=True)


def normalize_for_display(data: pd.DataFrame, sensors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = data[sensors].to_numpy(float)
    lo = np.percentile(values, 1, axis=0)
    hi = np.percentile(values, 99, axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    out = data.copy()
    out[sensors] = np.clip((values - lo) / span, 0, 1)
    scale = pd.DataFrame({"sensor": sensors, "p01": lo, "p99": hi})
    return out, scale


def fit_lda_hyperplane(data: pd.DataFrame, sensors: list[str]) -> dict[str, object]:
    subset = data[data["gas"].isin(["alcohol", "acetone"])].copy()
    x0 = subset.loc[subset["gas"] == "alcohol", sensors].to_numpy(float)
    x1 = subset.loc[subset["gas"] == "acetone", sensors].to_numpy(float)
    mu0 = x0.mean(axis=0)
    mu1 = x1.mean(axis=0)
    centered0 = x0 - mu0
    centered1 = x1 - mu1
    scatter = centered0.T @ centered0 + centered1.T @ centered1
    pooled_cov = scatter / max(1, len(x0) + len(x1) - 2)
    reg = 1e-4 * np.trace(pooled_cov) / pooled_cov.shape[0]
    w = np.linalg.solve(pooled_cov + reg * np.eye(pooled_cov.shape[0]), mu1 - mu0)
    prior0 = len(x0) / (len(x0) + len(x1))
    prior1 = len(x1) / (len(x0) + len(x1))
    b = -0.5 * float(w @ (mu0 + mu1)) + math.log(prior1 / prior0)
    scores = subset[sensors].to_numpy(float) @ w + b
    pred = np.where(scores >= 0, "acetone", "alcohol")
    actual = subset["gas"].to_numpy()
    alcohol_total = int((actual == "alcohol").sum())
    acetone_total = int((actual == "acetone").sum())
    alcohol_correct = int(((actual == "alcohol") & (pred == "alcohol")).sum())
    acetone_correct = int(((actual == "acetone") & (pred == "acetone")).sum())
    alcohol_as_acetone = alcohol_total - alcohol_correct
    acetone_as_alcohol = acetone_total - acetone_correct
    acc = float((pred == actual).mean())
    alcohol_recall = alcohol_correct / alcohol_total
    acetone_recall = acetone_correct / acetone_total
    bacc = 0.5 * (alcohol_recall + acetone_recall)
    return {
        "w": w.tolist(),
        "b": float(b),
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "alcohol_total": alcohol_total,
        "acetone_total": acetone_total,
        "alcohol_correct": alcohol_correct,
        "acetone_correct": acetone_correct,
        "alcohol_as_acetone": alcohol_as_acetone,
        "acetone_as_alcohol": acetone_as_alcohol,
        "alcohol_recall": alcohol_recall,
        "acetone_recall": acetone_recall,
    }


def plane_cells(w: list[float], b: float, n: int = 26) -> list[list[list[float]]]:
    normal = np.asarray(w, dtype=float)
    solve_axis = int(np.argmax(np.abs(normal)))
    axes = [0, 1, 2]
    axes.remove(solve_axis)
    vals = np.linspace(0, 1, n + 1)
    cells = []
    for i in range(n):
        for j in range(n):
            corners = [(vals[i], vals[j]), (vals[i + 1], vals[j]), (vals[i + 1], vals[j + 1]), (vals[i], vals[j + 1])]
            quad = []
            ok = True
            for a, c in corners:
                p = [0.0, 0.0, 0.0]
                p[axes[0]] = float(a)
                p[axes[1]] = float(c)
                p[solve_axis] = float(-(b + normal[axes[0]] * a + normal[axes[1]] * c) / normal[solve_axis])
                if p[solve_axis] < 0 or p[solve_axis] > 1:
                    ok = False
                    break
                quad.append([round(v, 5) for v in p])
            if ok:
                cells.append(quad)
    return cells


def make_points_payload(data: pd.DataFrame, sensors: list[str], days: list[str], hyperplane: dict[str, object]) -> list[list[float | int]]:
    day_index = {day: idx for idx, day in enumerate(days)}
    w = np.asarray(hyperplane["w"], dtype=float)
    b = float(hyperplane["b"])
    points = []
    for row in data[sensors + ["gas", "day", "group", "elapsed_s"]].itertuples(index=False, name=None):
        gas = str(row[3])
        pred_idx = -1
        error = 0
        score = 0.0
        if gas in {"alcohol", "acetone"}:
            x = np.asarray(row[:3], dtype=float)
            score = float(x @ w + b)
            pred = "acetone" if score >= 0 else "alcohol"
            pred_idx = GASES.index(pred)
            error = 1 if pred != gas else 0
        points.append([
            round(float(row[0]), 5),
            round(float(row[1]), 5),
            round(float(row[2]), 5),
            GASES.index(gas),
            day_index.get(str(row[4]), -1),
            int(row[5]),
            round(float(row[6]), 4),
            pred_idx,
            error,
            round(score, 5),
        ])
    return points


def make_html(title_suffix: str, sensors: list[str], points: list[list[float | int]], counts: dict[str, int], scale: pd.DataFrame, days: list[str], hyperplane: dict[str, object]) -> str:
    payload = {
        "points": points,
        "gases": GASES,
        "colors": COLORS,
        "dayColors": DAY_COLORS,
        "counts": counts,
        "scale": scale.to_dict(orient="records"),
        "labels": [sensor.upper() for sensor in sensors],
        "titleSuffix": title_suffix,
        "days": days,
        "plane": {
            "w": hyperplane["w"],
            "b": hyperplane["b"],
            "cells": plane_cells(hyperplane["w"], float(hyperplane["b"])),
            "accuracy": hyperplane["accuracy"],
            "balancedAccuracy": hyperplane["balanced_accuracy"],
            "alcoholRecall": hyperplane["alcohol_recall"],
            "acetoneRecall": hyperplane["acetone_recall"],
            "alcoholCorrect": hyperplane["alcohol_correct"],
            "alcoholTotal": hyperplane["alcohol_total"],
            "acetoneCorrect": hyperplane["acetone_correct"],
            "acetoneTotal": hyperplane["acetone_total"],
            "alcoholAsAcetone": hyperplane["alcohol_as_acetone"],
            "acetoneAsAlcohol": hyperplane["acetone_as_alcohol"],
        },
    }
    payload_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Interactive baseline-corrected {title_suffix} feature space with alcohol/acetone hyperplane</title>
<style>
  html, body {{ margin: 0; height: 100%; overflow: hidden; background: #ffffff; color: #22272e; font-family: Arial, sans-serif; }}
  #wrap {{ position: fixed; inset: 0; display: grid; grid-template-rows: auto 1fr; }}
  header {{ padding: 16px 24px 9px; border-bottom: 1px solid #e6e8eb; }}
  h1 {{ margin: 0; font-size: 22px; font-weight: 700; letter-spacing: 0; }}
  p {{ margin: 5px 0 0; color: #656d76; font-size: 14px; }}
  #stage {{ position: relative; min-height: 0; }}
  canvas {{ position: absolute; inset: 0; width: 100%; height: 100%; cursor: grab; }}
  canvas:active {{ cursor: grabbing; }}
  #legend, #controls {{ position: absolute; background: rgba(255,255,255,.91); border: 1px solid #e6e8eb; border-radius: 8px; box-shadow: 0 8px 24px rgba(27,31,36,.08); font-size: 14px; }}
  #legend {{ top: 18px; left: 22px; padding: 12px 14px; }}
  #controls {{ top: 18px; right: 22px; width: 335px; max-height: calc(100% - 88px); overflow: auto; padding: 14px; }}
  .row {{ display: flex; align-items: center; gap: 8px; margin: 6px 0; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  .section {{ border-top: 1px solid #e6e8eb; padding-top: 12px; margin-top: 12px; }}
  .section:first-child {{ border-top: 0; padding-top: 0; margin-top: 0; }}
  label {{ display: block; color: #3f4650; font-size: 12px; font-weight: 700; margin: 9px 0 4px; }}
  select, input {{ width: 100%; box-sizing: border-box; border: 1px solid #d0d7de; border-radius: 6px; padding: 7px 8px; background: #fff; color: #22272e; font-size: 13px; }}
  select[multiple] {{ min-height: 112px; }}
  button {{ border: 1px solid #d0d7de; border-radius: 6px; padding: 7px 10px; background: #f6f8fa; color: #22272e; cursor: pointer; font-size: 13px; }}
  button:hover {{ background: #eef2f6; }}
  button:disabled {{ color: #8c959f; cursor: not-allowed; background: #f6f8fa; }}
  .buttons {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }}
  .active {{ background: #0969da; border-color: #0969da; color: #fff; }}
  .hintline {{ color: #656d76; font-size: 12px; line-height: 1.35; margin-top: 6px; }}
  #animStatus, #planeStats {{ color: #3f4650; font-size: 12px; line-height: 1.45; margin-top: 8px; }}
  #status {{ margin-top: 10px; color: #3f4650; font-size: 12px; line-height: 1.35; }}
  #hint {{ position: absolute; right: 22px; bottom: 18px; color: #656d76; font-size: 13px; background: rgba(255,255,255,.82); padding: 8px 10px; border-radius: 6px; }}
  .toggle {{ display: flex; align-items: center; gap: 8px; margin-top: 8px; }}
  .toggle input {{ width: auto; }}
  code {{ font-family: Consolas, monospace; font-size: 12px; }}
</style>
</head>
<body>
<div id="wrap">
  <header>
    <h1>Interactive baseline-corrected {title_suffix} feature space</h1>
    <p>All current alcohol / acetone / air CSV files are included. Points are baseline-corrected original samples. The translucent plane is fitted only on alcohol vs acetone in this 3D space.</p>
  </header>
  <div id="stage">
    <canvas id="canvas"></canvas>
    <div id="legend"></div>
    <div id="controls">
      <div class="section">
        <strong>Alcohol/Acetone Hyperplane</strong>
        <div class="toggle"><input id="showPlane" type="checkbox" checked><span>show linear hyperplane</span></div>
        <div class="toggle"><input id="showPred" type="checkbox"><span>color alcohol/acetone by predicted side</span></div>
        <div class="toggle"><input id="showErrors" type="checkbox" checked><span>emphasize misclassified alcohol/acetone</span></div>
        <div id="planeStats"></div>
      </div>
      <div class="section">
        <strong>Highlight by Day</strong>
        <label for="daySelect">Day</label>
        <select id="daySelect"></select>
        <div class="hintline">Selected day's air/alcohol/acetone are shown with three alternate colors.</div>
        <div class="buttons"><button id="modeDay">Apply Day</button></div>
      </div>
      <div class="section">
        <strong>Highlight Gas Groups</strong>
        <label for="groupGas">Gas</label>
        <select id="groupGas"></select>
        <label for="groupsInput">Group number(s)</label>
        <input id="groupsInput" placeholder="e.g. 1, 4, 10-12">
        <div class="hintline">Selected trial groups are highlighted in amber.</div>
        <div class="buttons"><button id="modeGroups">Apply Groups</button></div>
        <div class="section">
          <strong>Single-Trial Trajectory</strong>
          <div class="hintline">Available when exactly one group is selected. The path follows trial order from start to end.</div>
          <div class="buttons">
            <button id="playAnim" disabled>Play</button>
            <button id="resetAnim" disabled>Reset</button>
          </div>
          <label for="speedRange">Animation speed</label>
          <input id="speedRange" type="range" min="1" max="24" value="8">
          <div id="animStatus">Select one group to enable trajectory animation.</div>
        </div>
      </div>
      <div class="section">
        <strong>Highlight Gas Days</strong>
        <label for="dayGas">Gas</label>
        <select id="dayGas"></select>
        <label for="multiDaySelect">Day(s)</label>
        <select id="multiDaySelect" multiple></select>
        <div class="hintline">Selected gas across selected days is highlighted in violet.</div>
        <div class="buttons">
          <button id="modeGasDays">Apply Gas Days</button>
          <button id="clearBtn">Clear</button>
        </div>
      </div>
      <div id="status">No highlight selected.</div>
    </div>
    <div id="hint">Drag to rotate · wheel to zoom · double click to reset</div>
  </div>
</div>
<script>
const DATA = {payload_json};
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const legend = document.getElementById('legend');
const gasNames = DATA.gases;
const gasIndex = Object.fromEntries(DATA.gases.map((g, i) => [g, i]));
const dayIndex = Object.fromEntries(DATA.days.map((d, i) => [d, i]));
const groupHighlight = '#ff8c00';
const gasDayHighlight = '#111827';
const currentPointColor = '#00d1b2';
const predAlcohol = '#1f9d55';
const predAcetone = '#d94841';
const errorColor = '#111827';
let yaw = -0.65, pitch = 0.38, zoom = 1.1;
let dragging = false, lastX = 0, lastY = 0, pending = false;
let highlight = {{ mode: 'none', day: -1, gas: -1, groups: new Set(), days: new Set() }};
let animation = {{ playing: false, index: 0, lastTs: 0, trajectory: [] }};

function pct(v) {{ return (v * 100).toFixed(2) + '%'; }}
document.getElementById('planeStats').innerHTML = `
  <div>Accuracy: <strong>${{pct(DATA.plane.accuracy)}}</strong></div>
  <div>Balanced accuracy: <strong>${{pct(DATA.plane.balancedAccuracy)}}</strong></div>
  <div>Alcohol recall: ${{DATA.plane.alcoholCorrect}} / ${{DATA.plane.alcoholTotal}} (${{pct(DATA.plane.alcoholRecall)}})</div>
  <div>Acetone recall: ${{DATA.plane.acetoneCorrect}} / ${{DATA.plane.acetoneTotal}} (${{pct(DATA.plane.acetoneRecall)}})</div>
  <div>Alcohol → acetone: ${{DATA.plane.alcoholAsAcetone}}; acetone → alcohol: ${{DATA.plane.acetoneAsAlcohol}}</div>
  <div class="hintline">This is an in-space descriptive LDA hyperplane, not a final train/test ML result.</div>
`;

function optionList(select, values) {{
  select.innerHTML = values.map(v => `<option value="${{v}}">${{v}}</option>`).join('');
}}
optionList(document.getElementById('daySelect'), DATA.days);
optionList(document.getElementById('multiDaySelect'), DATA.days);
optionList(document.getElementById('groupGas'), DATA.gases);
optionList(document.getElementById('dayGas'), DATA.gases);

legend.innerHTML = DATA.gases.map(g => `
  <div class="row"><span class="dot" style="background:${{DATA.colors[g]}}"></span><span>${{g}} (n=${{DATA.counts[g]}})</span></div>
`).join('') + `
  <div class="section">
    <div class="row"><span class="dot" style="background:${{DATA.dayColors.air}}"></span><span>selected day air</span></div>
    <div class="row"><span class="dot" style="background:${{DATA.dayColors.alcohol}}"></span><span>selected day alcohol</span></div>
    <div class="row"><span class="dot" style="background:${{DATA.dayColors.acetone}}"></span><span>selected day acetone</span></div>
    <div class="row"><span class="dot" style="background:${{groupHighlight}}"></span><span>selected group(s)</span></div>
    <div class="row"><span class="dot" style="background:${{gasDayHighlight}}"></span><span>selected gas day(s)</span></div>
    <div class="row"><span class="dot" style="background:rgba(9,105,218,.5)"></span><span>hyperplane</span></div>
    <div class="row"><span class="dot" style="background:${{errorColor}}"></span><span>wrong side</span></div>
  </div>`;

function parseGroups(text) {{
  const groups = new Set();
  for (const part of text.split(',')) {{
    const s = part.trim();
    if (!s) continue;
    const m = s.match(/^(\\d+)\\s*-\\s*(\\d+)$/);
    if (m) {{
      const a = Number(m[1]), b = Number(m[2]);
      for (let i = Math.min(a,b); i <= Math.max(a,b); i++) groups.add(i);
    }} else if (/^\\d+$/.test(s)) {{
      groups.add(Number(s));
    }}
  }}
  return groups;
}}

function setStatus(text) {{ document.getElementById('status').textContent = text; }}
function setAnimStatus(text) {{ document.getElementById('animStatus').textContent = text; }}
function activeButton(id) {{
  for (const btn of ['modeDay','modeGroups','modeGasDays']) document.getElementById(btn).classList.toggle('active', btn === id);
}}

function stopAnimation() {{
  animation.playing = false;
  animation.lastTs = 0;
  document.getElementById('playAnim').textContent = 'Play';
}}

function selectedGroupNumber() {{
  if (highlight.mode !== 'groups' || highlight.groups.size !== 1) return null;
  return [...highlight.groups][0];
}}

function refreshAnimation(resetIndex = true) {{
  const group = selectedGroupNumber();
  const playBtn = document.getElementById('playAnim');
  const resetBtn = document.getElementById('resetAnim');
  if (group === null || highlight.gas < 0) {{
    stopAnimation();
    animation.trajectory = [];
    animation.index = 0;
    playBtn.disabled = true;
    resetBtn.disabled = true;
    setAnimStatus('Select one group to enable trajectory animation.');
    return;
  }}
  animation.trajectory = DATA.points
    .filter(p => p[3] === highlight.gas && p[5] === group)
    .sort((a, b) => a[6] - b[6]);
  if (resetIndex) animation.index = 0;
  playBtn.disabled = animation.trajectory.length === 0;
  resetBtn.disabled = animation.trajectory.length === 0;
  const gas = gasNames[highlight.gas];
  const duration = animation.trajectory.length ? animation.trajectory[animation.trajectory.length - 1][6].toFixed(1) : '0.0';
  setAnimStatus(`${{gas}}_${{group}} trajectory ready: ${{animation.trajectory.length}} points, 0.0-${{duration}} s.`);
}}

for (const id of ['showPlane','showPred','showErrors']) {{
  document.getElementById(id).addEventListener('change', requestDraw);
}}

document.getElementById('modeDay').addEventListener('click', () => {{
  stopAnimation();
  const d = document.getElementById('daySelect').value;
  highlight = {{ mode: 'day', day: dayIndex[d], gas: -1, groups: new Set(), days: new Set() }};
  refreshAnimation();
  setStatus(`Highlighting ${{d}} for all gases.`);
  activeButton('modeDay');
  requestDraw();
}});
document.getElementById('modeGroups').addEventListener('click', () => {{
  stopAnimation();
  const gas = document.getElementById('groupGas').value;
  const groups = parseGroups(document.getElementById('groupsInput').value);
  highlight = {{ mode: 'groups', day: -1, gas: gasIndex[gas], groups, days: new Set() }};
  refreshAnimation();
  setStatus(`Highlighting ${{gas}} group(s): ${{[...groups].sort((a,b)=>a-b).join(', ') || 'none'}}.`);
  activeButton('modeGroups');
  requestDraw();
}});
document.getElementById('modeGasDays').addEventListener('click', () => {{
  stopAnimation();
  const gas = document.getElementById('dayGas').value;
  const days = new Set([...document.getElementById('multiDaySelect').selectedOptions].map(o => dayIndex[o.value]));
  highlight = {{ mode: 'gasDays', day: -1, gas: gasIndex[gas], groups: new Set(), days }};
  refreshAnimation();
  const labels = [...days].sort((a,b)=>a-b).map(i => DATA.days[i]).join(', ');
  setStatus(`Highlighting ${{gas}} on: ${{labels || 'none'}}.`);
  activeButton('modeGasDays');
  requestDraw();
}});
document.getElementById('clearBtn').addEventListener('click', () => {{
  stopAnimation();
  highlight = {{ mode: 'none', day: -1, gas: -1, groups: new Set(), days: new Set() }};
  refreshAnimation();
  setStatus('No highlight selected.');
  activeButton('');
  requestDraw();
}});

document.getElementById('playAnim').addEventListener('click', () => {{
  if (!animation.trajectory.length) return;
  animation.playing = !animation.playing;
  document.getElementById('playAnim').textContent = animation.playing ? 'Pause' : 'Play';
  if (animation.playing) {{
    if (animation.index >= animation.trajectory.length - 1) animation.index = 0;
    animation.lastTs = 0;
    requestAnimationFrame(animationTick);
  }}
  requestDraw();
}});

document.getElementById('resetAnim').addEventListener('click', () => {{
  stopAnimation();
  animation.index = 0;
  refreshAnimation(false);
  requestDraw();
}});

function animationTick(ts) {{
  if (!animation.playing || !animation.trajectory.length) return;
  if (!animation.lastTs) animation.lastTs = ts;
  const delta = Math.max(0, ts - animation.lastTs);
  animation.lastTs = ts;
  const speed = Number(document.getElementById('speedRange').value);
  animation.index = Math.min(animation.trajectory.length - 1, animation.index + Math.max(1, Math.round(delta / 16 * speed)));
  requestDraw();
  if (animation.index >= animation.trajectory.length - 1) {{
    stopAnimation();
  }} else {{
    requestAnimationFrame(animationTick);
  }}
}}

function requestDraw() {{
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => {{ pending = false; draw(); }});
}}

function resize() {{
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  requestDraw();
}}

function rotatePoint(x, y, z) {{
  x -= 0.5; y -= 0.5; z -= 0.5;
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  let x1 = x * cy - y * sy;
  let y1 = x * sy + y * cy;
  let z1 = z;
  let y2 = y1 * cp - z1 * sp;
  let z2 = y1 * sp + z1 * cp;
  return [x1, y2, z2];
}}

function project(p, rect) {{
  const scale = Math.min(rect.width, rect.height) * 0.78 * zoom;
  return [rect.width * 0.48 + p[0] * scale, rect.height * 0.56 - p[1] * scale, p[2]];
}}

function drawAxes(rect) {{
  const corners = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]];
  const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  const pts = corners.map(c => project(rotatePoint(...c), rect));
  ctx.lineWidth = 1; ctx.strokeStyle = '#d9dee3';
  for (const [a,b] of edges) {{ ctx.beginPath(); ctx.moveTo(pts[a][0], pts[a][1]); ctx.lineTo(pts[b][0], pts[b][1]); ctx.stroke(); }}
  const axes = [[[0,0,0],[1.12,0,0],DATA.labels[0]],[[0,0,0],[0,1.12,0],DATA.labels[1]],[[0,0,0],[0,0,1.12],DATA.labels[2]]];
  ctx.strokeStyle = '#4f565e'; ctx.fillStyle = '#4f565e'; ctx.font = '15px Arial';
  for (const [a,b,label] of axes) {{
    const pa = project(rotatePoint(...a), rect), pb = project(rotatePoint(...b), rect);
    ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
    ctx.fillText(label, pb[0] + 5, pb[1] - 5);
  }}
}}

function drawPlane(rect) {{
  if (!document.getElementById('showPlane').checked) return;
  const cells = DATA.plane.cells.map(cell => {{
    const projected = cell.map(p => project(rotatePoint(p[0], p[1], p[2]), rect));
    const depth = projected.reduce((s, p) => s + p[2], 0) / projected.length;
    return [projected, depth];
  }}).sort((a, b) => a[1] - b[1]);
  ctx.globalAlpha = 0.2;
  ctx.fillStyle = '#0969da';
  ctx.strokeStyle = '#0969da';
  ctx.lineWidth = 0.7;
  for (const [cell] of cells) {{
    ctx.beginPath();
    ctx.moveTo(cell[0][0], cell[0][1]);
    for (let i = 1; i < cell.length; i++) ctx.lineTo(cell[i][0], cell[i][1]);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }}
  ctx.globalAlpha = 1;
}}

function isHighlighted(p) {{
  if (highlight.mode === 'day') return p[4] === highlight.day;
  if (highlight.mode === 'groups') return p[3] === highlight.gas && highlight.groups.has(p[5]);
  if (highlight.mode === 'gasDays') return p[3] === highlight.gas && highlight.days.has(p[4]);
  return false;
}}

function pointColor(p) {{
  if (isHighlighted(p)) {{
    if (highlight.mode === 'day') return DATA.dayColors[gasNames[p[3]]];
    if (highlight.mode === 'groups') return groupHighlight;
    if (highlight.mode === 'gasDays') return gasDayHighlight;
  }}
  if (document.getElementById('showPred').checked && p[7] >= 0) {{
    return p[7] === gasIndex.acetone ? predAcetone : predAlcohol;
  }}
  return DATA.colors[gasNames[p[3]]];
}}

function draw() {{
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  drawAxes(rect);
  drawPlane(rect);
  const projected = DATA.points.map(p => [p, project(rotatePoint(p[0], p[1], p[2]), rect), isHighlighted(p)]);
  ctx.globalAlpha = 0.3;
  for (const [p, pp, selected] of projected) {{
    if (selected) continue;
    if (document.getElementById('showErrors').checked && p[8] === 1) continue;
    ctx.fillStyle = pointColor(p);
    ctx.beginPath(); ctx.arc(pp[0], pp[1], gasNames[p[3]] === 'air' ? 1.6 : 2.0, 0, Math.PI * 2); ctx.fill();
  }}
  ctx.globalAlpha = 0.95;
  let selectedCount = 0;
  for (const [p, pp, selected] of projected) {{
    if (!selected) continue;
    selectedCount++;
    ctx.fillStyle = pointColor(p);
    const r = animation.trajectory.length ? 2.6 : 4.0;
    ctx.beginPath(); ctx.arc(pp[0], pp[1], r, 0, Math.PI * 2); ctx.fill();
  }}
  if (document.getElementById('showErrors').checked) {{
    ctx.globalAlpha = 0.92;
    for (const [p, pp] of projected) {{
      if (p[8] !== 1) continue;
      ctx.fillStyle = errorColor;
      ctx.beginPath(); ctx.arc(pp[0], pp[1], 3.1, 0, Math.PI * 2); ctx.fill();
    }}
  }}
  if (animation.trajectory.length) {{
    const trail = animation.trajectory.slice(0, animation.index + 1).map(p => [p, project(rotatePoint(p[0], p[1], p[2]), rect)]);
    ctx.globalAlpha = 0.95;
    ctx.strokeStyle = groupHighlight;
    ctx.lineWidth = 2.8;
    ctx.beginPath();
    for (let i = 0; i < trail.length; i++) {{
      const pp = trail[i][1];
      if (i === 0) ctx.moveTo(pp[0], pp[1]); else ctx.lineTo(pp[0], pp[1]);
    }}
    ctx.stroke();
    for (const [p, pp] of trail) {{
      ctx.fillStyle = groupHighlight;
      ctx.beginPath(); ctx.arc(pp[0], pp[1], 3.2, 0, Math.PI * 2); ctx.fill();
    }}
    const current = trail[trail.length - 1];
    if (current) {{
      const p = current[0], pp = current[1];
      ctx.fillStyle = currentPointColor;
      ctx.beginPath(); ctx.arc(pp[0], pp[1], 7.0, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = '#111827';
      ctx.lineWidth = 1.8;
      ctx.stroke();
      const group = selectedGroupNumber();
      setAnimStatus(`${{gasNames[p[3]]}}_${{group}}: ${{animation.index + 1}}/${{animation.trajectory.length}} points, elapsed ${{p[6].toFixed(1)}} s.`);
    }}
  }}
  ctx.globalAlpha = 1;
  if (highlight.mode !== 'none') setStatus(document.getElementById('status').textContent.replace(/ \\| selected points:.*/, '') + ` | selected points: ${{selectedCount}}`);
}}

canvas.addEventListener('pointerdown', e => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); }});
canvas.addEventListener('pointermove', e => {{
  if (!dragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  yaw += dx * 0.008;
  pitch = Math.max(-1.25, Math.min(1.25, pitch + dy * 0.008));
  requestDraw();
}});
canvas.addEventListener('pointerup', () => {{ dragging = false; }});
canvas.addEventListener('wheel', e => {{
  e.preventDefault();
  zoom = Math.max(0.45, Math.min(3.2, zoom * (e.deltaY > 0 ? 0.92 : 1.08)));
  requestDraw();
}}, {{ passive: false }});
canvas.addEventListener('dblclick', () => {{ yaw = -0.65; pitch = 0.38; zoom = 1.1; requestDraw(); }});
window.addEventListener('resize', resize);
resize();
</script>
</body>
</html>
"""


def build_feature_space(slug: str, sensors: list[str], label: str, record_map: dict[str, dict[str, object]], days: list[str]) -> tuple[Path, pd.DataFrame, dict[str, int], dict[str, object]]:
    data = load_baseline_corrected_points(sensors, record_map)
    normalized, scale = normalize_for_display(data, sensors)
    hyperplane = fit_lda_hyperplane(normalized, sensors)
    points = make_points_payload(normalized, sensors, days, hyperplane)
    counts = {gas: int((normalized["gas"] == gas).sum()) for gas in GASES}
    figures = OUT_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    out_path = figures / f"interactive_bc_{slug}_feature_space_hyperplane.html"
    out_path.write_text(make_html(label, sensors, points, counts, scale, days, hyperplane), encoding="utf-8")
    scale = scale.copy()
    scale.insert(0, "feature_space", slug)
    scale["total_points"] = len(points)
    return out_path, scale, counts, hyperplane


def pdf_styles() -> dict[str, ParagraphStyle]:
    font_path = Path(r"C:\Windows\Fonts\simhei.ttf")
    pdfmetrics.registerFont(TTFont("SimHei", str(font_path)))
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("TitleCN", parent=styles["Title"], fontName="SimHei", fontSize=20, leading=26, alignment=TA_CENTER, spaceAfter=12),
        "h1": ParagraphStyle("H1CN", parent=styles["Heading1"], fontName="SimHei", fontSize=15, leading=20, textColor=colors.HexColor("#1f4e79"), spaceBefore=12, spaceAfter=6),
        "h2": ParagraphStyle("H2CN", parent=styles["Heading2"], fontName="SimHei", fontSize=12.5, leading=17, textColor=colors.HexColor("#333333"), spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("BodyCN", parent=styles["BodyText"], fontName="SimHei", fontSize=10.2, leading=15.2, alignment=TA_LEFT, spaceAfter=5),
        "small": ParagraphStyle("SmallCN", parent=styles["BodyText"], fontName="SimHei", fontSize=8.8, leading=12.5, textColor=colors.HexColor("#555555"), spaceAfter=4),
    }


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def make_table(rows: list[list[str]], col_widths: list[float] | None = None) -> Table:
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "SimHei"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("LEADING", (0, 0), (-1, -1), 10.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9eaf7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#16324f")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b9c6d3")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def build_pdf(summary_rows: list[dict[str, object]]) -> Path:
    reports = OUT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    out_pdf = reports / "pre_ml_plan_raw_vs_features_and_hyperplane_cn.pdf"
    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.7 * cm, bottomMargin=1.7 * cm)
    s = pdf_styles()
    story = []
    story.append(p("正式训练前计划：原始数据、5s窗口、UCI-style特征与3D超平面对比", s["title"]))
    story.append(p("目的：在进入正式机器学习分类之前，先固定数据协议、特征方案、验证方式和可视化判断标准，避免模型仅记住响应强度或日期/窗口泄漏。", s["body"]))
    story.append(p("一、总体路线", s["h1"]))
    story.append(make_table([
        ["阶段", "要做什么", "使用数据", "输出", "主要依据"],
        ["1", "整理metadata：gas/day/group/duration/phase/source_file", "air/alcohol/acetone全部CSV", "trial metadata表", "可复现实验的基本要求"],
        ["2", "每个CSV用前60s clean air做baseline correction", "所有CSV", "baseline-corrected time series", "Vergara 2012的实验流程包含baseline阶段"],
        ["3", "切分baseline/gas exposure/recovery", "180s、125s、130s", "phase索引表", "Vergara 2012区分gas injection与cleaning phase"],
        ["4", "准备原始数据表示", "优先180s主数据", "full/gas-only/recovery-only原始数据集", "作为无特征提取baseline"],
        ["5", "准备5s/10s early-rise与recovery窗口", "180s、125s、130s", "window数据表", "Muezzinoglu 2009的early transient思想"],
        ["6", "提取UCI-style特征", "每个trial、每个sensor", "DeltaR、normalized DeltaR、EMA rising/recovery", "Vergara 2012的128维特征方案"],
        ["7", "画原始与特征后的feature space", "原始点与提取特征", "交互式3D HTML", "Muezzinoglu 2009使用3D transient feature space"],
        ["8", "设计强度泄漏检查", "尤其alcohol/acetone", "intensity-only与amplitude-normalized对照", "Vergara 2012强调regardless of concentration"],
        ["9", "固定validation方案", "按group/day/future day", "split表", "防止同trial窗口泄漏，评估drift"],
    ], [1.0 * cm, 3.2 * cm, 3.0 * cm, 3.0 * cm, 5.2 * cm]))
    story.append(p("二、5s窗口的处理原则", s["h1"]))
    story.append(p("现阶段不要把180s、125s、130s的所有5s窗口混合训练。原因是这些窗口处于不同物理阶段：180s的60s gas exposure包含早期上升、后续缓慢变化或平台趋势；125s只有5s气体暴露；130s只有10s气体暴露；recovery又是另一种动力学过程。", s["body"]))
    story.append(make_table([
        ["数据集", "180s", "125s", "130s", "目的"],
        ["early 5s", "gas-on后0-5s", "完整5s gas collection", "gas-on后0-5s", "只比较早期上升响应"],
        ["early 10s", "gas-on后0-10s", "不参与", "完整10s gas collection", "比较更长早期上升响应"],
        ["recovery 5s", "gas-off后0-5s", "gas-off后0-5s", "gas-off后0-5s", "比较恢复初期"],
        ["recovery 60s", "完整recovery", "完整recovery", "完整recovery", "比较下降轨迹，但需注明暴露时长不同"],
    ], [2.2 * cm, 3.0 * cm, 3.1 * cm, 3.1 * cm, 4.0 * cm]))
    story.append(p("三、UCI-style特征提取方案", s["h1"]))
    story.append(p("以Vergara et al. 2012为主要参考。每个传感器提8个特征：DeltaR、normalized DeltaR、rising phase的EMA peak(alpha=0.1/0.01/0.001)、recovery phase的EMA minimum(alpha=0.1/0.01/0.001)。如果使用6个传感器，则得到48维特征；如果使用16个传感器，则对应论文中的128维特征。", s["body"]))
    story.append(make_table([
        ["类别", "特征", "说明"],
        ["steady-state", "DeltaR = max(r[k]) - min(r[k])", "传统气敏响应强度特征"],
        ["steady-state normalized", "normalized DeltaR", "降低绝对幅度差异影响"],
        ["rising transient", "max EMA_alpha(rising), alpha=0.1/0.01/0.001", "吸附/上升过程动态特征"],
        ["recovery transient", "min EMA_alpha(recovery), alpha=0.1/0.01/0.001", "清洗/恢复过程动态特征"],
    ], [3.4 * cm, 5.5 * cm, 6.5 * cm]))
    story.append(p("四、是否加入DFT/FFT", s["h1"]))
    story.append(p("现阶段不建议作为主特征。当前数据主要是低频慢响应的gas exposure与recovery过程，用户给出的两篇核心论文主线是DeltaR与EMA transient，而不是频域特征。DFT/FFT可以作为附录探索低频能量，但不应作为本科论文主线。", s["body"]))
    story.append(p("五、原始数据与特征提取后的对比", s["h1"]))
    story.append(make_table([
        ["对比层面", "怎么做", "判断依据"],
        ["可视化", "画baseline-corrected原始点、UCI-style特征、early/recovery特征的3D feature space", "类间是否更分离、不同day是否更稳定"],
        ["信息保留", "比较强度归一化后alcohol/acetone是否仍能分开", "避免只学到响应强弱"],
        ["稳定性", "使用leave-date-out和future-day split作为后续训练验证规则", "漂移任务必须看跨日期泛化"],
        ["消融", "逐步去掉steady-state、rising、recovery特征", "若移除后表现下降，说明该特征族有贡献"],
    ], [3.0 * cm, 6.2 * cm, 6.2 * cm]))
    story.append(p("六、正式训练前必须固定的validation规则", s["h1"]))
    story.append(p("窗口数据不能按采样点随机划分；同一个CSV/trial切出的窗口不能同时进入train和test。建议先固定group-level split、leave-date-out、future-day test和单独的alcohol-vs-acetone子测试。", s["body"]))
    story.append(p("七、本次新增3D超平面结果", s["h1"]))
    rows = [["feature space", "Accuracy", "Balanced Acc.", "Alcohol recall", "Acetone recall", "Alcohol->Acetone", "Acetone->Alcohol"]]
    for row in summary_rows:
        rows.append([
            str(row["feature_space"]),
            f"{row['accuracy']:.4f}",
            f"{row['balanced_accuracy']:.4f}",
            f"{row['alcohol_recall']:.4f}",
            f"{row['acetone_recall']:.4f}",
            str(row["alcohol_as_acetone"]),
            str(row["acetone_as_alcohol"]),
        ])
    story.append(make_table(rows, [3.2 * cm, 2.1 * cm, 2.2 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm]))
    story.append(p("说明：这里的超平面是为了描述当前3D feature space中alcohol和acetone是否近似线性可分，使用的是Fisher LDA线性平面。它不是正式机器学习train/test结论，因为它在同一批可视化点上拟合并评估。正式结论仍需要group/date/future-day划分后的验证。", s["small"]))
    story.append(PageBreak())
    story.append(p("文献与依据", s["h1"]))
    refs = [
        "Vergara, A. et al. (2012). Chemical gas sensor drift compensation using classifier ensembles. Sensors and Actuators B: Chemical, 166-167, 320-329. DOI: 10.1016/j.snb.2012.01.074. 依据：UCI drift dataset、多气体分类、baseline/gas/recovery协议、每传感器8个特征、按month/batch评估drift。",
        "Muezzinoglu, M. K. et al. (2009). Acceleration of chemo-sensory information processing using transient features. Sensors and Actuators B: Chemical, 137, 507-512. DOI: 10.1016/j.snb.2008.10.065. 依据：early transient E_alpha、feature bank、3D transient feature space、用早期瞬态替代steady-state的思想。",
        "Electronic nose feature extraction review. 依据：steady-state、transient、window slicing、derivative/integral等气敏阵列特征提取方法的综述。",
        "计算机实验原则。依据：group-level split防止同trial泄漏；leave-date-out/future-day test用于drift泛化；ablation用于判断特征族边际贡献；intensity-only与amplitude-normalized对照用于检查强度捷径。",
    ]
    for ref in refs:
        story.append(p(ref, s["body"]))
    doc.build(story)
    return out_pdf


def main() -> None:
    record_map, days, record_df = load_record_mapping()
    (OUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "reports").mkdir(parents=True, exist_ok=True)
    summary_rows = []
    scale_rows = []
    outputs = []
    for slug, sensors, label in FEATURE_SPACES:
        out_path, scale, counts, hp = build_feature_space(slug, sensors, label, record_map, days)
        outputs.append(out_path)
        scale_rows.append(scale)
        row = {
            "feature_space": slug,
            "sensors": "/".join(sensors).upper(),
            "accuracy": hp["accuracy"],
            "balanced_accuracy": hp["balanced_accuracy"],
            "alcohol_recall": hp["alcohol_recall"],
            "acetone_recall": hp["acetone_recall"],
            "alcohol_total": hp["alcohol_total"],
            "acetone_total": hp["acetone_total"],
            "alcohol_correct": hp["alcohol_correct"],
            "acetone_correct": hp["acetone_correct"],
            "alcohol_as_acetone": hp["alcohol_as_acetone"],
            "acetone_as_alcohol": hp["acetone_as_alcohol"],
            "w1": hp["w"][0],
            "w2": hp["w"][1],
            "w3": hp["w"][2],
            "b": hp["b"],
            "method": "regularized Fisher LDA hyperplane on normalized 3D display coordinates",
        }
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(OUT_ROOT / "tables" / "alcohol_acetone_3d_hyperplane_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(scale_rows, ignore_index=True).to_csv(OUT_ROOT / "tables" / "interactive_bc_hyperplane_feature_space_scales.csv", index=False, encoding="utf-8-sig")
    record_df.to_csv(OUT_ROOT / "tables" / "record_day_sample_mapping.csv", index=False, encoding="utf-8-sig")
    out_pdf = build_pdf(summary_rows)
    print(out_pdf)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
