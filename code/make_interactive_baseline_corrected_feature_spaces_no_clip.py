from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


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
    raw_min = np.nanmin(values, axis=0)
    raw_max = np.nanmax(values, axis=0)
    raw_span = np.where(raw_max - raw_min == 0, 1.0, raw_max - raw_min)
    padding = raw_span * 0.03
    lo = raw_min - padding
    hi = raw_max + padding
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    out = data.copy()
    out[sensors] = (values - lo) / span
    scale = pd.DataFrame({
        "sensor": sensors,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "display_min": lo,
        "display_max": hi,
        "padding": padding,
        "method": "full min/max linear scaling with 3% padding; no clipping",
    })
    return out, scale


def make_html(title_suffix: str, sensors: list[str], points: list[list[float | int]], counts: dict[str, int], scale: pd.DataFrame, days: list[str]) -> str:
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
    }
    payload_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Interactive baseline-corrected {title_suffix} feature space</title>
<style>
  html, body {{ margin: 0; height: 100%; overflow: hidden; background: #ffffff; color: #22272e; font-family: Arial, sans-serif; }}
  #wrap {{ position: fixed; inset: 0; display: grid; grid-template-rows: auto 1fr; }}
  header {{ padding: 18px 24px 10px; border-bottom: 1px solid #e6e8eb; }}
  h1 {{ margin: 0; font-size: 22px; font-weight: 700; letter-spacing: 0; }}
  p {{ margin: 5px 0 0; color: #656d76; font-size: 14px; }}
  #stage {{ position: relative; min-height: 0; }}
  canvas {{ position: absolute; inset: 0; width: 100%; height: 100%; cursor: grab; }}
  canvas:active {{ cursor: grabbing; }}
  #legend, #controls {{ position: absolute; background: rgba(255,255,255,.9); border: 1px solid #e6e8eb; border-radius: 8px; box-shadow: 0 8px 24px rgba(27,31,36,.08); font-size: 14px; }}
  #legend {{ top: 18px; left: 22px; padding: 12px 14px; }}
  #controls {{ top: 18px; right: 22px; width: 315px; max-height: calc(100% - 88px); overflow: auto; padding: 14px; }}
  .row {{ display: flex; align-items: center; gap: 8px; margin: 6px 0; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  .section {{ border-top: 1px solid #e6e8eb; padding-top: 12px; margin-top: 12px; }}
  .section:first-child {{ border-top: 0; padding-top: 0; margin-top: 0; }}
  label {{ display: block; color: #3f4650; font-size: 12px; font-weight: 700; margin: 9px 0 4px; }}
  select, input {{ width: 100%; box-sizing: border-box; border: 1px solid #d0d7de; border-radius: 6px; padding: 7px 8px; background: #fff; color: #22272e; font-size: 13px; }}
  select[multiple] {{ min-height: 116px; }}
  button {{ border: 1px solid #d0d7de; border-radius: 6px; padding: 7px 10px; background: #f6f8fa; color: #22272e; cursor: pointer; font-size: 13px; }}
  button:hover {{ background: #eef2f6; }}
  button:disabled {{ color: #8c959f; cursor: not-allowed; background: #f6f8fa; }}
  .buttons {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }}
  .active {{ background: #0969da; border-color: #0969da; color: #fff; }}
  .active:disabled {{ background: #0969da; color: #fff; }}
  .hintline {{ color: #656d76; font-size: 12px; line-height: 1.35; margin-top: 6px; }}
  #animStatus {{ color: #656d76; font-size: 12px; line-height: 1.35; margin-top: 8px; }}
  #status {{ margin-top: 10px; color: #3f4650; font-size: 12px; line-height: 1.35; }}
  #hint {{ position: absolute; right: 22px; bottom: 18px; color: #656d76; font-size: 13px; background: rgba(255,255,255,.82); padding: 8px 10px; border-radius: 6px; }}
</style>
</head>
<body>
<div id="wrap">
  <header>
    <h1>Interactive baseline-corrected {title_suffix} feature space</h1>
    <p>All current alcohol / acetone / air CSV files are included. Each point is an instantaneous sample after subtracting that file's first 60 s baseline. Display uses full min/max scaling with padding and no clipping.</p>
  </header>
  <div id="stage">
    <canvas id="canvas"></canvas>
    <div id="legend"></div>
    <div id="controls">
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
          <div class="hintline">Available when exactly one group is selected. The path follows the 180 s trial order from start to end.</div>
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
let yaw = -0.65, pitch = 0.38, zoom = 1.1;
let dragging = false, lastX = 0, lastY = 0, pending = false;
let highlight = {{ mode: 'none', day: -1, gas: -1, groups: new Set(), days: new Set() }};
let animation = {{ playing: false, index: 0, lastTs: 0, trajectory: [] }};

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

document.getElementById('speedRange').addEventListener('input', () => {{
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

function isHighlighted(p) {{
  if (highlight.mode === 'day') return p[4] === highlight.day;
  if (highlight.mode === 'groups') return p[3] === highlight.gas && highlight.groups.has(p[5]);
  if (highlight.mode === 'gasDays') return p[3] === highlight.gas && highlight.days.has(p[4]);
  return false;
}}

function highlightColor(p) {{
  if (highlight.mode === 'day') return DATA.dayColors[gasNames[p[3]]];
  if (highlight.mode === 'groups') return groupHighlight;
  if (highlight.mode === 'gasDays') return gasDayHighlight;
  return DATA.colors[gasNames[p[3]]];
}}

function draw() {{
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  drawAxes(rect);
  const projected = DATA.points.map(p => [p, project(rotatePoint(p[0], p[1], p[2]), rect), isHighlighted(p)]);
  ctx.globalAlpha = 0.32;
  for (const [p, pp, selected] of projected) {{
    if (selected) continue;
    ctx.fillStyle = DATA.colors[gasNames[p[3]]];
    ctx.beginPath(); ctx.arc(pp[0], pp[1], gasNames[p[3]] === 'air' ? 1.7 : 2.1, 0, Math.PI * 2); ctx.fill();
  }}
  ctx.globalAlpha = 0.95;
  let selectedCount = 0;
  for (const [p, pp, selected] of projected) {{
    if (!selected) continue;
    selectedCount++;
    ctx.fillStyle = highlightColor(p);
    const r = animation.trajectory.length ? 2.6 : 4.0;
    ctx.beginPath(); ctx.arc(pp[0], pp[1], r, 0, Math.PI * 2); ctx.fill();
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


def build_feature_space(slug: str, sensors: list[str], label: str, record_map: dict[str, dict[str, object]], days: list[str]) -> tuple[Path, pd.DataFrame, dict[str, int]]:
    data = load_baseline_corrected_points(sensors, record_map)
    normalized, scale = normalize_for_display(data, sensors)
    day_index = {day: idx for idx, day in enumerate(days)}
    points = []
    for row in normalized[sensors + ["gas", "day", "group", "elapsed_s"]].itertuples(index=False, name=None):
        points.append([
            round(float(row[0]), 5),
            round(float(row[1]), 5),
            round(float(row[2]), 5),
            GASES.index(row[3]),
            day_index.get(str(row[4]), -1),
            int(row[5]),
            round(float(row[6]), 4),
        ])
    counts = {gas: int((normalized["gas"] == gas).sum()) for gas in GASES}

    figures = OUT_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    out_path = figures / f"interactive_bc_no_clip_{slug}_feature_space.html"
    out_path.write_text(make_html(label, sensors, points, counts, scale, days), encoding="utf-8")

    scale = scale.copy()
    scale.insert(0, "feature_space", slug)
    scale["total_points"] = len(points)
    return out_path, scale, counts


def main() -> None:
    record_map, days, record_df = load_record_mapping()
    tables = OUT_ROOT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "code").mkdir(parents=True, exist_ok=True)

    scales = []
    summaries = []
    outputs = []
    for slug, sensors, label in FEATURE_SPACES:
        out_path, scale, counts = build_feature_space(slug, sensors, label, record_map, days)
        outputs.append(out_path)
        scales.append(scale)
        for gas, count in counts.items():
            summaries.append({"feature_space": slug, "gas": gas, "points": count})

    pd.concat(scales, ignore_index=True).to_csv(tables / "interactive_bc_no_clip_feature_space_scales.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(tables / "interactive_bc_no_clip_feature_space_summary.csv", index=False, encoding="utf-8-sig")
    record_df.to_csv(tables / "record_day_sample_mapping.csv", index=False, encoding="utf-8-sig")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
