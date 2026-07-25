from __future__ import annotations

import json
import math
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


def load_record_mapping() -> tuple[dict[str, dict[str, object]], list[str]]:
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
        source_file = f"{gas}_{group_num}.csv"
        meta = {
            "sample_name": f"{gas}_{group_num}",
            "source_file": source_file,
            "gas": gas,
            "group": group_num,
            "day": current_day,
            "date": current_date,
        }
        mapping[source_file] = meta
        records.append(meta)

    def day_num(label: str) -> int:
        return int(re.search(r"\d+", label).group(0))

    days = sorted({str(record["day"]) for record in records}, key=day_num)
    return mapping, days


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
        for path in sorted(gas_dir.glob("*.csv"), key=natural_key):
            raw = pd.read_csv(path)
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


def fit_group_lda(group_df: pd.DataFrame, sensors: list[str]) -> dict[str, object]:
    subset = group_df[group_df["gas"].isin(["alcohol", "acetone"])].copy()
    x0 = subset.loc[subset["gas"] == "alcohol", sensors].to_numpy(float)
    x1 = subset.loc[subset["gas"] == "acetone", sensors].to_numpy(float)
    mu0 = x0.mean(axis=0)
    mu1 = x1.mean(axis=0)
    scatter = (x0 - mu0).T @ (x0 - mu0) + (x1 - mu1).T @ (x1 - mu1)
    pooled_cov = scatter / max(1, len(x0) + len(x1) - 2)
    reg = 1e-3 * np.trace(pooled_cov) / pooled_cov.shape[0]
    w = np.linalg.solve(pooled_cov + reg * np.eye(pooled_cov.shape[0]), mu1 - mu0)
    prior0 = len(x0) / (len(x0) + len(x1))
    prior1 = len(x1) / (len(x0) + len(x1))
    b = -0.5 * float(w @ (mu0 + mu1)) + math.log(prior1 / prior0)
    scores = subset[sensors].to_numpy(float) @ w + b
    pred = np.where(scores >= 0, "acetone", "alcohol")
    actual = subset["gas"].to_numpy()
    subset["score"] = scores
    subset["predicted_gas"] = pred
    subset["correct"] = pred == actual
    alcohol_total = int((actual == "alcohol").sum())
    acetone_total = int((actual == "acetone").sum())
    alcohol_correct = int(((actual == "alcohol") & (pred == "alcohol")).sum())
    acetone_correct = int(((actual == "acetone") & (pred == "acetone")).sum())
    alcohol_recall = alcohol_correct / alcohol_total if alcohol_total else 0.0
    acetone_recall = acetone_correct / acetone_total if acetone_total else 0.0
    return {
        "w": w.tolist(),
        "b": float(b),
        "accuracy": float((pred == actual).mean()),
        "balanced_accuracy": 0.5 * (alcohol_recall + acetone_recall),
        "alcohol_total": alcohol_total,
        "acetone_total": acetone_total,
        "alcohol_correct": alcohol_correct,
        "acetone_correct": acetone_correct,
        "alcohol_as_acetone": alcohol_total - alcohol_correct,
        "acetone_as_alcohol": acetone_total - acetone_correct,
        "alcohol_recall": alcohol_recall,
        "acetone_recall": acetone_recall,
        "predictions": subset,
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
            quad = []
            ok = True
            for a, c in [(vals[i], vals[j]), (vals[i + 1], vals[j]), (vals[i + 1], vals[j + 1]), (vals[i], vals[j + 1])]:
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


def build_group_centroids(data: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    grouped = (
        data.groupby(["gas", "source_file", "group", "day"], as_index=False)
        .agg({**{sensor: "mean" for sensor in sensors}, "elapsed_s": "count"})
        .rename(columns={"elapsed_s": "post60_points"})
    )
    return grouped


def make_payload(data: pd.DataFrame, group_df: pd.DataFrame, sensors: list[str], days: list[str], hp: dict[str, object]) -> tuple[list[list[float | int]], list[list[float | int | str]]]:
    pred_map = {
        row.source_file: (row.predicted_gas, bool(row.correct), float(row.score))
        for row in hp["predictions"].itertuples(index=False)
    }
    day_index = {day: idx for idx, day in enumerate(days)}
    points = []
    for row in data[sensors + ["gas", "day", "group", "elapsed_s", "source_file"]].itertuples(index=False, name=None):
        gas = str(row[3])
        pred_idx = -1
        group_error = 0
        score = 0.0
        if gas in {"alcohol", "acetone"}:
            pred, correct, score = pred_map[str(row[7])]
            pred_idx = GASES.index(pred)
            group_error = 0 if correct else 1
        points.append([
            round(float(row[0]), 5),
            round(float(row[1]), 5),
            round(float(row[2]), 5),
            GASES.index(gas),
            day_index.get(str(row[4]), -1),
            int(row[5]),
            round(float(row[6]), 4),
            pred_idx,
            group_error,
            round(float(score), 5),
        ])
    centroids = []
    for row in group_df[sensors + ["gas", "day", "group", "source_file", "post60_points"]].itertuples(index=False, name=None):
        gas = str(row[3])
        pred_idx = -1
        group_error = 0
        score = 0.0
        if gas in {"alcohol", "acetone"}:
            pred, correct, score = pred_map[str(row[6])]
            pred_idx = GASES.index(pred)
            group_error = 0 if correct else 1
        centroids.append([
            round(float(row[0]), 5),
            round(float(row[1]), 5),
            round(float(row[2]), 5),
            GASES.index(gas),
            day_index.get(str(row[4]), -1),
            int(row[5]),
            str(row[6]),
            int(row[7]),
            pred_idx,
            group_error,
            round(float(score), 5),
        ])
    return points, centroids


def make_html(title_suffix: str, sensors: list[str], points: list[list[float | int]], centroids: list[list[float | int | str]], counts: dict[str, int], group_counts: dict[str, int], scale: pd.DataFrame, days: list[str], hp: dict[str, object]) -> str:
    payload = {
        "points": points,
        "centroids": centroids,
        "gases": GASES,
        "colors": COLORS,
        "dayColors": DAY_COLORS,
        "counts": counts,
        "groupCounts": group_counts,
        "scale": scale.to_dict(orient="records"),
        "labels": [sensor.upper() for sensor in sensors],
        "titleSuffix": title_suffix,
        "days": days,
        "plane": {
            "w": hp["w"],
            "b": hp["b"],
            "cells": plane_cells(hp["w"], float(hp["b"])),
            "accuracy": hp["accuracy"],
            "balancedAccuracy": hp["balanced_accuracy"],
            "alcoholRecall": hp["alcohol_recall"],
            "acetoneRecall": hp["acetone_recall"],
            "alcoholCorrect": hp["alcohol_correct"],
            "alcoholTotal": hp["alcohol_total"],
            "acetoneCorrect": hp["acetone_correct"],
            "acetoneTotal": hp["acetone_total"],
            "alcoholAsAcetone": hp["alcohol_as_acetone"],
            "acetoneAsAlcohol": hp["acetone_as_alcohol"],
        },
    }
    payload_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Post-60s group-level {title_suffix} hyperplane</title>
<style>
html,body{{margin:0;height:100%;overflow:hidden;background:#fff;color:#22272e;font-family:Arial,sans-serif}}
#wrap{{position:fixed;inset:0;display:grid;grid-template-rows:auto 1fr}}
header{{padding:16px 24px 9px;border-bottom:1px solid #e6e8eb}}
h1{{margin:0;font-size:22px;font-weight:700;letter-spacing:0}}
p{{margin:5px 0 0;color:#656d76;font-size:14px}}
#stage{{position:relative;min-height:0}}
canvas{{position:absolute;inset:0;width:100%;height:100%;cursor:grab}}
canvas:active{{cursor:grabbing}}
#legend,#controls{{position:absolute;background:rgba(255,255,255,.92);border:1px solid #e6e8eb;border-radius:8px;box-shadow:0 8px 24px rgba(27,31,36,.08);font-size:14px}}
#legend{{top:18px;left:22px;padding:12px 14px}}
#controls{{top:18px;right:22px;width:340px;max-height:calc(100% - 88px);overflow:auto;padding:14px}}
.row{{display:flex;align-items:center;gap:8px;margin:6px 0}}
.dot{{width:10px;height:10px;border-radius:50%;display:inline-block}}
.section{{border-top:1px solid #e6e8eb;padding-top:12px;margin-top:12px}}
.section:first-child{{border-top:0;padding-top:0;margin-top:0}}
label{{display:block;color:#3f4650;font-size:12px;font-weight:700;margin:9px 0 4px}}
select,input{{width:100%;box-sizing:border-box;border:1px solid #d0d7de;border-radius:6px;padding:7px 8px;background:#fff;color:#22272e;font-size:13px}}
button{{border:1px solid #d0d7de;border-radius:6px;padding:7px 10px;background:#f6f8fa;color:#22272e;cursor:pointer;font-size:13px}}
button:hover{{background:#eef2f6}}
.buttons{{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}}
.active{{background:#0969da;border-color:#0969da;color:#fff}}
.hintline,#status,#planeStats{{color:#3f4650;font-size:12px;line-height:1.4;margin-top:8px}}
.toggle{{display:flex;align-items:center;gap:8px;margin-top:8px}}
.toggle input{{width:auto}}
#hint{{position:absolute;right:22px;bottom:18px;color:#656d76;font-size:13px;background:rgba(255,255,255,.82);padding:8px 10px;border-radius:6px}}
</style>
</head>
<body>
<div id="wrap">
<header>
<h1>Post-60s baseline-corrected {title_suffix} feature space</h1>
<p>Only samples after 60 s are displayed. Alcohol/acetone hyperplane is fitted and evaluated on one centroid per trial/group, not on all sampling points.</p>
</header>
<div id="stage">
<canvas id="canvas"></canvas>
<div id="legend"></div>
<div id="controls">
<div class="section">
<strong>Group-level Alcohol/Acetone Hyperplane</strong>
<div class="toggle"><input id="showPlane" type="checkbox" checked><span>show hyperplane</span></div>
<div class="toggle"><input id="showPoints" type="checkbox" checked><span>show post-60s samples</span></div>
<div class="toggle"><input id="showCentroids" type="checkbox" checked><span>show group centroids</span></div>
<div class="toggle"><input id="showPred" type="checkbox"><span>color alcohol/acetone by predicted side</span></div>
<div class="toggle"><input id="showErrors" type="checkbox" checked><span>emphasize wrong groups</span></div>
<div id="planeStats"></div>
</div>
<div class="section">
<strong>Highlight by Day</strong>
<label for="daySelect">Day</label><select id="daySelect"></select>
<div class="buttons"><button id="modeDay">Apply Day</button></div>
</div>
<div class="section">
<strong>Highlight Gas Groups</strong>
<label for="groupGas">Gas</label><select id="groupGas"></select>
<label for="groupsInput">Group number(s)</label><input id="groupsInput" placeholder="e.g. 1, 4, 10-12">
<div class="buttons"><button id="modeGroups">Apply Groups</button><button id="clearBtn">Clear</button></div>
</div>
<div id="status">No highlight selected.</div>
</div>
<div id="hint">Drag to rotate · wheel to zoom · double click to reset</div>
</div>
</div>
<script>
const DATA={payload_json};
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),legend=document.getElementById('legend');
const gasNames=DATA.gases,gasIndex=Object.fromEntries(DATA.gases.map((g,i)=>[g,i])),dayIndex=Object.fromEntries(DATA.days.map((d,i)=>[d,i]));
let yaw=-0.65,pitch=0.38,zoom=1.1,dragging=false,lastX=0,lastY=0,pending=false;
let highlight={{mode:'none',day:-1,gas:-1,groups:new Set()}};
const groupHighlight='#ff8c00',predAlcohol='#1f9d55',predAcetone='#d94841',errorColor='#111827';
function pct(v){{return (v*100).toFixed(2)+'%'}}
document.getElementById('planeStats').innerHTML=`<div>Group accuracy: <strong>${{pct(DATA.plane.accuracy)}}</strong></div>
<div>Balanced accuracy: <strong>${{pct(DATA.plane.balancedAccuracy)}}</strong></div>
<div>Alcohol groups: ${{DATA.plane.alcoholCorrect}}/${{DATA.plane.alcoholTotal}} (${{pct(DATA.plane.alcoholRecall)}})</div>
<div>Acetone groups: ${{DATA.plane.acetoneCorrect}}/${{DATA.plane.acetoneTotal}} (${{pct(DATA.plane.acetoneRecall)}})</div>
<div>Alcohol->acetone: ${{DATA.plane.alcoholAsAcetone}}; acetone->alcohol: ${{DATA.plane.acetoneAsAlcohol}}</div>
<div class="hintline">Plane uses post-60s group centroids. Samples are shown only for visualization.</div>`;
function optionList(sel,vals){{sel.innerHTML=vals.map(v=>`<option value="${{v}}">${{v}}</option>`).join('')}}
optionList(document.getElementById('daySelect'),DATA.days);optionList(document.getElementById('groupGas'),DATA.gases);
legend.innerHTML=DATA.gases.map(g=>`<div class="row"><span class="dot" style="background:${{DATA.colors[g]}}"></span><span>${{g}} points=${{DATA.counts[g]}}, groups=${{DATA.groupCounts[g]||0}}</span></div>`).join('')+
`<div class="section"><div class="row"><span class="dot" style="background:${{groupHighlight}}"></span><span>selected groups</span></div><div class="row"><span class="dot" style="background:rgba(9,105,218,.5)"></span><span>hyperplane</span></div><div class="row"><span class="dot" style="background:${{errorColor}}"></span><span>wrong group</span></div></div>`;
function parseGroups(text){{const groups=new Set();for(const part of text.split(',')){{const s=part.trim();if(!s)continue;const m=s.match(/^(\\d+)\\s*-\\s*(\\d+)$/);if(m){{const a=Number(m[1]),b=Number(m[2]);for(let i=Math.min(a,b);i<=Math.max(a,b);i++)groups.add(i)}}else if(/^\\d+$/.test(s))groups.add(Number(s))}}return groups}}
function setStatus(t){{document.getElementById('status').textContent=t}}
function activeButton(id){{for(const btn of ['modeDay','modeGroups'])document.getElementById(btn).classList.toggle('active',btn===id)}}
for(const id of ['showPlane','showPoints','showCentroids','showPred','showErrors'])document.getElementById(id).addEventListener('change',requestDraw);
document.getElementById('modeDay').addEventListener('click',()=>{{const d=document.getElementById('daySelect').value;highlight={{mode:'day',day:dayIndex[d],gas:-1,groups:new Set()}};setStatus(`Highlighting ${{d}}.`);activeButton('modeDay');requestDraw()}});
document.getElementById('modeGroups').addEventListener('click',()=>{{const gas=document.getElementById('groupGas').value;const groups=parseGroups(document.getElementById('groupsInput').value);highlight={{mode:'groups',day:-1,gas:gasIndex[gas],groups}};setStatus(`Highlighting ${{gas}} group(s): ${{[...groups].sort((a,b)=>a-b).join(', ')||'none'}}.`);activeButton('modeGroups');requestDraw()}});
document.getElementById('clearBtn').addEventListener('click',()=>{{highlight={{mode:'none',day:-1,gas:-1,groups:new Set()}};setStatus('No highlight selected.');activeButton('');requestDraw()}});
function requestDraw(){{if(pending)return;pending=true;requestAnimationFrame(()=>{{pending=false;draw()}})}}
function resize(){{const dpr=Math.max(1,window.devicePixelRatio||1),rect=canvas.getBoundingClientRect();canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);requestDraw()}}
function rotatePoint(x,y,z){{x-=.5;y-=.5;z-=.5;const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);const x1=x*cy-y*sy,y1=x*sy+y*cy,z1=z,y2=y1*cp-z1*sp,z2=y1*sp+z1*cp;return[x1,y2,z2]}}
function project(p,rect){{const scale=Math.min(rect.width,rect.height)*.78*zoom;return[rect.width*.48+p[0]*scale,rect.height*.56-p[1]*scale,p[2]]}}
function drawAxes(rect){{const corners=[[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],edges=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]],pts=corners.map(c=>project(rotatePoint(...c),rect));ctx.lineWidth=1;ctx.strokeStyle='#d9dee3';for(const [a,b] of edges){{ctx.beginPath();ctx.moveTo(pts[a][0],pts[a][1]);ctx.lineTo(pts[b][0],pts[b][1]);ctx.stroke()}}const axes=[[[0,0,0],[1.12,0,0],DATA.labels[0]],[[0,0,0],[0,1.12,0],DATA.labels[1]],[[0,0,0],[0,0,1.12],DATA.labels[2]]];ctx.strokeStyle='#4f565e';ctx.fillStyle='#4f565e';ctx.font='15px Arial';for(const [a,b,label] of axes){{const pa=project(rotatePoint(...a),rect),pb=project(rotatePoint(...b),rect);ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(pa[0],pa[1]);ctx.lineTo(pb[0],pb[1]);ctx.stroke();ctx.fillText(label,pb[0]+5,pb[1]-5)}}}}
function drawPlane(rect){{if(!document.getElementById('showPlane').checked)return;const cells=DATA.plane.cells.map(cell=>[cell.map(p=>project(rotatePoint(p[0],p[1],p[2]),rect)),0]).sort((a,b)=>a[1]-b[1]);ctx.globalAlpha=.2;ctx.fillStyle='#0969da';ctx.strokeStyle='#0969da';ctx.lineWidth=.7;for(const [cell] of cells){{ctx.beginPath();ctx.moveTo(cell[0][0],cell[0][1]);for(let i=1;i<cell.length;i++)ctx.lineTo(cell[i][0],cell[i][1]);ctx.closePath();ctx.fill();ctx.stroke()}}ctx.globalAlpha=1}}
function isHighlighted(p){{if(highlight.mode==='day')return p[4]===highlight.day;if(highlight.mode==='groups')return p[3]===highlight.gas&&highlight.groups.has(p[5]);return false}}
function color(p){{if(isHighlighted(p))return groupHighlight;if(document.getElementById('showPred').checked&&p[7]>=0)return p[7]===gasIndex.acetone?predAcetone:predAlcohol;return DATA.colors[gasNames[p[3]]]}}
function drawSet(arr,rBase,alpha,centroid){{ctx.globalAlpha=alpha;for(const p of arr){{if(document.getElementById('showErrors').checked&&p[8]===1&&!centroid)continue;const pp=project(rotatePoint(p[0],p[1],p[2]),canvas.getBoundingClientRect());ctx.fillStyle=color(p);ctx.beginPath();ctx.arc(pp[0],pp[1],isHighlighted(p)?rBase+1.8:rBase,0,Math.PI*2);ctx.fill();if(centroid){{ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke()}}}}ctx.globalAlpha=1}}
function drawErrors(arr,r){{if(!document.getElementById('showErrors').checked)return;for(const p of arr){{if(p[8]!==1)continue;const pp=project(rotatePoint(p[0],p[1],p[2]),canvas.getBoundingClientRect());ctx.fillStyle=errorColor;ctx.beginPath();ctx.arc(pp[0],pp[1],r,0,Math.PI*2);ctx.fill()}}}}
function draw(){{const rect=canvas.getBoundingClientRect();ctx.clearRect(0,0,rect.width,rect.height);drawAxes(rect);drawPlane(rect);if(document.getElementById('showPoints').checked)drawSet(DATA.points,1.8,.27,false);if(document.getElementById('showCentroids').checked){{drawSet(DATA.centroids,5,.92,true);drawErrors(DATA.centroids,7)}}}}
canvas.addEventListener('pointerdown',e=>{{dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId)}});
canvas.addEventListener('pointermove',e=>{{if(!dragging)return;const dx=e.clientX-lastX,dy=e.clientY-lastY;lastX=e.clientX;lastY=e.clientY;yaw+=dx*.008;pitch=Math.max(-1.25,Math.min(1.25,pitch+dy*.008));requestDraw()}});
canvas.addEventListener('pointerup',()=>{{dragging=false}});
canvas.addEventListener('wheel',e=>{{e.preventDefault();zoom=Math.max(.45,Math.min(3.2,zoom*(e.deltaY>0?.92:1.08)));requestDraw()}},{{passive:false}});
canvas.addEventListener('dblclick',()=>{{yaw=-.65;pitch=.38;zoom=1.1;requestDraw()}});
window.addEventListener('resize',resize);resize();
</script>
</body>
</html>"""


def build_feature_space(slug: str, sensors: list[str], label: str, record_map: dict[str, dict[str, object]], days: list[str]) -> tuple[Path, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    all_data = load_baseline_corrected_points(sensors, record_map)
    post60 = all_data[all_data["elapsed_s"] >= 60].copy()
    normalized, scale = normalize_for_display(post60, sensors)
    group_df = build_group_centroids(normalized, sensors)
    hp = fit_group_lda(group_df, sensors)
    points, centroids = make_payload(normalized, group_df, sensors, days, hp)
    counts = {gas: int((normalized["gas"] == gas).sum()) for gas in GASES}
    group_counts = {gas: int((group_df["gas"] == gas).sum()) for gas in GASES}
    figures = OUT_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    out_path = figures / f"interactive_bc_post60_group_hyperplane_{slug}.html"
    out_path.write_text(make_html(label, sensors, points, centroids, counts, group_counts, scale, days, hp), encoding="utf-8")
    predictions = hp["predictions"].copy()
    predictions.insert(0, "feature_space", slug)
    for i, sensor in enumerate(sensors):
        predictions[f"w_{sensor}"] = hp["w"][i]
    predictions["b"] = hp["b"]
    scale.insert(0, "feature_space", slug)
    return out_path, scale, predictions, hp


def main() -> None:
    record_map, days = load_record_mapping()
    (OUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    outputs = []
    scales = []
    predictions = []
    summaries = []
    for slug, sensors, label in FEATURE_SPACES:
        out_path, scale, pred, hp = build_feature_space(slug, sensors, label, record_map, days)
        outputs.append(out_path)
        scales.append(scale)
        predictions.append(pred)
        summaries.append({
            "feature_space": slug,
            "sensors": "/".join(sensors).upper(),
            "unit": "one post-60s centroid per CSV trial/group",
            "accuracy": hp["accuracy"],
            "balanced_accuracy": hp["balanced_accuracy"],
            "alcohol_total_groups": hp["alcohol_total"],
            "acetone_total_groups": hp["acetone_total"],
            "alcohol_correct_groups": hp["alcohol_correct"],
            "acetone_correct_groups": hp["acetone_correct"],
            "alcohol_as_acetone_groups": hp["alcohol_as_acetone"],
            "acetone_as_alcohol_groups": hp["acetone_as_alcohol"],
            "alcohol_recall": hp["alcohol_recall"],
            "acetone_recall": hp["acetone_recall"],
            "method": "regularized Fisher LDA hyperplane fitted on post-60s group centroids",
        })
    pd.DataFrame(summaries).to_csv(OUT_ROOT / "tables" / "alcohol_acetone_post60_group_hyperplane_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(predictions, ignore_index=True).to_csv(OUT_ROOT / "tables" / "alcohol_acetone_post60_group_hyperplane_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(scales, ignore_index=True).to_csv(OUT_ROOT / "tables" / "interactive_bc_post60_group_hyperplane_scales.csv", index=False, encoding="utf-8-sig")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
