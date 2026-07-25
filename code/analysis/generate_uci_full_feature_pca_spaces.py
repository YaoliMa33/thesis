from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


OUT_ROOT = Path(r"D:\thesis")
FEATURE_TABLE = OUT_ROOT / "tables" / "uci_style_trial_level_features.csv"
GASES = ["air", "alcohol", "acetone"]
COLORS = {"air": "#2d72b2", "alcohol": "#52a046", "acetone": "#c35359"}
DAY_COLORS = {"air": "#00a6d6", "alcohol": "#f2b705", "acetone": "#8b5cf6"}
TRIPLETS = [
    ("s1_s2_s3", ["s1", "s2", "s3"], "UCI full features S1/S2/S3 -> PCA 3D"),
    ("s2_s3_s5", ["s2", "s3", "s5"], "UCI full features S2/S3/S5 -> PCA 3D"),
]
FEATURE_FAMILIES = [
    "deltaR_{sensor}",
    "normDeltaR_{sensor}",
    "riseEMA_a0.1_{sensor}",
    "riseEMA_a0.01_{sensor}",
    "riseEMA_a0.001_{sensor}",
    "recEMA_a0.1_{sensor}",
    "recEMA_a0.01_{sensor}",
    "recEMA_a0.001_{sensor}",
]


def sensor_feature_columns(sensors: list[str]) -> list[str]:
    return [pattern.format(sensor=sensor) for sensor in sensors for pattern in FEATURE_FAMILIES]


def day_sort_key(day: str) -> int:
    digits = "".join(ch for ch in str(day) if ch.isdigit())
    return int(digits) if digits else 10**9


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(std == 0, 1.0, std)
    return (x - mean) / std, mean, std


def pca_3d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z, _, _ = standardize(x)
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    coords = z @ vt[:3].T
    explained = (s**2) / max(1, z.shape[0] - 1)
    ratio = explained / explained.sum()
    return coords, vt[:3], ratio[:3]


def scale_no_clip(coords: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    raw_min = np.nanmin(coords, axis=0)
    raw_max = np.nanmax(coords, axis=0)
    span = np.where(raw_max - raw_min == 0, 1.0, raw_max - raw_min)
    padding = span * 0.03
    display_min = raw_min - padding
    display_max = raw_max + padding
    scaled = (coords - display_min) / np.where(display_max - display_min == 0, 1.0, display_max - display_min)
    scale = pd.DataFrame({
        "axis": ["PC1", "PC2", "PC3"],
        "raw_min": raw_min,
        "raw_max": raw_max,
        "display_min": display_min,
        "display_max": display_max,
        "padding": padding,
        "method": "full min/max linear scaling with 3% padding; no clipping",
    })
    return scaled, scale


def fit_lda(coords: np.ndarray, labels: np.ndarray, files: np.ndarray) -> dict[str, object]:
    mask = np.isin(labels, ["alcohol", "acetone"])
    x = coords[mask]
    y = labels[mask]
    f = files[mask]
    x0 = x[y == "alcohol"]
    x1 = x[y == "acetone"]
    mu0 = x0.mean(axis=0)
    mu1 = x1.mean(axis=0)
    scatter = (x0 - mu0).T @ (x0 - mu0) + (x1 - mu1).T @ (x1 - mu1)
    pooled_cov = scatter / max(1, len(x0) + len(x1) - 2)
    reg = 1e-3 * np.trace(pooled_cov) / pooled_cov.shape[0]
    w = np.linalg.solve(pooled_cov + reg * np.eye(pooled_cov.shape[0]), mu1 - mu0)
    prior0 = len(x0) / (len(x0) + len(x1))
    prior1 = len(x1) / (len(x0) + len(x1))
    b = -0.5 * float(w @ (mu0 + mu1)) + math.log(prior1 / prior0)
    scores = x @ w + b
    pred = np.where(scores >= 0, "acetone", "alcohol")
    alcohol_total = int((y == "alcohol").sum())
    acetone_total = int((y == "acetone").sum())
    alcohol_correct = int(((y == "alcohol") & (pred == "alcohol")).sum())
    acetone_correct = int(((y == "acetone") & (pred == "acetone")).sum())
    alcohol_recall = alcohol_correct / alcohol_total if alcohol_total else 0.0
    acetone_recall = acetone_correct / acetone_total if acetone_total else 0.0
    pred_df = pd.DataFrame({
        "source_file": f,
        "gas": y,
        "score": scores,
        "predicted_gas": pred,
        "correct": pred == y,
    })
    return {
        "w": w.tolist(),
        "b": float(b),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": 0.5 * (alcohol_recall + acetone_recall),
        "alcohol_total": alcohol_total,
        "acetone_total": acetone_total,
        "alcohol_correct": alcohol_correct,
        "acetone_correct": acetone_correct,
        "alcohol_as_acetone": alcohol_total - alcohol_correct,
        "acetone_as_alcohol": acetone_total - acetone_correct,
        "alcohol_recall": alcohol_recall,
        "acetone_recall": acetone_recall,
        "predictions": pred_df,
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


def make_points(df: pd.DataFrame, scaled: np.ndarray, hp: dict[str, object], days: list[str]) -> list[list[float | int | str]]:
    pred_map = {
        row.source_file: (row.predicted_gas, bool(row.correct), float(row.score))
        for row in hp["predictions"].itertuples(index=False)
    }
    day_index = {day: idx for idx, day in enumerate(days)}
    points = []
    for i, row in enumerate(df.itertuples(index=False)):
        gas = str(row.gas)
        pred_idx = -1
        error = 0
        score = 0.0
        if gas in {"alcohol", "acetone"}:
            pred, correct, score = pred_map[str(row.source_file)]
            pred_idx = GASES.index(pred)
            error = 0 if correct else 1
        points.append([
            round(float(scaled[i, 0]), 5),
            round(float(scaled[i, 1]), 5),
            round(float(scaled[i, 2]), 5),
            GASES.index(gas),
            day_index.get(str(row.day), -1),
            int(row.group),
            str(row.source_file),
            pred_idx,
            error,
            round(float(score), 5),
        ])
    return points


def make_html(title: str, points: list[list[float | int | str]], days: list[str], counts: dict[str, int], hp: dict[str, object], explained: np.ndarray, sensors: list[str]) -> str:
    labels = [
        f"PC1 ({explained[0] * 100:.1f}%)",
        f"PC2 ({explained[1] * 100:.1f}%)",
        f"PC3 ({explained[2] * 100:.1f}%)",
    ]
    payload = {
        "points": points,
        "gases": GASES,
        "colors": COLORS,
        "dayColors": DAY_COLORS,
        "counts": counts,
        "labels": labels,
        "title": title,
        "days": days,
        "sensors": [s.upper() for s in sensors],
        "featuresPerSensor": FEATURE_FAMILIES,
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
    data_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
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
#controls{{top:18px;right:22px;width:350px;max-height:calc(100% - 88px);overflow:auto;padding:14px}}
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
<h1>{title}</h1>
<p>Each point is one CSV trial/group. Axes are PCA components from all 8 UCI-style features per selected sensor. Display uses full min/max scaling with 3% padding and no clipping.</p>
</header>
<div id="stage">
<canvas id="canvas"></canvas>
<div id="legend"></div>
<div id="controls">
<div class="section">
<strong>Full-feature PCA Space</strong>
<div id="planeStats"></div>
<div class="toggle"><input id="showPlane" type="checkbox" checked><span>show alcohol/acetone hyperplane</span></div>
<div class="toggle"><input id="showPred" type="checkbox"><span>color alcohol/acetone by predicted side</span></div>
<div class="toggle"><input id="showErrors" type="checkbox" checked><span>emphasize wrong groups</span></div>
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
const DATA={data_json};
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),legend=document.getElementById('legend');
const gasNames=DATA.gases,gasIndex=Object.fromEntries(DATA.gases.map((g,i)=>[g,i])),dayIndex=Object.fromEntries(DATA.days.map((d,i)=>[d,i]));
let yaw=-0.65,pitch=0.38,zoom=1.1,dragging=false,lastX=0,lastY=0,pending=false;
let highlight={{mode:'none',day:-1,gas:-1,groups:new Set()}};
const groupHighlight='#ff8c00',predAlcohol='#1f9d55',predAcetone='#d94841',errorColor='#111827';
function pct(v){{return (v*100).toFixed(2)+'%'}}
document.getElementById('planeStats').innerHTML=`<div>Sensors: <strong>${{DATA.sensors.join('/')}}</strong></div>
<div>Axes: ${{DATA.labels.join(', ')}}</div>
<div>Group BAcc: <strong>${{pct(DATA.plane.balancedAccuracy)}}</strong></div>
<div>Alcohol: ${{DATA.plane.alcoholCorrect}}/${{DATA.plane.alcoholTotal}}; Acetone: ${{DATA.plane.acetoneCorrect}}/${{DATA.plane.acetoneTotal}}</div>
<div>Alcohol->acetone: ${{DATA.plane.alcoholAsAcetone}}; acetone->alcohol: ${{DATA.plane.acetoneAsAlcohol}}</div>
<div class="hintline">PCA is fitted on all gases for visualization. Hyperplane is descriptive in-sample alcohol/acetone separability in this 3D PCA space.</div>`;
function optionList(sel,vals){{sel.innerHTML=vals.map(v=>`<option value="${{v}}">${{v}}</option>`).join('')}}
optionList(document.getElementById('daySelect'),DATA.days);optionList(document.getElementById('groupGas'),DATA.gases);
legend.innerHTML=DATA.gases.map(g=>`<div class="row"><span class="dot" style="background:${{DATA.colors[g]}}"></span><span>${{g}} groups=${{DATA.counts[g]||0}}</span></div>`).join('')+
`<div class="section"><div class="row"><span class="dot" style="background:${{groupHighlight}}"></span><span>selected groups</span></div><div class="row"><span class="dot" style="background:rgba(9,105,218,.5)"></span><span>hyperplane</span></div><div class="row"><span class="dot" style="background:${{errorColor}}"></span><span>wrong group</span></div></div>`;
function parseGroups(text){{const groups=new Set();for(const part of text.split(',')){{const s=part.trim();if(!s)continue;const m=s.match(/^(\\d+)\\s*-\\s*(\\d+)$/);if(m){{const a=Number(m[1]),b=Number(m[2]);for(let i=Math.min(a,b);i<=Math.max(a,b);i++)groups.add(i)}}else if(/^\\d+$/.test(s))groups.add(Number(s))}}return groups}}
function setStatus(t){{document.getElementById('status').textContent=t}}
function activeButton(id){{for(const btn of ['modeDay','modeGroups'])document.getElementById(btn).classList.toggle('active',btn===id)}}
for(const id of ['showPlane','showPred','showErrors'])document.getElementById(id).addEventListener('change',requestDraw);
document.getElementById('modeDay').addEventListener('click',()=>{{const d=document.getElementById('daySelect').value;highlight={{mode:'day',day:dayIndex[d],gas:-1,groups:new Set()}};setStatus(`Highlighting ${{d}}.`);activeButton('modeDay');requestDraw()}});
document.getElementById('modeGroups').addEventListener('click',()=>{{const gas=document.getElementById('groupGas').value;const groups=parseGroups(document.getElementById('groupsInput').value);highlight={{mode:'groups',day:-1,gas:gasIndex[gas],groups}};setStatus(`Highlighting ${{gas}} group(s): ${{[...groups].sort((a,b)=>a-b).join(', ')||'none'}}.`);activeButton('modeGroups');requestDraw()}});
document.getElementById('clearBtn').addEventListener('click',()=>{{highlight={{mode:'none',day:-1,gas:-1,groups:new Set()}};setStatus('No highlight selected.');activeButton('');requestDraw()}});
function requestDraw(){{if(pending)return;pending=true;requestAnimationFrame(()=>{{pending=false;draw()}})}}
function resize(){{const dpr=Math.max(1,window.devicePixelRatio||1),rect=canvas.getBoundingClientRect();canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);requestDraw()}}
function rotatePoint(x,y,z){{x-=.5;y-=.5;z-=.5;const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);const x1=x*cy-y*sy,y1=x*sy+y*cy,z1=z,y2=y1*cp-z1*sp,z2=y1*sp+z1*cp;return[x1,y2,z2]}}
function project(p,rect){{const scale=Math.min(rect.width,rect.height)*.78*zoom;return[rect.width*.48+p[0]*scale,rect.height*.56-p[1]*scale,p[2]]}}
function drawAxes(rect){{const corners=[[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],edges=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]],pts=corners.map(c=>project(rotatePoint(...c),rect));ctx.lineWidth=1;ctx.strokeStyle='#d9dee3';for(const [a,b] of edges){{ctx.beginPath();ctx.moveTo(pts[a][0],pts[a][1]);ctx.lineTo(pts[b][0],pts[b][1]);ctx.stroke()}}const axes=[[[0,0,0],[1.12,0,0],DATA.labels[0]],[[0,0,0],[0,1.12,0],DATA.labels[1]],[[0,0,0],[0,0,1.12],DATA.labels[2]]];ctx.strokeStyle='#4f565e';ctx.fillStyle='#4f565e';ctx.font='13px Arial';for(const [a,b,label] of axes){{const pa=project(rotatePoint(...a),rect),pb=project(rotatePoint(...b),rect);ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(pa[0],pa[1]);ctx.lineTo(pb[0],pb[1]);ctx.stroke();ctx.fillText(label,pb[0]+5,pb[1]-5)}}}}
function drawPlane(rect){{if(!document.getElementById('showPlane').checked)return;const cells=DATA.plane.cells.map(cell=>[cell.map(p=>project(rotatePoint(p[0],p[1],p[2]),rect)),0]);ctx.globalAlpha=.2;ctx.fillStyle='#0969da';ctx.strokeStyle='#0969da';ctx.lineWidth=.7;for(const [cell] of cells){{ctx.beginPath();ctx.moveTo(cell[0][0],cell[0][1]);for(let i=1;i<cell.length;i++)ctx.lineTo(cell[i][0],cell[i][1]);ctx.closePath();ctx.fill();ctx.stroke()}}ctx.globalAlpha=1}}
function isHighlighted(p){{if(highlight.mode==='day')return p[4]===highlight.day;if(highlight.mode==='groups')return p[3]===highlight.gas&&highlight.groups.has(p[5]);return false}}
function color(p){{if(isHighlighted(p))return groupHighlight;if(document.getElementById('showPred').checked&&p[7]>=0)return p[7]===gasIndex.acetone?predAcetone:predAlcohol;return DATA.colors[gasNames[p[3]]]}}
function draw(){{const rect=canvas.getBoundingClientRect();ctx.clearRect(0,0,rect.width,rect.height);drawAxes(rect);drawPlane(rect);ctx.globalAlpha=.9;for(const p of DATA.points){{const pp=project(rotatePoint(p[0],p[1],p[2]),rect);ctx.fillStyle=(document.getElementById('showErrors').checked&&p[8]===1)?errorColor:color(p);ctx.beginPath();ctx.arc(pp[0],pp[1],isHighlighted(p)?7:5,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1.2;ctx.stroke()}}ctx.globalAlpha=1}}
canvas.addEventListener('pointerdown',e=>{{dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId)}});
canvas.addEventListener('pointermove',e=>{{if(!dragging)return;const dx=e.clientX-lastX,dy=e.clientY-lastY;lastX=e.clientX;lastY=e.clientY;yaw+=dx*.008;pitch=Math.max(-1.25,Math.min(1.25,pitch+dy*.008));requestDraw()}});
canvas.addEventListener('pointerup',()=>{{dragging=false}});
canvas.addEventListener('wheel',e=>{{e.preventDefault();zoom=Math.max(.45,Math.min(3.2,zoom*(e.deltaY>0?.92:1.08)));requestDraw()}},{{passive:false}});
canvas.addEventListener('dblclick',()=>{{yaw=-.65;pitch=.38;zoom=1.1;requestDraw()}});
window.addEventListener('resize',resize);resize();
</script>
</body>
</html>"""


def build_space(df: pd.DataFrame, slug: str, sensors: list[str], title: str, days: list[str]) -> tuple[Path, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    columns = sensor_feature_columns(sensors)
    x = df[columns].to_numpy(float)
    coords, components, explained = pca_3d(x)
    scaled, scale = scale_no_clip(coords)
    hp = fit_lda(scaled, df["gas"].to_numpy(), df["source_file"].to_numpy())
    points = make_points(df, scaled, hp, days)
    counts = {gas: int((df["gas"] == gas).sum()) for gas in GASES}
    out_path = OUT_ROOT / "figures" / f"uci_full_features_pca_no_clip_{slug}.html"
    out_path.write_text(make_html(title, points, days, counts, hp, explained, sensors), encoding="utf-8")
    scale.insert(0, "feature_space", slug)
    scale["explained_variance_ratio"] = explained
    loading_rows = []
    for pc_idx in range(3):
        for col, loading in zip(columns, components[pc_idx]):
            loading_rows.append({
                "feature_space": slug,
                "pc": f"PC{pc_idx + 1}",
                "feature": col,
                "loading": loading,
                "abs_loading": abs(loading),
            })
    predictions = hp["predictions"].copy()
    predictions.insert(0, "feature_space", slug)
    return out_path, scale, pd.DataFrame(loading_rows), predictions, hp


def main() -> None:
    (OUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURE_TABLE)
    days = sorted(df["day"].dropna().astype(str).unique().tolist(), key=day_sort_key)
    summaries = []
    scales = []
    loadings = []
    predictions = []
    outputs = []
    for slug, sensors, title in TRIPLETS:
        out_path, scale, loading_df, pred_df, hp = build_space(df, slug, sensors, title, days)
        outputs.append(out_path)
        scales.append(scale)
        loadings.append(loading_df)
        predictions.append(pred_df)
        summaries.append({
            "feature_space": slug,
            "sensors": "/".join(sensors).upper(),
            "features_used": len(sensor_feature_columns(sensors)),
            "projection": "PCA to 3D after z-score standardization of all selected UCI-style features",
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
            "note": "Hyperplane is fitted in 3D PCA display space; final classification should use full high-dimensional features with train/test split.",
        })
    pd.DataFrame(summaries).to_csv(OUT_ROOT / "tables" / "uci_full_features_pca_space_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(scales, ignore_index=True).to_csv(OUT_ROOT / "tables" / "uci_full_features_pca_space_scales.csv", index=False, encoding="utf-8-sig")
    pd.concat(loadings, ignore_index=True).to_csv(OUT_ROOT / "tables" / "uci_full_features_pca_loadings.csv", index=False, encoding="utf-8-sig")
    pd.concat(predictions, ignore_index=True).to_csv(OUT_ROOT / "tables" / "uci_full_features_pca_space_predictions.csv", index=False, encoding="utf-8-sig")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
