from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SENSORS = [f"s{i}" for i in range(1, 7)]
CLASSES = ["air", "alcohol", "acetone"]
COLORS = {"air": "#111827", "alcohol": "#2563eb", "acetone": "#dc2626"}


def locate_output_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if parent.name.lower() == "thesis":
            return parent
        if (parent / "thesis_out").exists():
            return parent / "thesis_out"
    return path.parents[2] / "thesis_out"


def locate_data_root(output_root: Path) -> Path:
    candidates = [
        output_root / "enose_data",
        output_root.parent / "enose_data",
        Path(r"D:\datasets\enose_data"),
        Path(r"D:\datasets\enose_export\enose_data"),
    ]
    for candidate in candidates:
        if all((candidate / label).exists() for label in CLASSES):
            return candidate
    raise FileNotFoundError("Could not locate enose_data with air/alcohol/acetone folders.")


OUT_ROOT = locate_output_root()
DATA_ROOT = locate_data_root(OUT_ROOT)
FIG_DIR = OUT_ROOT / "figures"


def relative_time_seconds(df: pd.DataFrame) -> np.ndarray:
    t = df["arduino_time"].to_numpy(dtype=float)
    return (t - t[0]) / 1000.0


def baseline_mask(t: np.ndarray, label: str) -> np.ndarray:
    # This is the only phase-aware operation: baseline correction.
    if label == "air":
        end = min(60.0, max(20.0, float(np.nanmax(t)) * 0.35))
    else:
        end = 60.0
    mask = t <= end
    if mask.sum() < 3:
        mask = np.arange(len(t)) < max(3, int(len(t) * 0.2))
    return mask


def corrected_sample(path: Path) -> tuple[str, dict[str, np.ndarray]]:
    df = pd.read_csv(path)
    label = str(df["gas_type"].iloc[0]).strip().lower()
    t = relative_time_seconds(df)
    bmask = baseline_mask(t, label)
    corrected = {}
    for sensor in SENSORS:
        x = df[sensor].to_numpy(dtype=float)
        b0 = float(np.nanmedian(x[bmask]))
        corrected[sensor] = (x - b0) / max(abs(b0), 1e-9)
    return label, corrected


def collect_points(axes: tuple[str, str, str]) -> dict:
    grouped = {label: {"x": [], "y": [], "z": []} for label in CLASSES}
    files_by_label = {}
    for label in CLASSES:
        paths = sorted((DATA_ROOT / label).glob("*.csv"))
        files_by_label[label] = len(paths)
        for path in paths:
            sample_label, corrected = corrected_sample(path)
            x = corrected[axes[0]]
            y = corrected[axes[1]]
            z = corrected[axes[2]]
            n = min(len(x), len(y), len(z))
            xyz = np.column_stack([x[:n], y[:n], z[:n]])
            finite = np.isfinite(xyz).all(axis=1)
            xyz = xyz[finite]
            grouped[sample_label]["x"].extend(np.round(xyz[:, 0], 6).tolist())
            grouped[sample_label]["y"].extend(np.round(xyz[:, 1], 6).tolist())
            grouped[sample_label]["z"].extend(np.round(xyz[:, 2], 6).tolist())
    counts = {label: len(grouped[label]["x"]) for label in CLASSES}
    return {
        "axes": [a.upper() for a in axes],
        "classes": CLASSES,
        "colors": COLORS,
        "points": grouped,
        "pointCounts": counts,
        "fileCounts": files_by_label,
        "processing": "baseline correction only; no feature extraction; no resampling; original CSV sampling points",
    }


def html_template(payload: dict, title: str) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172033;
      --muted: #5b667a;
      --line: #d7dee8;
      --panel: #f8fafc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background: #ffffff;
      overflow: hidden;
    }}
    header {{
      position: fixed;
      inset: 0 0 auto 0;
      height: 76px;
      padding: 14px 22px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,0.94);
      z-index: 5;
    }}
    h1 {{
      margin: 0 0 5px;
      font-size: 20px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    #stage {{
      position: fixed;
      inset: 76px 0 0 0;
      width: 100vw;
      height: calc(100vh - 76px);
      display: block;
      cursor: grab;
      background: linear-gradient(#ffffff, #f9fafb);
    }}
    #stage:active {{ cursor: grabbing; }}
    .panel {{
      position: fixed;
      right: 16px;
      top: 92px;
      width: 270px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,0.92);
      box-shadow: 0 12px 28px rgba(15,23,42,0.08);
      z-index: 6;
      font-size: 13px;
    }}
    .legend {{ display: grid; gap: 7px; margin-top: 8px; }}
    .legend-row {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; }}
    .label {{ display: flex; align-items: center; gap: 7px; }}
    .dot {{ width: 10px; height: 10px; border-radius: 999px; display: inline-block; }}
    .controls {{
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      line-height: 1.45;
    }}
    button {{
      margin-top: 10px;
      width: 100%;
      height: 32px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--ink);
      font: inherit;
      cursor: pointer;
    }}
    button:hover {{ background: #eef2f7; }}
    #status {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="meta">Baseline corrected original sampling points only. No feature extraction, no resampling.</div>
  </header>
  <canvas id="stage"></canvas>
  <aside class="panel">
    <strong>Axes</strong>
    <div id="axes"></div>
    <div class="legend" id="legend"></div>
    <div class="controls">
      Drag: rotate 360°<br>
      Wheel: zoom<br>
      Shift + drag: pan<br>
      Double click: reset view
    </div>
    <button id="reset">Reset view</button>
    <div id="status">Loading point cloud...</div>
  </aside>
  <script>
    window.addEventListener('error', e => {{
      const box = document.getElementById('status');
      if (box) box.textContent = 'Render error: ' + e.message;
    }});
    const payload = {payload_json};
    const canvas = document.getElementById('stage');
    const ctx = canvas.getContext('2d', {{ alpha: false }});
    const axesBox = document.getElementById('axes');
    const legend = document.getElementById('legend');
    const statusBox = document.getElementById('status');
    axesBox.textContent = payload.axes.join(' / ');

    const allPoints = [];
    for (const label of payload.classes) {{
      const p = payload.points[label];
      const color = payload.colors[label];
      for (let i = 0; i < p.x.length; i++) {{
        allPoints.push({{x:p.x[i], y:p.y[i], z:p.z[i], label, color}});
      }}
      const row = document.createElement('div');
      row.className = 'legend-row';
      row.innerHTML = `<span class="label"><span class="dot" style="background:${{color}}"></span>${{label}}</span><span>${{p.x.length.toLocaleString()}} pts</span>`;
      legend.appendChild(row);
    }}

    const bounds = allPoints.reduce((b, p) => {{
      b.minX = Math.min(b.minX, p.x); b.maxX = Math.max(b.maxX, p.x);
      b.minY = Math.min(b.minY, p.y); b.maxY = Math.max(b.maxY, p.y);
      b.minZ = Math.min(b.minZ, p.z); b.maxZ = Math.max(b.maxZ, p.z);
      return b;
    }}, {{minX:Infinity,maxX:-Infinity,minY:Infinity,maxY:-Infinity,minZ:Infinity,maxZ:-Infinity}});

    const center = {{
      x: (bounds.minX + bounds.maxX) / 2,
      y: (bounds.minY + bounds.maxY) / 2,
      z: (bounds.minZ + bounds.maxZ) / 2
    }};
    const span = Math.max(bounds.maxX-bounds.minX, bounds.maxY-bounds.minY, bounds.maxZ-bounds.minZ, 1e-6);

    let state = {{ rx: -0.55, ry: 0.72, zoom: 1.0, panX: 0, panY: 0 }};
    let dragging = false;
    let lastX = 0, lastY = 0;
    let moved = false;

    function resize() {{
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }}

    function rotatePoint(p) {{
      let x = (p.x - center.x) / span;
      let y = (p.y - center.y) / span;
      let z = (p.z - center.z) / span;
      const cx = Math.cos(state.rx), sx = Math.sin(state.rx);
      const cy = Math.cos(state.ry), sy = Math.sin(state.ry);
      const y1 = y * cx - z * sx;
      const z1 = y * sx + z * cx;
      const x2 = x * cy + z1 * sy;
      const z2 = -x * sy + z1 * cy;
      return {{x:x2, y:y1, z:z2}};
    }}

    function project(r, w, h) {{
      const scale = Math.min(w, h) * 0.82 * state.zoom;
      const perspective = 1.9 / (2.35 - r.z);
      return {{
        x: w/2 + state.panX + r.x * scale * perspective,
        y: h/2 + state.panY - r.y * scale * perspective,
        z: r.z,
        s: Math.max(1.1, 2.15 * perspective)
      }};
    }}

    function drawAxes(w, h) {{
      const axisLen = 0.62;
      const origin = project({{x:0,y:0,z:0}}, w, h);
      const axes = [
        {{name: payload.axes[0], p: {{x:axisLen,y:0,z:0}}, color:'#334155'}},
        {{name: payload.axes[1], p: {{x:0,y:axisLen,z:0}}, color:'#334155'}},
        {{name: payload.axes[2], p: {{x:0,y:0,z:axisLen}}, color:'#334155'}},
      ];
      ctx.save();
      ctx.lineWidth = 1.2;
      ctx.font = '13px Arial';
      ctx.fillStyle = '#334155';
      for (const a of axes) {{
        const p = project(rotatePoint({{x:center.x + a.p.x*span, y:center.y + a.p.y*span, z:center.z + a.p.z*span}}), w, h);
        ctx.strokeStyle = a.color;
        ctx.beginPath();
        ctx.moveTo(origin.x, origin.y);
        ctx.lineTo(p.x, p.y);
        ctx.stroke();
        ctx.fillText(a.name, p.x + 6, p.y - 6);
      }}
      ctx.restore();
    }}

    function draw() {{
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = '#f8fafc';
      ctx.fillRect(0, 0, w, h);
      drawAxes(w, h);

      const projected = new Array(allPoints.length);
      for (let i = 0; i < allPoints.length; i++) {{
        const r = rotatePoint(allPoints[i]);
        projected[i] = {{...project(r, w, h), color: allPoints[i].color}};
      }}
      projected.sort((a,b) => a.z - b.z);

      ctx.globalAlpha = 0.54;
      for (const p of projected) {{
        if (p.x < -10 || p.x > w + 10 || p.y < -10 || p.y > h + 10) continue;
        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.s, 0, Math.PI * 2);
        ctx.fill();
      }}
      ctx.globalAlpha = 1;
      statusBox.textContent = `Rendered ${{allPoints.length.toLocaleString()}} finite original points.`;
    }}

    canvas.addEventListener('pointerdown', e => {{
      dragging = true; moved = false; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId);
    }});
    canvas.addEventListener('pointermove', e => {{
      if (!dragging) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      if (Math.abs(dx) + Math.abs(dy) > 1) moved = true;
      lastX = e.clientX; lastY = e.clientY;
      if (e.shiftKey) {{
        state.panX += dx;
        state.panY += dy;
      }} else {{
        state.ry += dx * 0.009;
        state.rx += dy * 0.009;
      }}
      draw();
    }});
    canvas.addEventListener('pointerup', e => {{ dragging = false; }});
    canvas.addEventListener('wheel', e => {{
      e.preventDefault();
      const factor = Math.exp(-e.deltaY * 0.001);
      state.zoom = Math.min(8, Math.max(0.2, state.zoom * factor));
      draw();
    }}, {{ passive: false }});
    canvas.addEventListener('dblclick', () => reset());
    document.getElementById('reset').addEventListener('click', reset);

    function reset() {{
      state = {{ rx: -0.55, ry: 0.72, zoom: 1.0, panX: 0, panY: 0 }};
      draw();
    }}

    window.addEventListener('resize', resize);
    requestAnimationFrame(() => {{
      resize();
      setTimeout(resize, 80);
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        (("s1", "s2", "s3"), "feature_space_original_points_bc_s1_s2_s3.html", "S1 / S2 / S3 Feature Space"),
        (("s2", "s3", "s5"), "feature_space_original_points_bc_s2_s3_s5.html", "S2 / S3 / S5 Feature Space"),
    ]
    for axes, filename, title in outputs:
        payload = collect_points(axes)
        out = FIG_DIR / filename
        out.write_text(html_template(payload, title), encoding="utf-8")
        print(out)
        print(payload["pointCounts"])


if __name__ == "__main__":
    main()
