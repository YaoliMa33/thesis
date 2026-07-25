"""Build a self-contained interactive dashboard for ML and DA thesis results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\thesis")
TABLES = ROOT / "tables"
DA_DIR = TABLES / "unified_da_source_target"
OUTPUT = ROOT / "figures" / "unified_results_dashboard.html"


def clean_records(frame: pd.DataFrame) -> list[dict]:
    frame = frame.replace({np.nan: None, np.inf: None, -np.inf: None})
    return frame.to_dict(orient="records")


def compact_table(frame: pd.DataFrame, columns: list[str]) -> dict:
    available = [column for column in columns if column in frame.columns]
    values = frame[available].replace({np.nan: None, np.inf: None, -np.inf: None})
    return {"columns": available, "rows": values.values.tolist()}


def require(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_ml() -> tuple[pd.DataFrame, pd.DataFrame]:
    standard = require(TABLES / "manifest_da_baseline_split_results.csv").copy()
    standard["scope"] = "Chronological S1-S7"
    standard["test_batch"] = "all later target data"
    batch = require(TABLES / "manifest_da_baseline_period_i_per_batch_results.csv").copy()
    batch["scope"] = "Period I model -> Batch1-Batch5"
    summaries = pd.concat([standard, batch], ignore_index=True, sort=False)

    standard_pred = require(TABLES / "manifest_da_baseline_split_test_predictions.csv").copy()
    standard_pred["scope"] = "Chronological S1-S7"
    batch_pred = require(TABLES / "manifest_da_baseline_period_i_per_batch_test_predictions.csv").copy()
    batch_pred["scope"] = "Period I model -> Batch1-Batch5"
    predictions = pd.concat([standard_pred, batch_pred], ignore_index=True, sort=False)
    predictions["seed"] = predictions["random_state"].fillna("deterministic").astype(str)
    return summaries, predictions


def load_da() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    summary_files = sorted(DA_DIR.glob("summary__*__uda_semi.csv"))
    prediction_files = sorted(DA_DIR.glob("predictions__*__uda_semi.csv"))
    if not summary_files or not prediction_files:
        raise FileNotFoundError("Unified DA summary/prediction files were not found.")
    summaries = pd.concat([pd.read_csv(path).assign(result_file=path.name) for path in summary_files], ignore_index=True, sort=False)
    predictions = pd.concat([pd.read_csv(path).assign(result_file=path.name) for path in prediction_files], ignore_index=True, sort=False)
    summaries = summaries.drop_duplicates(["architecture", "split", "variant"], keep="last")
    predictions = predictions.drop_duplicates(["architecture", "split", "variant", "seed", "sample_index"], keep="last")
    return summaries, predictions, [path.name for path in summary_files]


def load_curves() -> pd.DataFrame:
    specs = [
        (TABLES / "dann_interactive_training_curves.csv", "DANN family", None),
        (TABLES / "dann_conditional_training_curves.csv", "C-DANN legacy diagnostic", "Legacy two-layer"),
        (TABLES / "cdan_hybrid_training_curves.csv", "CDAN / Hybrid legacy diagnostic", "Legacy two-layer"),
        (TABLES / "deep_coral_multi_split_training_curves.csv", "Deep CORAL legacy diagnostic", "Legacy two-layer"),
    ]
    optional = TABLES / "unified_da_diagnostics_training_curves.csv"
    if optional.exists():
        specs.append((optional, "Unified selected-run diagnostics", None))
    frames = []
    for path, family, fixed_arch in specs:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["diagnostic_family"] = family
        if fixed_arch is not None:
            frame["architecture"] = fixed_arch
        elif "architecture" not in frame:
            frame["architecture"] = "Unspecified"
        frame["seed"] = frame.get("seed", pd.Series([0] * len(frame))).astype(str)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def load_features() -> pd.DataFrame:
    specs = [
        (TABLES / "dann_interactive_feature_space.csv", "DANN family", None),
        (TABLES / "dann_conditional_feature_space.csv", "C-DANN legacy diagnostic", "Legacy two-layer"),
        (TABLES / "cdan_hybrid_feature_space.csv", "CDAN / Hybrid legacy diagnostic", "Legacy two-layer"),
        (TABLES / "deep_coral_multi_split_feature_space.csv", "Deep CORAL legacy diagnostic", "Legacy two-layer"),
    ]
    optional = TABLES / "unified_da_diagnostics_feature_space.csv"
    if optional.exists():
        specs.append((optional, "Unified selected-run diagnostics", None))
    frames = []
    for path, family, fixed_arch in specs:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["diagnostic_family"] = family
        if fixed_arch is not None:
            frame["architecture"] = fixed_arch
        elif "architecture" not in frame:
            frame["architecture"] = "Unspecified"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def payload() -> dict:
    ml, ml_predictions = load_ml()
    da, da_predictions, files = load_da()
    curves, features = load_curves(), load_features()
    return {
        "ml": clean_records(ml),
        "ml_predictions_compact": compact_table(ml_predictions, [
            "scope", "split", "model", "seed", "sample_id", "target_file", "true_label",
            "predicted_label", "correct", "period", "day_label", "batch", "test_batch",
        ]),
        "da": clean_records(da),
        "da_predictions_compact": compact_table(da_predictions, [
            "architecture", "split", "variant", "seed", "sample_index", "sample_id",
            "target_file", "true_label", "predicted_label", "correct", "period", "day_label", "batch",
        ]),
        "curves_compact": compact_table(curves, [
            "diagnostic_family", "architecture", "split", "variant", "seed", "epoch",
            "L", "Ld", "Ld_cdan", "Lc", "class_loss", "domain_loss", "coral_loss",
            "total_loss", "backward_loss_scalar", "train_accuracy", "validation_accuracy",
            "test_accuracy", "domain_discriminator_accuracy",
        ]),
        "features_compact": compact_table(features, [
            "diagnostic_family", "architecture", "split", "variant", "space", "label",
            "domain", "partition", "target_file", "x", "y",
        ]),
        "da_files": files,
        "limitations": [
            "Unified ML and DA metrics/confusions use current prediction files.",
            "Training curves and PCA are shown only when the corresponding run saved diagnostic artifacts.",
            "Legacy diagnostic curves/PCA are labeled explicitly and must not be interpreted as the current unified run.",
            "MK-MMD unified runs currently have metrics and predictions but no saved latent checkpoints; their after-DA PCA is unavailable until selected-run diagnostics are generated.",
        ],
    }


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>e-Nose ML and Domain Adaptation Dashboard</title>
<style>
:root{--ink:#111827;--muted:#5b6472;--line:#d7dde5;--panel:#f7f9fb;--blue:#1769aa;--red:#b91c2b;--green:#087f5b;--purple:#6d3eb3;--orange:#c45d08}
*{box-sizing:border-box}body{margin:0;font:14px Arial,sans-serif;color:var(--ink);background:#fff}header{padding:20px 26px 14px;border-bottom:1px solid var(--line)}h1{font-size:24px;margin:0 0 5px}h2{font-size:18px;margin:0 0 12px}h3{font-size:15px;margin:0 0 9px}.muted,.note{color:var(--muted)}nav{display:flex;gap:4px;padding:10px 26px;border-bottom:1px solid var(--line);background:#fafbfc;position:sticky;top:0;z-index:4}.tab{border:1px solid transparent;background:transparent;padding:8px 13px;font-weight:700;cursor:pointer}.tab.active{border-color:var(--line);background:#fff;color:var(--blue)}main{padding:18px 26px 40px}.view{display:none}.view.active{display:block}.controls{display:grid;grid-template-columns:repeat(5,minmax(150px,1fr));gap:10px;margin-bottom:15px}label{font-size:12px;font-weight:700;color:#374151}select{display:block;width:100%;margin-top:5px;padding:7px;border:1px solid #b8c2cf;background:#fff}.band{border-top:1px solid var(--line);padding:17px 0}.metrics{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}.metric{border:1px solid var(--line);background:var(--panel);padding:10px;min-height:68px}.metric b{display:block;font-size:19px;margin-top:7px}.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}.three{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.chart{border:1px solid var(--line);background:#fff;min-height:290px;padding:8px;overflow:hidden}.chart svg{width:100%;height:270px}.matrix td,.matrix th{width:70px;height:48px;text-align:center;border:1px solid #fff}.matrix td{cursor:pointer}.matrix th{font-size:12px}.tablewrap{overflow:auto;max-height:410px;border:1px solid var(--line)}table{border-collapse:collapse;width:100%}th,td{padding:7px 9px;border-bottom:1px solid #e5e9ef;text-align:left;white-space:nowrap}th{position:sticky;top:0;background:#f5f7fa;z-index:1}.error{color:var(--red);font-weight:700}.empty{padding:40px;text-align:center;color:var(--muted)}@media(max-width:1000px){.controls{grid-template-columns:1fr 1fr}.metrics{grid-template-columns:1fr 1fr}.two,.three{grid-template-columns:1fr}}
</style></head><body>
<header><h1>e-Nose Classification and Domain Adaptation</h1><div class="muted">Three-class task: air / alcohol / acetone. Interactive metrics, confusion matrices, errors, curves and PCA.</div></header>
<nav><button class="tab active" data-view="ml">Machine Learning</button><button class="tab" data-view="da">Domain Adaptation</button><button class="tab" data-view="diag">Curves & PCA</button><button class="tab" data-view="notes">Protocol Notes</button></nav>
<main>
<section id="ml" class="view active"><div class="controls"><label>Scope<select id="mlScope"></select></label><label>Split / Batch<select id="mlSplit"></select></label><label>Model<select id="mlModel"></select></label><label>Seed<select id="mlSeed"></select></label><label>Metric<select id="mlMetric"><option value="accuracy">Accuracy</option><option value="balanced_accuracy">Balanced accuracy</option><option value="macro_f1">Macro-F1</option></select></label></div><div id="mlMetrics" class="metrics"></div><div class="band two"><div><h2>Confusion Matrix</h2><div id="mlConfusion"></div></div><div><h2>Misclassified Trials and Seeds</h2><div id="mlErrors" class="tablewrap"></div></div></div><div class="band"><h2>All Models in Selected Split</h2><div id="mlTable" class="tablewrap"></div></div></section>
<section id="da" class="view"><div class="controls"><label>Architecture<select id="daArch"></select></label><label>Split<select id="daSplit"></select></label><label>Variant<select id="daVariant"></select></label><label>Seed<select id="daSeed"></select></label><label>Metric<select id="daMetric"><option value="accuracy">Accuracy</option><option value="balanced_accuracy">Balanced accuracy</option><option value="macro_f1">Macro-F1</option></select></label></div><div id="daMetrics" class="metrics"></div><div class="band two"><div><h2>Confusion Matrix</h2><div id="daConfusion"></div></div><div><h2>Misclassified Trials and Seeds</h2><div id="daErrors" class="tablewrap"></div></div></div><div class="band"><h2>All DA Methods in Selected Split</h2><div id="daTable" class="tablewrap"></div></div></section>
<section id="diag" class="view"><div class="controls"><label>Diagnostic source<select id="dgFamily"></select></label><label>Architecture<select id="dgArch"></select></label><label>Split<select id="dgSplit"></select></label><label>Variant<select id="dgVariant"></select></label><label>Space<select id="dgSpace"></select></label></div><div class="band"><h2>Optimization Curves</h2><div id="curveNote" class="note"></div><div class="two"><div id="lossChart" class="chart"></div><div id="accChart" class="chart"></div></div></div><div class="band"><h2>Feature-space PCA</h2><div class="note">Each panel fixes one gas; blue is source and red is target. PCA is diagnostic only and does not determine model selection.</div><div id="pcaCharts" class="three"></div></div></section>
<section id="notes" class="view"><div class="band"><h2>Availability and Methodological Boundaries</h2><ul id="limitations"></ul><h3>Loaded unified DA files</h3><div id="files" class="note"></div></div></section>
</main><script>
const DATA=__PAYLOAD__; function inflate(t){return t.rows.map(row=>Object.fromEntries(t.columns.map((c,i)=>[c,row[i]])))} DATA.ml_predictions=inflate(DATA.ml_predictions_compact);DATA.da_predictions=inflate(DATA.da_predictions_compact);DATA.curves=inflate(DATA.curves_compact);DATA.features=inflate(DATA.features_compact);delete DATA.ml_predictions_compact;delete DATA.da_predictions_compact;delete DATA.curves_compact;delete DATA.features_compact; const GASES=['air','alcohol','acetone']; const $=id=>document.getElementById(id); const uniq=a=>[...new Set(a.filter(v=>v!==null&&v!==undefined).map(String))]; const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function options(el,vals,keep=true){const old=el.value;el.innerHTML=vals.map(v=>`<option>${esc(v)}</option>`).join('');if(keep&&vals.includes(old))el.value=old}
function fmt(v,d=3){return v===null||v===undefined||Number.isNaN(Number(v))?'NA':Number(v).toFixed(d)}
function metricCards(el,row,prefix='test_'){const items=[['Train acc','train_accuracy'],['Validation acc','validation_accuracy'],['Test acc',prefix+'accuracy'],['Balanced acc',prefix+'balanced_accuracy'],['Macro-F1',prefix+'macro_f1'],['Seed SD',prefix+'accuracy_std']];el.innerHTML=items.map(([a,k])=>`<div class="metric">${a}<b>${fmt(row?.[k])}</b></div>`).join('')}
function table(el,rows,cols){if(!rows.length){el.innerHTML='<div class="empty">No rows for this selection.</div>';return}el.innerHTML=`<table><thead><tr>${cols.map(c=>`<th>${esc(c[0])}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${cols.map(c=>`<td>${esc(typeof c[1]==='function'?c[1](r):r[c[1]])}</td>`).join('')}</tr>`).join('')}</tbody></table>`}
function confusion(el,rows){const counts={};GASES.forEach(t=>GASES.forEach(p=>counts[t+'|'+p]=0));rows.forEach(r=>counts[r.true_label+'|'+r.predicted_label]++);const max=Math.max(1,...Object.values(counts));el.innerHTML=`<table class="matrix"><tr><th>True \\ Pred.</th>${GASES.map(g=>`<th>${g}</th>`).join('')}</tr>${GASES.map(t=>`<tr><th>${t}</th>${GASES.map(p=>{const n=counts[t+'|'+p],a=.12+.68*n/max;return`<td style="background:rgba(23,105,170,${a})">${n}</td>`}).join('')}</tr>`).join('')}</table>`}
function errorGroups(rows){const groups={};rows.forEach(r=>{const k=r.sample_id+'|'+r.target_file+'|'+r.true_label;(groups[k]??=[]).push(r)});return Object.values(groups).map(g=>{const wrong=g.filter(r=>String(r.correct).toLowerCase()!=='true');return{sample:g[0].sample_id,file:g[0].target_file,true:g[0].true_label,wrong:wrong.length,total:g.length,seeds:wrong.map(r=>r.seed).join(','),pred:wrong.map(r=>r.seed+':'+r.predicted_label).join('; ')}}).filter(r=>r.wrong).sort((a,b)=>b.wrong-a.wrong)}
function errors(el,rows){table(el,errorGroups(rows),[['Sample','sample'],['File','file'],['True','true'],['Wrong','wrong'],['Total','total'],['Wrong seeds','seeds'],['Predictions','pred']])}
function lineChart(el,rows,keys,title){if(!rows.length){el.innerHTML='<div class="empty">No saved curve for this selection.</div>';return}const W=620,H=270,m={l:45,r:15,t:28,b:35},xs=rows.map(r=>+r.epoch),series=keys.map(([k,c,l])=>({k,c,l,v:rows.map(r=>r[k]).filter(v=>v!==null&&v!==undefined&&isFinite(v)).map(Number)})).filter(s=>s.v.length);if(!series.length){el.innerHTML='<div class="empty">Requested curve columns were not recorded.</div>';return}const xmin=Math.min(...xs),xmax=Math.max(...xs),all=series.flatMap(s=>s.v),ymin=Math.min(...all),ymax=Math.max(...all),X=x=>m.l+(x-xmin)/Math.max(1,xmax-xmin)*(W-m.l-m.r),Y=y=>H-m.b-(y-ymin)/Math.max(1e-9,ymax-ymin)*(H-m.t-m.b);const paths=series.map(s=>{const pts=rows.filter(r=>r[s.k]!=null&&isFinite(r[s.k])).map(r=>`${X(+r.epoch)},${Y(+r[s.k])}`).join(' ');return`<polyline points="${pts}" fill="none" stroke="${s.c}" stroke-width="2"/>`}).join('');el.innerHTML=`<svg viewBox="0 0 ${W} ${H}"><text x="${W/2}" y="18" text-anchor="middle" font-weight="700">${title}</text><line x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${H-m.b}" stroke="#64748b"/><line x1="${m.l}" y1="${H-m.b}" x2="${W-m.r}" y2="${H-m.b}" stroke="#64748b"/><text x="${m.l}" y="${H-8}" font-size="11">${xmin}</text><text x="${W-m.r}" y="${H-8}" text-anchor="end" font-size="11">${xmax} epoch</text><text x="5" y="${m.t+5}" font-size="11">${fmt(ymax,2)}</text><text x="5" y="${H-m.b}" font-size="11">${fmt(ymin,2)}</text>${paths}${series.map((s,i)=>`<text x="${m.l+8+i*135}" y="${m.t+14}" fill="${s.c}" font-size="11">${s.l}</text>`).join('')}</svg>`}
function scatter(el,rows,gas){const W=480,H=270,m=25;if(!rows.length){el.innerHTML='<div class="empty">No PCA points.</div>';return}const xs=rows.map(r=>+r.x),ys=rows.map(r=>+r.y),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),X=x=>m+(x-xmin)/Math.max(1e-9,xmax-xmin)*(W-2*m),Y=y=>H-m-(y-ymin)/Math.max(1e-9,ymax-ymin)*(H-2*m);el.innerHTML=`<svg viewBox="0 0 ${W} ${H}"><text x="${W/2}" y="17" text-anchor="middle" font-weight="700">${gas}</text><line x1="${m}" y1="${H-m}" x2="${W-m}" y2="${H-m}" stroke="#94a3b8"/><line x1="${m}" y1="${m}" x2="${m}" y2="${H-m}" stroke="#94a3b8"/>${rows.map(r=>`<circle cx="${X(+r.x)}" cy="${Y(+r.y)}" r="3" fill="${r.domain==='source'?'#1769aa':'#b91c2b'}" fill-opacity=".78"/>`).join('')}</svg>`}
function mlRefresh(){const scope=$('mlScope').value;const a=DATA.ml.filter(r=>r.scope===scope);options($('mlSplit'),uniq(a.map(r=>r.split)));const split=$('mlSplit').value,b=a.filter(r=>r.split===split);options($('mlModel'),uniq(b.map(r=>r.model)));const model=$('mlModel').value,row=b.find(r=>r.model===model);const pred0=DATA.ml_predictions.filter(r=>r.scope===scope&&r.split===split&&r.model===model);options($('mlSeed'),['All seeds',...uniq(pred0.map(r=>r.seed))]);const seed=$('mlSeed').value,pred=pred0.filter(r=>seed==='All seeds'||String(r.seed)===seed);metricCards($('mlMetrics'),row);confusion($('mlConfusion'),pred);errors($('mlErrors'),pred);table($('mlTable'),b,[['Model','model'],['Test acc',r=>fmt(r.test_accuracy)],['Balanced',r=>fmt(r.test_balanced_accuracy)],['Macro-F1',r=>fmt(r.test_macro_f1)],['Seed SD',r=>fmt(r.test_accuracy_std)],['Best parameters','best_params']])}
function daRefresh(){options($('daArch'),uniq(DATA.da.map(r=>r.architecture)));const arch=$('daArch').value,a=DATA.da.filter(r=>r.architecture===arch);options($('daSplit'),uniq(a.map(r=>r.split)));const split=$('daSplit').value,b=a.filter(r=>r.split===split);options($('daVariant'),uniq(b.map(r=>r.variant)));const variant=$('daVariant').value,row=b.find(r=>r.variant===variant);const pred0=DATA.da_predictions.filter(r=>r.architecture===arch&&r.split===split&&r.variant===variant);options($('daSeed'),['All seeds',...uniq(pred0.map(r=>r.seed))]);const seed=$('daSeed').value,pred=pred0.filter(r=>seed==='All seeds'||String(r.seed)===seed);metricCards($('daMetrics'),row);confusion($('daConfusion'),pred);errors($('daErrors'),pred);table($('daTable'),b.sort((x,y)=>y.test_accuracy-x.test_accuracy),[['Variant','variant'],['Test acc',r=>fmt(r.test_accuracy)],['Balanced',r=>fmt(r.test_balanced_accuracy)],['Macro-F1',r=>fmt(r.test_macro_f1)],['Seed SD',r=>fmt(r.test_accuracy_std)],['Domain BA',r=>fmt(r.domain_balanced_accuracy)],['Training','final_training']])}
function diagRefresh(){options($('dgFamily'),uniq(DATA.curves.map(r=>r.diagnostic_family).concat(DATA.features.map(r=>r.diagnostic_family))));const fam=$('dgFamily').value,allC=DATA.curves.filter(r=>r.diagnostic_family===fam),allF=DATA.features.filter(r=>r.diagnostic_family===fam),archs=uniq(allC.concat(allF).map(r=>r.architecture));options($('dgArch'),archs);const arch=$('dgArch').value,c1=allC.filter(r=>r.architecture===arch),f1=allF.filter(r=>r.architecture===arch);options($('dgSplit'),uniq(c1.concat(f1).map(r=>r.split)));const split=$('dgSplit').value,c2=c1.filter(r=>r.split===split),f2=f1.filter(r=>r.split===split);options($('dgVariant'),uniq(c2.concat(f2).map(r=>r.variant)));const variant=$('dgVariant').value,c=c2.filter(r=>r.variant===variant),f=f2.filter(r=>r.variant===variant);options($('dgSpace'),uniq(f.map(r=>r.space)));const space=$('dgSpace').value,fp=f.filter(r=>r.space===space);$('curveNote').textContent=fam.includes('legacy')?'Legacy diagnostic artifact: use for mechanism inspection, not as the current unified metric run.':'Saved diagnostic artifact for the selected run family.';lineChart($('lossChart'),c,[['L','#1769aa','L'],['class_loss','#1769aa','class loss'],['Ld','#b91c2b','Ld'],['Ld_cdan','#c45d08','Ld cdan'],['Lc','#6d3eb3','Lc'],['coral_loss','#6d3eb3','CORAL'],['total_loss','#087f5b','total']], 'Loss curves');lineChart($('accChart'),c,[['train_accuracy','#1769aa','train'],['validation_accuracy','#b91c2b','validation'],['test_accuracy','#087f5b','test'],['domain_discriminator_accuracy','#6d3eb3','domain']], 'Accuracy / domain curves');$('pcaCharts').innerHTML=GASES.map(g=>`<div id="pca_${g}" class="chart"></div>`).join('');GASES.forEach(g=>scatter($('pca_'+g),fp.filter(r=>r.label===g),g))}
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab,.view').forEach(x=>x.classList.remove('active'));b.classList.add('active');$(b.dataset.view).classList.add('active')});
['mlScope','mlSplit','mlModel','mlSeed','mlMetric'].forEach(id=>$(id).onchange=mlRefresh);['daArch','daSplit','daVariant','daSeed','daMetric'].forEach(id=>$(id).onchange=daRefresh);['dgFamily','dgArch','dgSplit','dgVariant','dgSpace'].forEach(id=>$(id).onchange=diagRefresh);
options($('mlScope'),uniq(DATA.ml.map(r=>r.scope)),false);mlRefresh();daRefresh();diagRefresh();$('limitations').innerHTML=DATA.limitations.map(x=>`<li>${esc(x)}</li>`).join('');$('files').textContent=DATA.da_files.join(' | ');const initial=location.hash.replace('#','');const initialTab=document.querySelector(`.tab[data-view="${initial}"]`);if(initialTab)initialTab.click();
</script></body></html>"""


def main() -> None:
    data = json.dumps(payload(), ensure_ascii=True, separators=(",", ":"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(HTML.replace("__PAYLOAD__", data), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
