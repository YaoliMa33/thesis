"""Build an interactive report for the train+validation refit comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\thesis")
INPUT_PATH = ROOT / "tables" / "dann_train_val_refit" / "dann_train_val_refit_vs_sklearn_mlp.csv"
PREDICTION_PATH = ROOT / "tables" / "dann_train_val_refit" / "dann_train_val_refit_test_predictions.csv"
OUTPUT_PATH = ROOT / "figures" / "dann_learning" / "dann_train_val_refit_comparison.html"


def records(frame: pd.DataFrame) -> list[dict]:
    output = []
    for row in frame.to_dict("records"):
        output.append(
            {
                key: None
                if isinstance(value, (float, np.floating)) and np.isnan(value)
                else value.item()
                if isinstance(value, np.generic)
                else value
                for key, value in row.items()
            }
        )
    return output


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing refit comparison table: {INPUT_PATH}. Run run_dann_train_val_refit_comparison.py first."
        )
    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            f"Missing per-seed predictions: {PREDICTION_PATH}. Run run_dann_train_val_refit_comparison.py first."
        )
    frame = pd.read_csv(INPUT_PATH)
    data = json.dumps(records(frame), ensure_ascii=False)
    predictions = json.dumps(records(pd.read_csv(PREDICTION_PATH)), ensure_ascii=False)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DANN Train + Validation Refit Comparison</title>
<style>
:root{{--bg:#f5f7f9;--panel:#fff;--ink:#18222c;--muted:#63717e;--line:#d4dde5;--blue:#2368a2;--green:#18734b;--red:#a63d31}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:Arial,sans-serif;letter-spacing:0}}
header{{padding:24px max(20px,calc((100vw - 1200px)/2));background:#fff;border-bottom:1px solid var(--line)}}
h1{{margin:0 0 6px;font-size:25px}} .sub,.note{{color:var(--muted);font-size:13px;line-height:1.5}}
main{{max-width:1200px;margin:auto;padding:20px}} .controls{{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:12px;margin-bottom:18px}}
label{{display:block;font-size:12px;font-weight:700;margin-bottom:5px}} select{{width:100%;padding:9px;border:1px solid #aebbc7;background:#fff}}
section{{background:var(--panel);border:1px solid var(--line);padding:16px;margin-bottom:18px}} h2{{font-size:18px;margin:0 0 12px}}
.metrics{{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:10px}} .metric{{border-left:3px solid var(--blue);padding:10px;background:#f8fafc}}
.k{{font-size:11px;color:var(--muted)}} .v{{font-size:19px;font-weight:700;margin-top:4px}} table{{width:100%;border-collapse:collapse;font-size:12px}}
th,td{{padding:9px;border-bottom:1px solid var(--line);text-align:left}} th{{background:#edf2f6}} .positive{{color:var(--green)}} .negative{{color:var(--red)}}
.confusion th,.confusion td{{text-align:center;min-width:105px}} .confusion td{{font-size:18px;font-weight:700;cursor:pointer}} .confusion td:hover{{background:#e8f1f8}}
.unanimous{{color:var(--red);font-weight:700}} .seed-list{{font-family:Consolas,monospace;font-size:11px}}
@media(max-width:760px){{.controls,.metrics{{grid-template-columns:1fr}} main{{padding:10px}}}}
</style></head><body>
<header><h1>DANN Train + Validation Refit Comparison</h1><div class="sub">Validation selects hyperparameters and epoch budget, then final models are initialized from scratch and trained on train + validation. Test labels remain evaluation-only.</div></header>
<main><div class="controls">
<div><label>Architecture</label><select id="architecture"></select></div>
<div><label>Split</label><select id="split"></select></div>
<div><label>Final model</label><select id="variant"></select></div>
<div><label>Confusion/error seed</label><select id="seed"></select></div>
</div>
<section><h2>Selected Result</h2><div id="metrics" class="metrics"></div><p id="protocol" class="note"></p></section>
<section><h2>Current Split Comparison</h2><div style="overflow-x:auto"><table><thead><tr><th>Architecture</th><th>Model</th><th>Final N</th><th>Selected lambda</th><th>Epoch</th><th>Test accuracy</th><th>Balanced accuracy</th><th>Macro-F1</th><th>vs sklearn MLP</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<section><h2>Confusion Matrix</h2><p id="confusionNote" class="note"></p><div id="confusion" style="overflow-x:auto"></div></section>
<section><h2>Misclassified Trials and Seeds</h2><p class="note">Click a confusion-matrix cell to filter this table. Select the same cell again by changing a selector to clear the filter.</p><div style="overflow-x:auto"><table><thead><tr><th>Sample</th><th>File</th><th>Period/day</th><th>Batch</th><th>True gas</th><th>Predicted gas by wrong seeds</th><th>Wrong seeds</th><th>Wrong count</th><th>All seeds wrong</th></tr></thead><tbody id="errors"></tbody></table></div></section>
<section><h2>Interpretation Boundary</h2><p class="note">For S4-S7, target validation labels are consumed by final refit, so these final models are semi-supervised and must not be called UDA. Validation accuracy is not reported after refit because validation is no longer independent. Test labels are never used for training or selection.</p></section>
</main><script>
const DATA={data}; const PREDICTIONS={predictions}; const GASES=["air","alcohol","acetone"]; const $=id=>document.getElementById(id); const uniq=x=>[...new Set(x)]; let confusionFilter=null;
const fmt=(x,d=3)=>x===null||x===undefined||Number.isNaN(Number(x))?'NA':Number(x).toFixed(d);
function options(el,values){{const old=el.value;el.innerHTML=values.map(v=>`<option>${{v}}</option>`).join('');if(values.includes(old))el.value=old}}
function refreshVariants(){{const a=$('architecture').value,s=$('split').value;options($('variant'),uniq(DATA.filter(r=>r.architecture===a&&r.split===s).map(r=>r.final_variant)))}}
function selectedPredictions(){{const a=$('architecture').value,s=$('split').value,v=$('variant').value,seed=$('seed').value;return PREDICTIONS.filter(r=>r.architecture===a&&r.split===s&&r.final_variant===v&&(seed==='All seeds'||String(r.seed)===seed))}}
function renderConfusion(){{const rows=selectedPredictions(),seed=$('seed').value;const counts={{}};GASES.forEach(t=>GASES.forEach(p=>counts[`${{t}}||${{p}}`]=0));rows.forEach(r=>counts[`${{r.true_label}}||${{r.predicted_label}}`]++);
const head=`<table class="confusion"><thead><tr><th>True \\ Predicted</th>${{GASES.map(g=>`<th>${{g}}</th>`).join('')}}</tr></thead><tbody>`;
const body=GASES.map(t=>`<tr><th>${{t}}</th>${{GASES.map(p=>`<td data-true="${{t}}" data-pred="${{p}}">${{counts[`${{t}}||${{p}}`]}}</td>`).join('')}}</tr>`).join('');$('confusion').innerHTML=head+body+'</tbody></table>';
$('confusionNote').textContent=seed==='All seeds'?`Counts sum predictions over all ${{uniq(rows.map(r=>r.seed)).length}} seeds; total predictions=${{rows.length}}.`:`Counts for seed ${{seed}}; total trials=${{rows.length}}.`;
$('confusion').querySelectorAll('td').forEach(cell=>cell.onclick=()=>{{confusionFilter={{trueLabel:cell.dataset.true,predictedLabel:cell.dataset.pred}};renderErrors()}})}}
function renderErrors(){{const rows=selectedPredictions(),seedChoice=$('seed').value;const grouped=new Map();rows.forEach(r=>{{if(!grouped.has(r.sample_id))grouped.set(r.sample_id,[]);grouped.get(r.sample_id).push(r)}});let errors=[];
grouped.forEach(group=>{{const wrong=group.filter(r=>!r.correct);if(!wrong.length)return;const first=group[0];if(confusionFilter&&!wrong.some(r=>r.true_label===confusionFilter.trueLabel&&r.predicted_label===confusionFilter.predictedLabel))return;const wrongSeeds=wrong.map(r=>r.seed).sort((a,b)=>a-b);const predictionsBySeed=wrong.map(r=>`${{r.seed}}:${{r.predicted_label}}`).join('; ');const allSeedsSelected=$('seed').value==='All seeds';errors.push({{first,wrongSeeds,predictionsBySeed,wrongCount:wrong.length,total:group.length,unanimous:allSeedsSelected&&wrong.length===group.length,allSeedsSelected}})}});
errors.sort((a,b)=>Number(b.unanimous)-Number(a.unanimous)||b.wrongCount-a.wrongCount||String(a.first.sample_id).localeCompare(String(b.first.sample_id)));
$('errors').innerHTML=errors.map(e=>`<tr><td>${{e.first.sample_id}}</td><td>${{e.first.target_file}}</td><td>${{e.first.period}} / ${{e.first.day_label}}</td><td>${{e.first.batch}}</td><td>${{e.first.true_label}}</td><td class="seed-list">${{e.predictionsBySeed}}</td><td class="seed-list">${{e.wrongSeeds.join(',')}}</td><td>${{e.wrongCount}}/${{e.total}}</td><td class="${{e.unanimous?'unanimous':''}}">${{e.allSeedsSelected?(e.unanimous?'YES':'No'):'N/A'}}</td></tr>`).join('')||'<tr><td colspan="9">No misclassified trials for this selection.</td></tr>'}}
function render(){{const a=$('architecture').value,s=$('split').value,v=$('variant').value;const r=DATA.find(x=>x.architecture===a&&x.split===s&&x.final_variant===v);if(!r)return;
const cards=[['Train accuracy',r.train_accuracy],['Test accuracy',r.test_accuracy],['Test balanced accuracy',r.test_balanced_accuracy],['Test macro-F1',r.test_macro_f1]];
$('metrics').innerHTML=cards.map(x=>`<div class="metric"><div class="k">${{x[0]}}</div><div class="v">${{fmt(x[1])}}</div></div>`).join('');
$('protocol').textContent=`selection variant: ${{r.selection_variant}}; validation consumed: ${{r.validation_n_consumed}}; final train N: ${{r.final_train_n}}; seeds: ${{r.seed_values}}; target labels in final refit: ${{r.target_labels_used_in_final_refit}}`;
const rows=DATA.filter(x=>x.split===s);$('rows').innerHTML=rows.map(x=>{{const delta=Number(x.test_accuracy_minus_sklearn_mlp);return `<tr><td>${{x.architecture}}</td><td>${{x.final_variant}}</td><td>${{x.final_train_n}}</td><td>${{fmt(x.selected_lambda,2)}}</td><td>${{fmt(x.selected_epoch,1)}}</td><td>${{fmt(x.test_accuracy)}} +/- ${{fmt(x.test_accuracy_std)}}</td><td>${{fmt(x.test_balanced_accuracy)}}</td><td>${{fmt(x.test_macro_f1)}}</td><td class="${{delta>=0?'positive':'negative'}}">${{delta>=0?'+':''}}${{fmt(delta)}}</td></tr>`}}).join('');confusionFilter=null;renderConfusion();renderErrors()}}
options($('architecture'),uniq(DATA.map(r=>r.architecture)));options($('split'),uniq(DATA.map(r=>r.split)));options($('seed'),['All seeds',...uniq(PREDICTIONS.map(r=>String(r.seed))).sort()]);refreshVariants();
$('architecture').onchange=()=>{{refreshVariants();render()}};$('split').onchange=()=>{{refreshVariants();render()}};$('variant').onchange=render;$('seed').onchange=()=>{{confusionFilter=null;renderConfusion();renderErrors()}};render();
</script></body></html>"""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(document, encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
