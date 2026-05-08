"""
llmbench/api/reporter.py
Self-contained HTML report generator with:
  - Metric score bar chart
  - Latency distribution histogram
  - Confidence reliability diagram (when confidence scores exist)   ← NEW
  - Per-sample table with confidence column                         ← NEW

Usage:
    from llmbench.api.reporter import generate_html_report
    html = generate_html_report(run_dict, sample_dicts)
    Path("report.html").write_text(html)

CLI:
    python -m llmbench.api.reporter <run_id> [output.html]
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

from jinja2 import Template

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>LLMBench — {{ run.run_id[:8] }}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#0f1117;--surface:#1a1d27;--border:#2a2d3e;--accent:#7c6af7;
      --accent2:#06b6d4;--green:#22c55e;--red:#ef4444;--amber:#f59e0b;
      --text:#e2e8f0;--muted:#64748b;--r:10px;}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);
     line-height:1.6;padding:2rem;}
.header{display:flex;justify-content:space-between;align-items:flex-start;
        margin-bottom:2rem;padding-bottom:1.5rem;border-bottom:1px solid var(--border);}
.header h1{font-size:1.6rem;font-weight:700;}
.header h1 span{color:var(--accent);}
.badge{display:inline-block;padding:.2rem .7rem;border-radius:999px;font-size:.75rem;
       font-weight:600;background:var(--surface);border:1px solid var(--border);color:var(--muted);}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
            gap:1rem;margin-bottom:2rem;}
.stat-card{background:var(--surface);border:1px solid var(--border);
           border-radius:var(--r);padding:1.2rem 1.4rem;}
.stat-card .label{font-size:.75rem;color:var(--muted);text-transform:uppercase;
                  letter-spacing:.05em;margin-bottom:.4rem;}
.stat-card .value{font-size:1.6rem;font-weight:700;}
.stat-card .value.green{color:var(--green);}
.stat-card .value.amber{color:var(--amber);}
.stat-card .value.red{color:var(--red);}
.section{margin-bottom:2.5rem;}
.section h2{font-size:1rem;font-weight:600;color:var(--muted);text-transform:uppercase;
            letter-spacing:.08em;margin-bottom:1rem;}
.score-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;}
.score-card{background:var(--surface);border:1px solid var(--border);
            border-radius:var(--r);padding:1rem 1.2rem;}
.score-card .metric-name{font-size:.8rem;color:var(--muted);margin-bottom:.5rem;}
.score-card .metric-value{font-size:1.4rem;font-weight:700;color:var(--accent);}
.score-card .bar{height:4px;background:var(--border);border-radius:2px;
                 margin-top:.6rem;overflow:hidden;}
.score-card .bar-fill{height:100%;border-radius:2px;
                      background:linear-gradient(90deg,var(--accent),var(--accent2));}
.chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
            gap:1.5rem;margin-bottom:2rem;}
.chart-box{background:var(--surface);border:1px solid var(--border);
           border-radius:var(--r);padding:1.2rem;}
.chart-box h3{font-size:.85rem;color:var(--muted);margin-bottom:1rem;}
.table-wrap{overflow-x:auto;}
table{width:100%;border-collapse:collapse;font-size:.85rem;}
thead tr{border-bottom:2px solid var(--border);}
th{padding:.6rem .8rem;text-align:left;color:var(--muted);font-weight:600;
   font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;}
td{padding:.6rem .8rem;border-bottom:1px solid var(--border);
   vertical-align:top;max-width:300px;word-break:break-word;}
tr:last-child td{border-bottom:none;}
tr:hover td{background:rgba(124,106,247,.04);}
.tag{display:inline-block;padding:.1rem .5rem;border-radius:4px;font-size:.7rem;font-weight:600;}
.tag.ok{background:rgba(34,197,94,.15);color:var(--green);}
.tag.err{background:rgba(239,68,68,.15);color:var(--red);}
.conf-bar{display:inline-block;width:50px;height:6px;background:var(--border);
          border-radius:3px;vertical-align:middle;overflow:hidden;}
.conf-fill{height:100%;background:var(--accent2);border-radius:3px;}
.config-box{background:var(--surface);border:1px solid var(--border);
            border-radius:var(--r);padding:1.2rem 1.4rem;font-size:.85rem;}
.config-row{display:flex;gap:1rem;margin-bottom:.4rem;}
.config-row .key{color:var(--muted);min-width:160px;}
.config-row .val{font-family:monospace;}
footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--border);
       font-size:.75rem;color:var(--muted);text-align:center;}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>🔬 LLMBench — <span>Run Report</span></h1>
    <p style="color:var(--muted);font-size:.85rem;margin-top:.4rem;">
      {{ run.run_id }} &nbsp;·&nbsp;
      {{ run.created_at[:16].replace('T',' ') if run.created_at else '' }} UTC
    </p>
  </div>
  <div style="text-align:right;">
    <span class="badge">{{ run.model_slug }}</span>&nbsp;
    <span class="badge">{{ run.task_type }}</span>&nbsp;
    <span class="badge">{{ run.dataset_name }}</span>
  </div>
</div>

<!-- stat cards -->
<div class="stats-grid">
  <div class="stat-card">
    <div class="label">Samples</div>
    <div class="value">{{ run.total_samples }}</div>
  </div>
  <div class="stat-card">
    <div class="label">Failed</div>
    <div class="value {{ 'red' if run.failed_samples > 0 else 'green' }}">
      {{ run.failed_samples }}</div>
  </div>
  <div class="stat-card">
    <div class="label">Tokens</div>
    <div class="value">{{ "{:,}".format(run.total_tokens) }}</div>
  </div>
  <div class="stat-card">
    <div class="label">Avg latency</div>
    <div class="value amber">{{ "%.0f"|format(avg_latency_ms) }}ms</div>
  </div>
  {% if has_confidence %}
  <div class="stat-card">
    <div class="label">Avg confidence</div>
    <div class="value" style="color:var(--accent2);">{{ "%.3f"|format(avg_confidence) }}</div>
  </div>
  <div class="stat-card">
    <div class="label">ECE</div>
    <div class="value {{ 'green' if ece_val < 0.1 else 'amber' if ece_val < 0.2 else 'red' }}">
      {{ "%.4f"|format(ece_val) if ece_val == ece_val else "N/A" }}</div>
  </div>
  {% endif %}
  {% if primary_score is not none %}
  <div class="stat-card">
    <div class="label">{{ primary_metric }}</div>
    <div class="value" style="color:var(--accent);">{{ "%.4f"|format(primary_score) }}</div>
  </div>
  {% endif %}
</div>

<!-- metric scores -->
<div class="section">
  <h2>Aggregate Scores</h2>
  <div class="score-grid">
    {% for metric, score in display_scores.items() %}
    <div class="score-card">
      <div class="metric-name">{{ metric }}</div>
      <div class="metric-value">{{ "%.4f"|format(score) }}</div>
      <div class="bar"><div class="bar-fill" style="width:{{ [score*100,100]|min }}%;"></div></div>
    </div>
    {% endfor %}
  </div>
</div>

<!-- charts -->
<div class="section">
  <h2>Visualizations</h2>
  <div class="chart-grid">
    <div class="chart-box">
      <h3>Metric Scores</h3>
      <canvas id="scoreBar"></canvas>
    </div>
    <div class="chart-box">
      <h3>Latency Distribution (ms)</h3>
      <canvas id="latHist"></canvas>
    </div>
    {% if has_confidence %}
    <div class="chart-box">
      <h3>Reliability Diagram — Confidence vs Accuracy</h3>
      <canvas id="reliabilityChart"></canvas>
    </div>
    <div class="chart-box">
      <h3>Confidence Distribution</h3>
      <canvas id="confHist"></canvas>
    </div>
    {% endif %}
  </div>
</div>

<!-- sample table -->
{% if samples %}
<div class="section">
  <h2>Sample Breakdown ({{ samples|length }} shown)</h2>
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>#</th><th>Generated</th><th>Expected</th>
        {% for m in score_metrics %}<th>{{ m }}</th>{% endfor %}
        <th>Conf</th><th>ms</th><th>Status</th>
      </tr></thead>
      <tbody>
        {% for s in samples %}
        <tr>
          <td style="color:var(--muted);font-size:.75rem;">{{ loop.index }}</td>
          <td>{{ (s.generated_output or '')[:120] }}{% if (s.generated_output or '')|length > 120 %}…{% endif %}</td>
          <td style="color:var(--muted);">{{ (s.expected_output or '—')[:80] }}</td>
          {% set sc = s.scores if s.scores is mapping else {} %}
          {% for m in score_metrics %}
            <td>{{ "%.3f"|format(sc[m]) if m in sc else "—" }}</td>
          {% endfor %}
          <td>
            {% if s.confidence is not none and s.confidence == s.confidence %}
              <span class="conf-bar"><span class="conf-fill"
                style="width:{{ (s.confidence * 100)|int }}%;"></span></span>
              {{ "%.2f"|format(s.confidence) }}
            {% else %}—{% endif %}
          </td>
          <td style="color:var(--muted);">{{ "%.0f"|format(s.latency_ms or 0) }}</td>
          <td>{% if s.error %}<span class="tag err">ERR</span>
              {% else %}<span class="tag ok">OK</span>{% endif %}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>
{% endif %}

<!-- config -->
<div class="section">
  <h2>Run Configuration</h2>
  <div class="config-box">
    <div class="config-row"><span class="key">Run ID</span><span class="val">{{ run.run_id }}</span></div>
    <div class="config-row"><span class="key">Model</span><span class="val">{{ run.model_slug }}</span></div>
    <div class="config-row"><span class="key">Dataset</span><span class="val">{{ run.dataset_name }}</span></div>
    <div class="config-row"><span class="key">Task</span><span class="val">{{ run.task_type }}</span></div>
    <div class="config-row"><span class="key">Total samples</span><span class="val">{{ run.total_samples }}</span></div>
    <div class="config-row"><span class="key">Total tokens</span><span class="val">{{ "{:,}".format(run.total_tokens) }}</span></div>
    {% if run.tags %}<div class="config-row"><span class="key">Tags</span>
    <span class="val">{{ run.tags }}</span></div>{% endif %}
  </div>
</div>

<footer>Generated by <strong>LLMBench v0.2.0</strong> · {{ generated_at }} UTC</footer>

<script>
const A='#7c6af7',A2='#06b6d4',BORDER='#2a2d3e',MUTED='#64748b';
Chart.defaults.color=MUTED; Chart.defaults.borderColor=BORDER;

// Score bar
const sd={{ chart_data.scores | tojson }};
new Chart(document.getElementById('scoreBar'),{type:'bar',
  data:{labels:sd.labels,datasets:[{label:'Score',data:sd.values,
    backgroundColor:sd.labels.map((_,i)=>i%2===0?A:A2),borderRadius:5}]},
  options:{responsive:true,plugins:{legend:{display:false}},
    scales:{y:{min:0,max:1,grid:{color:BORDER}},x:{grid:{display:false}}}}});

// Latency histogram
const lb={{ chart_data.lat_bins | tojson }}, lc={{ chart_data.lat_counts | tojson }};
new Chart(document.getElementById('latHist'),{type:'bar',
  data:{labels:lb.map(b=>b+'ms'),datasets:[{label:'Samples',data:lc,
    backgroundColor:A2,borderRadius:3}]},
  options:{responsive:true,plugins:{legend:{display:false}},
    scales:{y:{grid:{color:BORDER}},x:{grid:{display:false},ticks:{maxRotation:30}}}}});

{% if has_confidence %}
// Reliability diagram
const rd={{ chart_data.reliability | tojson }};
new Chart(document.getElementById('reliabilityChart'),{type:'scatter',
  data:{datasets:[
    {label:'Perfect calibration',data:[{x:0,y:0},{x:1,y:1}],type:'line',
     borderColor:MUTED,borderDash:[6,3],borderWidth:1,pointRadius:0,fill:false},
    {label:'Model',data:rd.points,backgroundColor:A,borderColor:A,
     pointRadius:rd.sizes.map(s=>Math.max(5,s/3)),
     pointHoverRadius:10},
  ]},
  options:{responsive:true,
    scales:{x:{title:{display:true,text:'Mean confidence'},min:0,max:1,grid:{color:BORDER}},
            y:{title:{display:true,text:'Fraction correct'},min:0,max:1,grid:{color:BORDER}}},
    plugins:{tooltip:{callbacks:{label:ctx=>`conf=${ctx.parsed.x.toFixed(3)} acc=${ctx.parsed.y.toFixed(3)} n=${rd.sizes[ctx.dataIndex]}`}}}}});

// Confidence histogram
const ch={{ chart_data.conf_hist | tojson }};
new Chart(document.getElementById('confHist'),{type:'bar',
  data:{labels:ch.bins.map(b=>b.toFixed(2)),datasets:[{label:'Samples',data:ch.counts,
    backgroundColor:A2,borderRadius:3}]},
  options:{responsive:true,plugins:{legend:{display:false}},
    scales:{y:{grid:{color:BORDER}},x:{grid:{display:false}}}}});
{% endif %}
</script>
</body></html>
"""


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def _compute_reliability(
    samples: list[dict], n_bins: int = 10
) -> dict | None:
    """Compute binned confidence/accuracy for the reliability diagram."""
    import math as _math

    def _norm(t: str) -> str:
        import string
        t = t.lower().strip()
        t = t.translate(str.maketrans("", "", string.punctuation))
        return " ".join(t.split())

    valid = [
        s for s in samples
        if s.get("confidence") is not None
        and not _math.isnan(s["confidence"])
        and s.get("expected_output")
        and not s.get("error")
    ]
    if len(valid) < n_bins:
        return None

    confs = [s["confidence"] for s in valid]
    accs  = [
        1.0 if _norm(s["generated_output"]) == _norm(s["expected_output"]) else 0.0
        for s in valid
    ]

    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    points, sizes = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        bucket = [(c, a) for c, a in zip(confs, accs)
                  if c >= lo and (c <= hi if hi == 1.0 else c < hi)]
        if bucket:
            mean_c = sum(x[0] for x in bucket) / len(bucket)
            mean_a = sum(x[1] for x in bucket) / len(bucket)
            points.append({"x": round(mean_c, 4), "y": round(mean_a, 4)})
            sizes.append(len(bucket))

    ece = sum(
        (sz / len(valid)) * abs(p["x"] - p["y"])
        for p, sz in zip(points, sizes)
    )
    return {"points": points, "sizes": sizes, "ece": ece}


def _build_chart_data(samples: list[dict]) -> dict:
    latencies = [s.get("latency_ms") or 0.0 for s in samples]
    if latencies:
        min_l   = min(latencies)
        max_l   = max(latencies) or 1.0
        n_bins  = min(10, len(latencies))
        bw      = (max_l - min_l) / n_bins or 1
        lat_bins   = [round(min_l + i * bw) for i in range(n_bins)]
        lat_counts = [0] * n_bins
        for lat in latencies:
            idx = min(int((lat - min_l) / bw), n_bins - 1)
            lat_counts[idx] += 1
    else:
        lat_bins, lat_counts = [], []

    # Confidence histogram
    conf_vals = [s["confidence"] for s in samples
                 if s.get("confidence") is not None and not math.isnan(s["confidence"])]
    if conf_vals:
        conf_bins   = [round(i * 0.05, 2) for i in range(21)]
        conf_counts = [0] * 20
        for c in conf_vals:
            idx = min(int(c / 0.05), 19)
            conf_counts[idx] += 1
        conf_hist = {"bins": conf_bins[:20], "counts": conf_counts}
    else:
        conf_hist = {"bins": [], "counts": []}

    return {
        "latencies":   latencies,
        "lat_bins":    lat_bins,
        "lat_counts":  lat_counts,
        "scores":      {"labels": [], "values": []},
        "reliability": {"points": [], "sizes": []},
        "conf_hist":   conf_hist,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_html_report(
    run: dict[str, Any],
    samples: list[dict[str, Any]],
    max_samples_shown: int = 200,
) -> str:
    raw_scores = run.get("scores", {})
    if isinstance(raw_scores, str):
        try: raw_scores = json.loads(raw_scores)
        except: raw_scores = {}

    display_scores = {
        k: v for k, v in raw_scores.items()
        if isinstance(v, (int, float)) and not math.isnan(v)
    }

    primary_metric = primary_score = None
    for m in ["f1", "exact_match", "rouge_l", "bertscore",
              "turn_f1", "turn_exact_match"]:
        if m in display_scores:
            primary_metric = m
            primary_score  = display_scores[m]
            break

    # Parse scores in samples
    for s in samples:
        if isinstance(s.get("scores"), str):
            try: s["scores"] = json.loads(s["scores"])
            except: s["scores"] = {}

    chart_data = _build_chart_data(samples)
    if display_scores:
        chart_data["scores"] = {
            "labels": list(display_scores),
            "values": list(display_scores.values()),
        }

    # Reliability diagram
    reliability = _compute_reliability(samples)
    has_confidence = reliability is not None
    ece_val = reliability["ece"] if reliability else float("nan")
    if reliability:
        chart_data["reliability"] = {
            "points": reliability["points"],
            "sizes":  reliability["sizes"],
        }

    # Score columns for sample table
    score_metrics: list[str] = []
    for s in samples[:5]:
        for k in (s.get("scores") or {}):
            if k not in score_metrics:
                score_metrics.append(k)

    latencies   = [s.get("latency_ms") or 0.0 for s in samples]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    conf_vals    = [s["confidence"] for s in samples
                    if s.get("confidence") is not None]
    avg_confidence = sum(conf_vals) / len(conf_vals) if conf_vals else 0.0

    # Wrap samples for template (ensure confidence is accessible)
    class _Sample:
        def __init__(self, d): self.__dict__.update(d)

    wrapped = [_Sample(s) for s in samples[:max_samples_shown]]

    return Template(_TEMPLATE).render(
        run=run,
        samples=wrapped,
        chart_data=chart_data,
        display_scores=display_scores,
        score_metrics=score_metrics,
        primary_metric=primary_metric,
        primary_score=primary_score,
        avg_latency_ms=avg_latency,
        has_confidence=has_confidence,
        avg_confidence=avg_confidence,
        ece_val=ece_val,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from llmbench.store.db import store as _store

    if len(sys.argv) < 2:
        print("Usage: python -m llmbench.api.reporter <run_id> [output.html]")
        sys.exit(1)

    run_id   = sys.argv[1]
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"report_{run_id[:8]}.html")
    run_data = _store.get_run(run_id)
    if not run_data:
        print(f"Run '{run_id}' not found.")
        sys.exit(1)
    html = generate_html_report(run_data, _store.get_samples(run_id))
    out_path.write_text(html, encoding="utf-8")
    print(f"Written to {out_path}  ({len(html):,} bytes)")
