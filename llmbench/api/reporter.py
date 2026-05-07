"""
llmbench/api/reporter.py
Self-contained HTML report generator.
All CSS, JS (Chart.js via CDN), and data are embedded — one file, zero deps.

Usage:
    from llmbench.api.reporter import generate_html_report
    html = generate_html_report(run_dict, sample_dicts)
    Path("report.html").write_text(html)
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

from jinja2 import Template

# ---------------------------------------------------------------------------
# Jinja2 HTML template
# ---------------------------------------------------------------------------

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLMBench Report — {{ run.run_id[:8] }}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root {
    --bg:       #0f1117;
    --surface:  #1a1d27;
    --border:   #2a2d3e;
    --accent:   #7c6af7;
    --accent2:  #06b6d4;
    --green:    #22c55e;
    --red:      #ef4444;
    --amber:    #f59e0b;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --radius:   10px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 2rem;
  }
  a { color: var(--accent); }

  /* Header */
  .header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
  }
  .header h1 { font-size: 1.6rem; font-weight: 700; }
  .header h1 span { color: var(--accent); }
  .badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--muted);
  }

  /* Stat cards */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }
  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
  }
  .stat-card .label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem; }
  .stat-card .value { font-size: 1.6rem; font-weight: 700; }
  .stat-card .value.green { color: var(--green); }
  .stat-card .value.amber { color: var(--amber); }
  .stat-card .value.red   { color: var(--red);   }

  /* Section */
  .section { margin-bottom: 2.5rem; }
  .section h2 {
    font-size: 1rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
  }

  /* Metric scores */
  .score-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
  }
  .score-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
  }
  .score-card .metric-name {
    font-size: 0.8rem;
    color: var(--muted);
    margin-bottom: 0.5rem;
  }
  .score-card .metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent);
  }
  .score-card .progress-bar {
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    margin-top: 0.6rem;
    overflow: hidden;
  }
  .score-card .progress-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
  }

  /* Charts */
  .chart-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-bottom: 2rem;
  }
  @media (max-width: 800px) { .chart-grid { grid-template-columns: 1fr; } }
  .chart-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem;
  }
  .chart-box h3 { font-size: 0.85rem; color: var(--muted); margin-bottom: 1rem; }

  /* Table */
  .table-wrap { overflow-x: auto; }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
  }
  thead tr { border-bottom: 2px solid var(--border); }
  th { padding: 0.6rem 0.8rem; text-align: left; color: var(--muted); font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
  td { padding: 0.6rem 0.8rem; border-bottom: 1px solid var(--border); vertical-align: top; max-width: 320px; word-break: break-word; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(124,106,247,0.04); }

  .tag { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600; }
  .tag.ok   { background: rgba(34,197,94,0.15); color: var(--green); }
  .tag.err  { background: rgba(239,68,68,0.15);  color: var(--red);   }

  /* Config section */
  .config-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    font-size: 0.85rem;
  }
  .config-row { display: flex; gap: 1rem; margin-bottom: 0.4rem; }
  .config-row .key { color: var(--muted); min-width: 160px; }
  .config-row .val { color: var(--text); font-family: 'Fira Code', monospace; }

  /* Footer */
  footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); font-size: 0.75rem; color: var(--muted); text-align: center; }
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div>
    <h1>🔬 LLMBench — <span>Run Report</span></h1>
    <p style="color:var(--muted);font-size:0.85rem;margin-top:0.4rem;">
      {{ run.run_id }} &nbsp;·&nbsp; {{ run.created_at[:16].replace('T', ' ') if run.created_at else 'Unknown' }} UTC
    </p>
  </div>
  <div style="text-align:right;">
    <span class="badge">{{ run.model_slug }}</span>&nbsp;
    <span class="badge">{{ run.task_type }}</span>&nbsp;
    <span class="badge">{{ run.dataset_name }}</span>
  </div>
</div>

<!-- STAT CARDS -->
<div class="stats-grid">
  <div class="stat-card">
    <div class="label">Total Samples</div>
    <div class="value">{{ run.total_samples }}</div>
  </div>
  <div class="stat-card">
    <div class="label">Failed</div>
    <div class="value {{ 'red' if run.failed_samples > 0 else 'green' }}">{{ run.failed_samples }}</div>
  </div>
  <div class="stat-card">
    <div class="label">Total Tokens</div>
    <div class="value">{{ "{:,}".format(run.total_tokens) }}</div>
  </div>
  <div class="stat-card">
    <div class="label">Avg Latency</div>
    <div class="value amber">{{ "%.0f"|format(avg_latency_ms) }}ms</div>
  </div>
  {% if primary_score is not none %}
  <div class="stat-card">
    <div class="label">{{ primary_metric }}</div>
    <div class="value" style="color:var(--accent);">{{ "%.4f"|format(primary_score) }}</div>
  </div>
  {% endif %}
</div>

<!-- METRIC SCORES -->
<div class="section">
  <h2>Aggregate Scores</h2>
  <div class="score-grid">
    {% for metric, score in run.scores.items() %}
    <div class="score-card">
      <div class="metric-name">{{ metric }}</div>
      <div class="metric-value">
        {% if score != score %}N/A
        {% else %}{{ "%.4f"|format(score) }}{% endif %}
      </div>
      {% if score == score %}
      <div class="progress-bar">
        <div class="progress-fill" style="width: {{ [score * 100, 100]|min }}%;"></div>
      </div>
      {% endif %}
    </div>
    {% endfor %}
  </div>
</div>

<!-- CHARTS -->
{% if chart_data %}
<div class="section">
  <h2>Visualizations</h2>
  <div class="chart-grid">
    <!-- Score bar chart -->
    <div class="chart-box">
      <h3>Metric Scores</h3>
      <canvas id="scoreBar"></canvas>
    </div>
    <!-- Latency histogram -->
    <div class="chart-box">
      <h3>Latency Distribution (ms)</h3>
      <canvas id="latencyHist"></canvas>
    </div>
  </div>
</div>
{% endif %}

<!-- SAMPLE TABLE -->
{% if samples %}
<div class="section">
  <h2>Sample Breakdown ({{ samples|length }} shown)</h2>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Input</th>
          <th>Generated</th>
          <th>Expected</th>
          {% for m in score_metrics %}<th>{{ m }}</th>{% endfor %}
          <th>Latency</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {% for s in samples %}
        <tr>
          <td style="color:var(--muted);font-size:0.75rem;">{{ loop.index }}</td>
          <td>{{ (s.get('sample_id','') or '')[:40] }}</td>
          <td>{{ (s.generated_output or '')[:120] }}{% if s.generated_output and s.generated_output|length > 120 %}…{% endif %}</td>
          <td style="color:var(--muted);">{{ (s.expected_output or '—')[:80] }}{% if s.expected_output and s.expected_output|length > 80 %}…{% endif %}</td>
          {% for m in score_metrics %}
            <td>
              {% set sc = s.get('scores', {}) %}
              {% if sc is mapping and m in sc %}{{ "%.3f"|format(sc[m]) }}{% else %}—{% endif %}
            </td>
          {% endfor %}
          <td style="color:var(--muted);">{{ "%.0f"|format(s.latency_ms or 0) }}ms</td>
          <td>
            {% if s.error %}
              <span class="tag err">ERR</span>
            {% else %}
              <span class="tag ok">OK</span>
            {% endif %}
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>
{% endif %}

<!-- RUN CONFIG -->
<div class="section">
  <h2>Run Configuration</h2>
  <div class="config-box">
    <div class="config-row"><span class="key">Run ID</span><span class="val">{{ run.run_id }}</span></div>
    <div class="config-row"><span class="key">Model</span><span class="val">{{ run.model_slug }}</span></div>
    <div class="config-row"><span class="key">Dataset</span><span class="val">{{ run.dataset_name }}</span></div>
    <div class="config-row"><span class="key">Task type</span><span class="val">{{ run.task_type }}</span></div>
    <div class="config-row"><span class="key">Total samples</span><span class="val">{{ run.total_samples }}</span></div>
    <div class="config-row"><span class="key">Failed samples</span><span class="val">{{ run.failed_samples }}</span></div>
    <div class="config-row"><span class="key">Total tokens</span><span class="val">{{ "{:,}".format(run.total_tokens) }}</span></div>
    {% if run.tags %}
    <div class="config-row"><span class="key">Tags</span><span class="val">{{ run.tags }}</span></div>
    {% endif %}
  </div>
</div>

<footer>Generated by <strong>LLMBench</strong> · {{ generated_at }} UTC</footer>

<!-- CHART SCRIPTS -->
{% if chart_data %}
<script>
const ACCENT  = '#7c6af7';
const ACCENT2 = '#06b6d4';
const SURFACE = '#1a1d27';
const BORDER  = '#2a2d3e';
const TEXT    = '#e2e8f0';
const MUTED   = '#64748b';

Chart.defaults.color = MUTED;
Chart.defaults.borderColor = BORDER;

// ── Score bar chart ──────────────────────────────────────────────────────
const scoreData = {{ chart_data.scores | tojson }};
new Chart(document.getElementById('scoreBar'), {
  type: 'bar',
  data: {
    labels: scoreData.labels,
    datasets: [{
      label: 'Score',
      data: scoreData.values,
      backgroundColor: scoreData.labels.map((_, i) =>
        i % 2 === 0 ? ACCENT : ACCENT2),
      borderRadius: 6,
    }]
  },
  options: {
    responsive: true,
    plugins: { legend: { display: false } },
    scales: {
      y: {
        min: 0, max: 1,
        grid: { color: BORDER },
        ticks: { color: MUTED },
      },
      x: { grid: { display: false }, ticks: { color: MUTED } }
    }
  }
});

// ── Latency histogram ────────────────────────────────────────────────────
const latencies = {{ chart_data.latencies | tojson }};
const bins = {{ chart_data.lat_bins | tojson }};
const counts = {{ chart_data.lat_counts | tojson }};

new Chart(document.getElementById('latencyHist'), {
  type: 'bar',
  data: {
    labels: bins.map(b => b + 'ms'),
    datasets: [{
      label: 'Samples',
      data: counts,
      backgroundColor: ACCENT2,
      borderRadius: 4,
    }]
  },
  options: {
    responsive: true,
    plugins: { legend: { display: false } },
    scales: {
      y: { grid: { color: BORDER }, ticks: { color: MUTED } },
      x: { grid: { display: false }, ticks: { color: MUTED, maxRotation: 30 } }
    }
  }
});
</script>
{% endif %}
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Report generation function
# ---------------------------------------------------------------------------

def _build_chart_data(samples: list[dict]) -> dict | None:
    """Prepare chart data from sample results."""
    if not samples:
        return None

    latencies = [s.get("latency_ms") or 0.0 for s in samples]

    # Latency histogram — 10 bins
    if latencies:
        min_lat = min(latencies)
        max_lat = max(latencies) or 1.0
        n_bins = min(10, len(latencies))
        bin_width = (max_lat - min_lat) / n_bins or 1
        bins = [round(min_lat + i * bin_width) for i in range(n_bins)]
        counts = [0] * n_bins
        for lat in latencies:
            idx = min(int((lat - min_lat) / bin_width), n_bins - 1)
            counts[idx] += 1
    else:
        bins, counts = [], []

    return {
        "latencies": latencies,
        "lat_bins":  bins,
        "lat_counts": counts,
        "scores": {"labels": [], "values": []},   # filled below
    }


def generate_html_report(
    run: dict[str, Any],
    samples: list[dict[str, Any]],
    max_samples_shown: int = 200,
) -> str:
    """
    Generate a fully self-contained HTML report.

    Args:
        run:     Run dict from ResultsStore.get_run()
        samples: Sample dicts from ResultsStore.get_samples()
        max_samples_shown: Cap how many rows appear in the table.

    Returns:
        HTML string ready to write to disk or serve via HTTP.
    """
    # Parse scores — handle both dict and JSON string
    raw_scores = run.get("scores", {})
    if isinstance(raw_scores, str):
        try:
            raw_scores = json.loads(raw_scores)
        except Exception:
            raw_scores = {}

    # Filter out NaN scores for display
    display_scores = {
        k: v for k, v in raw_scores.items()
        if isinstance(v, (int, float)) and not math.isnan(v)
    }

    # Primary metric — pick first non-ECE metric
    primary_metric = None
    primary_score  = None
    for m in ["f1", "exact_match", "rouge_l", "bertscore"]:
        if m in display_scores:
            primary_metric = m
            primary_score  = display_scores[m]
            break

    # Chart data
    chart_data = _build_chart_data(samples)
    if chart_data and display_scores:
        chart_data["scores"] = {
            "labels": list(display_scores),
            "values": list(display_scores.values()),
        }

    # Score metrics to show as columns in sample table
    score_metrics: list[str] = []
    if samples:
        for s in samples[:5]:
            sc = s.get("scores", {})
            if isinstance(sc, str):
                try:
                    sc = json.loads(sc)
                except Exception:
                    sc = {}
            for k in sc:
                if k not in score_metrics:
                    score_metrics.append(k)

    # Avg latency
    latencies = [s.get("latency_ms") or 0.0 for s in samples]
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

    # Render
    template = Template(_TEMPLATE)
    html = template.render(
        run=run,
        samples=samples[:max_samples_shown],
        chart_data=chart_data,
        score_metrics=score_metrics,
        primary_metric=primary_metric,
        primary_score=primary_score,
        avg_latency_ms=avg_latency_ms,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    )
    return html


# ---------------------------------------------------------------------------
# CLI convenience — call directly to generate a report from a run_id
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from llmbench.store.db import store as _store

    if len(sys.argv) < 2:
        print("Usage: python -m llmbench.api.reporter <run_id> [output.html]")
        sys.exit(1)

    run_id = sys.argv[1]
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"report_{run_id[:8]}.html")

    run_data = _store.get_run(run_id)
    if not run_data:
        print(f"Run '{run_id}' not found.")
        sys.exit(1)

    sample_data = _store.get_samples(run_id)
    html = generate_html_report(run_data, sample_data)
    out_path.write_text(html, encoding="utf-8")
    print(f"Report written to {out_path}  ({len(html):,} bytes)")
