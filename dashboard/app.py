"""
dashboard/app.py
LLMBench Streamlit dashboard.

Tabs:
  📊 Overview          — run table + metric trend chart
  🔁 Compare Runs      — regression diff + radar chart
  🔍 Sample Inspector  — single-turn per-sample table + latency histogram
  💬 Conversation View — multi-turn per-turn breakdown          ← NEW
  📈 Calibration       — confidence vs accuracy reliability diagram ← NEW
  🏆 Leaderboard       — model ranking by metric
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from llmbench.store.db import ResultsStore, RegressionTracker

store   = ResultsStore()
tracker = RegressionTracker(store)

st.set_page_config(
    page_title="LLMBench",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .metric-pill {
    display:inline-block; padding:2px 10px; border-radius:999px;
    background:#1e293b; border:1px solid #334155;
    font-size:0.78rem; margin:2px;
  }
  .turn-card {
    background:#1a1d27; border:1px solid #2a2d3e;
    border-radius:8px; padding:1rem; margin-bottom:0.8rem;
  }
  .conf-badge { font-size:0.75rem; color:#94a3b8; }
</style>
""", unsafe_allow_html=True)

st.title("🔬 LLMBench")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("Filters")
runs = store.list_runs(limit=300)

if not runs:
    st.info("No runs yet. Run `llmbench run ...` or `llmbench run-conv ...` to get started.")
    st.stop()

datasets = sorted({r["dataset_name"] for r in runs})
models   = sorted({r["model_slug"]   for r in runs})
tasks    = sorted({r["task_type"]    for r in runs})

sel_dataset = st.sidebar.selectbox("Dataset",  ["All"] + datasets)
sel_model   = st.sidebar.selectbox("Model",    ["All"] + models)
sel_task    = st.sidebar.selectbox("Task",     ["All"] + tasks)

filtered = [
    r for r in runs
    if (sel_dataset == "All" or r["dataset_name"] == sel_dataset)
    and (sel_model   == "All" or r["model_slug"]   == sel_model)
    and (sel_task    == "All" or r["task_type"]    == sel_task)
]

run_ids = [r["run_id"] for r in filtered]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

(tab_overview, tab_compare, tab_samples,
 tab_conv, tab_calib, tab_leader) = st.tabs([
    "📊 Overview", "🔁 Compare Runs", "🔍 Sample Inspector",
    "💬 Conversation View", "📈 Calibration", "🏆 Leaderboard",
])

# ============================================================
# 📊 Overview
# ============================================================

with tab_overview:
    st.subheader(f"Recent runs — {len(filtered)} shown")

    if not filtered:
        st.warning("No runs match the selected filters.")
    else:
        all_metrics: list[str] = []
        for r in filtered:
            for m in r["scores"]:
                if m not in all_metrics:
                    all_metrics.append(m)

        rows = []
        for r in filtered:
            row = {
                "run_id":   r["run_id"][:10],
                "model":    r["model_slug"],
                "dataset":  r["dataset_name"],
                "task":     r["task_type"],
                "samples":  r["total_samples"],
                "failed":   r["failed_samples"],
                "tokens":   r["total_tokens"],
                "created":  (r["created_at"] or "")[:16],
            }
            row.update({
                m: round(r["scores"].get(m, float("nan")), 4)
                for m in all_metrics
            })
            rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(df, width='stretch', hide_index=True)

        if len(filtered) > 1 and all_metrics:
            st.markdown("---")
            sel_metric = st.selectbox("Metric trend", all_metrics)
            trend_df = df[["run_id", "model", sel_metric]].dropna()
            fig = px.bar(
                trend_df, x="run_id", y=sel_metric, color="model",
                title=f"{sel_metric} across runs",
            )
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, width='stretch')


# ============================================================
# 🔁 Compare Runs
# ============================================================

with tab_compare:
    st.subheader("Regression check")

    if len(run_ids) < 2:
        st.info("Need at least 2 runs to compare.")
    else:
        c1, c2, c3 = st.columns([3, 3, 1])
        baseline_id  = c1.selectbox("Baseline",  run_ids, index=0)
        candidate_id = c2.selectbox("Candidate", run_ids, index=min(1, len(run_ids)-1))
        threshold    = c3.number_input("Threshold", value=0.02, step=0.005, format="%.3f")

        if st.button("Compare", type="primary"):
            try:
                report = tracker.compare(baseline_id, candidate_id, threshold=threshold)
            except Exception as e:
                st.error(str(e))
            else:
                b, c = report["baseline"], report["candidate"]
                col_b, col_c = st.columns(2)
                col_b.metric("Baseline",  b["model"])
                col_c.metric("Candidate", c["model"])

                rows = []
                for metric in set(b["scores"]) | set(c["scores"]):
                    bv = b["scores"].get(metric, float("nan"))
                    cv = c["scores"].get(metric, float("nan"))
                    dv = report["deltas"].get(metric, float("nan"))
                    if metric in report["regressions"]:    status = "🔴 REGRESSION"
                    elif metric in report["improvements"]: status = "🟢 improved"
                    else:                                  status = "⚪ stable"
                    rows.append({"Metric": metric,
                                 "Baseline":  round(bv, 4) if bv == bv else None,
                                 "Candidate": round(cv, 4) if cv == cv else None,
                                 "Δ":         round(dv, 4) if dv == dv else None,
                                 "Status":    status})

                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

                if report["has_regression"]:
                    st.error(f"⚠ {len(report['regressions'])} regression(s) detected")
                else:
                    st.success("✓ No regressions detected")

                # Radar chart
                radar_rows = [r for r in rows if r["Baseline"] and r["Candidate"]]
                if radar_rows:
                    cats   = [r["Metric"]    for r in radar_rows]
                    b_vals = [r["Baseline"]  for r in radar_rows]
                    c_vals = [r["Candidate"] for r in radar_rows]
                    fig = go.Figure([
                        go.Scatterpolar(r=b_vals, theta=cats, fill="toself", name=b["model"]),
                        go.Scatterpolar(r=c_vals, theta=cats, fill="toself", name=c["model"]),
                    ])
                    fig.update_layout(
                        title="Metric Radar",
                        polar=dict(radialaxis=dict(range=[0, 1])),
                    )
                    st.plotly_chart(fig, width='stretch')


# ============================================================
# 🔍 Sample Inspector  (single-turn)
# ============================================================

with tab_samples:
    st.subheader("Per-sample breakdown")

    single_run_ids = [r["run_id"] for r in filtered if r["task_type"] != "multi_turn"]
    if not single_run_ids:
        st.info("No single-turn runs available.")
    else:
        inspect_id = st.selectbox("Select run", single_run_ids, key="sample_run")
        if inspect_id:
            samples = store.get_samples(inspect_id)
            if not samples:
                st.info("No samples stored.")
            else:
                sample_df = pd.DataFrame(samples)

                q = st.text_input("Filter output (substring)", key="sample_filter")
                if q:
                    sample_df = sample_df[
                        sample_df["generated_output"].str.contains(q, case=False, na=False)
                    ]
                if st.checkbox("Show only errors", key="sample_err"):
                    sample_df = sample_df[sample_df["error"].notna()]

                # Parse scores JSON if stored as string
                def _parse_scores(v):
                    if isinstance(v, str):
                        try: return json.loads(v)
                        except: return {}
                    return v or {}

                sample_df["scores"] = sample_df["scores"].apply(_parse_scores)

                display_cols = ["sample_id", "generated_output", "expected_output",
                                "confidence", "latency_ms", "error"]
                st.dataframe(sample_df[display_cols], width='stretch')

                # Confidence distribution
                if "confidence" in sample_df.columns:
                    conf_vals = sample_df["confidence"].dropna()
                    if not conf_vals.empty:
                        fig = px.histogram(conf_vals, nbins=20,
                                           title="Confidence distribution",
                                           labels={"value": "Confidence"})
                        st.plotly_chart(fig, width='stretch')

                # Latency
                if "latency_ms" in sample_df.columns:
                    fig = px.histogram(sample_df["latency_ms"].dropna(), nbins=30,
                                       title="Latency distribution (ms)")
                    st.plotly_chart(fig, width='stretch')


# ============================================================
# 💬 Conversation View  (multi-turn)              ← NEW
# ============================================================

with tab_conv:
    st.subheader("Multi-turn conversation breakdown")

    mt_run_ids = [r["run_id"] for r in filtered if r["task_type"] == "multi_turn"]
    if not mt_run_ids:
        st.info("No multi-turn runs yet. Run `llmbench run-conv ...` first.")
    else:
        mt_run_id = st.selectbox("Select run", mt_run_ids, key="conv_run")
        if mt_run_id:
            conv_samples = store.get_samples(mt_run_id)

            if not conv_samples:
                st.info("No conversation samples found.")
            else:
                # Summary table across all conversations
                rows = []
                for s in conv_samples:
                    sc = s.get("scores", {})
                    if isinstance(sc, str):
                        try: sc = json.loads(sc)
                        except: sc = {}
                    rows.append({
                        "conversation_id": (s["sample_id"] or "")[:12],
                        "turn_em":  round(sc.get("turn_exact_match", float("nan")), 3),
                        "turn_f1":  round(sc.get("turn_f1",           float("nan")), 3),
                        "rouge_l":  round(sc.get("turn_rouge_l",      float("nan")), 3),
                        "confidence": round(s.get("confidence") or float("nan"), 3),
                        "latency_ms": round(s.get("latency_ms") or 0, 1),
                    })
                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

                st.markdown("---")
                st.markdown("#### Turn-by-turn drill-down")

                conv_ids = [s["sample_id"] for s in conv_samples]
                sel_conv = st.selectbox("Select conversation", conv_ids, key="sel_conv")

                if sel_conv:
                    turns = store.get_turn_results(mt_run_id, conversation_id=sel_conv)

                    if not turns:
                        st.info("No turn records stored for this conversation. "
                                "Turn records are saved when `store.save_turn_results()` "
                                "is called explicitly after the run.")
                    else:
                        for t in sorted(turns, key=lambda x: x["turn_index"]):
                            sc = t.get("scores", {})
                            if isinstance(sc, str):
                                try: sc = json.loads(sc)
                                except: sc = {}

                            em  = sc.get("turn_exact_match", float("nan"))
                            f1  = sc.get("turn_f1",          float("nan"))
                            rl  = sc.get("turn_rouge_l",     float("nan"))
                            conf = t.get("confidence")

                            em_icon = "✅" if em == 1.0 else ("❌" if em == 0.0 else "—")

                            with st.expander(
                                f"Turn {t['turn_index'] + 1}  {em_icon}  "
                                f"EM={em:.2f}  F1={f1:.2f}  ROUGE-L={rl:.2f}  "
                                f"conf={conf:.2f if conf else '—'}",
                                expanded=(t["turn_index"] == 0),
                            ):
                                col_g, col_e = st.columns(2)
                                col_g.markdown("**Generated**")
                                col_g.markdown(
                                    f"> {t['generated_output'] or '*(empty)*'}"
                                )
                                col_e.markdown("**Expected**")
                                col_e.markdown(
                                    f"> {t['expected_output'] or '*(none)*'}"
                                )
                                if t.get("error"):
                                    st.error(f"Error: {t['error']}")
                                if conf is not None:
                                    st.caption(
                                        f"Confidence: {conf:.4f}  ·  "
                                        f"Latency: {t.get('latency_ms', 0):.0f}ms"
                                    )

                        # Turn-level score chart
                        if len(turns) > 1:
                            chart_rows = []
                            for t in sorted(turns, key=lambda x: x["turn_index"]):
                                sc = t.get("scores", {})
                                if isinstance(sc, str):
                                    try: sc = json.loads(sc)
                                    except: sc = {}
                                chart_rows.append({
                                    "turn": f"Turn {t['turn_index']+1}",
                                    "Exact Match": sc.get("turn_exact_match", float("nan")),
                                    "F1":          sc.get("turn_f1",          float("nan")),
                                    "ROUGE-L":     sc.get("turn_rouge_l",     float("nan")),
                                })
                            cdf = pd.DataFrame(chart_rows).set_index("turn")
                            fig = px.line(
                                cdf.reset_index().melt(id_vars="turn"),
                                x="turn", y="value", color="variable",
                                title="Scores across turns",
                                markers=True,
                            )
                            fig.update_layout(yaxis=dict(range=[0, 1.05]))
                            st.plotly_chart(fig, width='stretch')


# ============================================================
# 📈 Calibration — reliability diagram               ← NEW
# ============================================================

with tab_calib:
    st.subheader("Calibration — confidence vs accuracy")
    st.caption(
        "A well-calibrated model should have accuracy ≈ confidence in each bin. "
        "Points on the diagonal = perfect calibration. "
        "Confidence is extracted from token log-probabilities by the provider."
    )

    calib_run_ids = [r["run_id"] for r in filtered
                     if r["task_type"] not in ("multi_turn",)]
    if not calib_run_ids:
        st.info("No single-turn runs available for calibration analysis.")
    else:
        calib_run_id = st.selectbox("Select run", calib_run_ids, key="calib_run")
        n_bins       = st.slider("Bins", min_value=5, max_value=20, value=10)

        if calib_run_id:
            calib_samples = store.get_samples(calib_run_id)

            valid = [
                s for s in calib_samples
                if s.get("confidence") is not None
                and s.get("expected_output") is not None
                and not s.get("error")
            ]

            if len(valid) < n_bins:
                st.warning(
                    f"Only {len(valid)} samples have confidence scores "
                    f"(need ≥ {n_bins}). "
                    "Confidence requires a provider that supports log-probs "
                    "(OpenAI, Groq, vLLM). Anthropic always returns None."
                )
            else:
                import numpy as np

                # Normalise text for exact-match correctness
                def _norm(t: str) -> str:
                    import string
                    t = t.lower().strip()
                    t = t.translate(str.maketrans("", "", string.punctuation))
                    return " ".join(t.split())

                confs  = np.array([s["confidence"] for s in valid])
                accs   = np.array([
                    1.0 if _norm(s["generated_output"]) == _norm(s["expected_output"]) else 0.0
                    for s in valid
                ])

                # Bin
                bin_edges = np.linspace(0, 1, n_bins + 1)
                bin_confs, bin_accs, bin_sizes = [], [], []
                for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                    mask = (confs >= lo) & (confs <= hi)
                    if mask.sum() > 0:
                        bin_confs.append(float(confs[mask].mean()))
                        bin_accs.append(float(accs[mask].mean()))
                        bin_sizes.append(int(mask.sum()))

                # ECE
                ece = sum(
                    (sz / len(valid)) * abs(bc - ba)
                    for bc, ba, sz in zip(bin_confs, bin_accs, bin_sizes)
                )

                col_ece, col_n = st.columns(2)
                col_ece.metric("ECE (lower = better)", f"{ece:.4f}")
                col_n.metric("Samples with confidence", len(valid))

                # Reliability diagram
                fig = go.Figure()

                # Perfect calibration diagonal
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines", name="Perfect calibration",
                    line=dict(dash="dash", color="#64748b", width=1),
                ))

                # Actual calibration points (sized by bin count)
                fig.add_trace(go.Scatter(
                    x=bin_confs, y=bin_accs,
                    mode="markers+lines",
                    name="Model calibration",
                    marker=dict(
                        size=[max(8, s // 2) for s in bin_sizes],
                        color="#7c6af7",
                        line=dict(width=1, color="#fff"),
                    ),
                    line=dict(color="#7c6af7", width=2),
                    text=[f"n={s}" for s in bin_sizes],
                    hovertemplate=(
                        "Confidence: %{x:.3f}<br>"
                        "Accuracy: %{y:.3f}<br>"
                        "%{text}<extra></extra>"
                    ),
                ))

                # Gap shading (overconfident = red, underconfident = green)
                for bc, ba in zip(bin_confs, bin_accs):
                    color = "rgba(239,68,68,0.15)" if bc > ba else "rgba(34,197,94,0.10)"
                    fig.add_shape(
                        type="rect",
                        x0=bc - 0.01, x1=bc + 0.01,
                        y0=min(bc, ba), y1=max(bc, ba),
                        fillcolor=color, line_width=0,
                    )

                fig.update_layout(
                    title="Reliability Diagram",
                    xaxis=dict(title="Mean confidence", range=[0, 1]),
                    yaxis=dict(title="Fraction correct",  range=[0, 1]),
                    legend=dict(x=0.02, y=0.98),
                    width=600, height=500,
                )
                st.plotly_chart(fig, width='stretch')

                # Confidence histogram
                fig2 = px.histogram(
                    confs, nbins=30,
                    title="Confidence score distribution",
                    labels={"value": "Confidence", "count": "Samples"},
                    color_discrete_sequence=["#06b6d4"],
                )
                st.plotly_chart(fig2, width='stretch')

                # Sample table
                with st.expander("Raw samples"):
                    df_raw = pd.DataFrame([{
                        "generated":  (s["generated_output"] or "")[:80],
                        "expected":   (s["expected_output"]  or "")[:50],
                        "confidence": round(s["confidence"], 4),
                        "correct":    _norm(s["generated_output"]) == _norm(s["expected_output"]),
                    } for s in valid])
                    st.dataframe(df_raw, width='stretch', hide_index=True)


# ============================================================
# 🏆 Leaderboard
# ============================================================

with tab_leader:
    st.subheader("Model leaderboard")

    all_metrics_flat: list[str] = []
    for r in filtered:
        for m in r["scores"]:
            if m not in all_metrics_flat:
                all_metrics_flat.append(m)

    if not all_metrics_flat:
        st.info("No scored runs yet.")
    else:
        rank_metric  = st.selectbox("Rank by",  all_metrics_flat, key="lb_metric")
        rank_dataset = st.selectbox("Dataset",  ["All"] + datasets, key="lb_dataset")

        lb_runs = store.list_runs(
            dataset_name=rank_dataset if rank_dataset != "All" else None,
            limit=200,
        )
        lb_rows = [
            {
                "model":   r["model_slug"],
                "dataset": r["dataset_name"],
                "task":    r["task_type"],
                "run_id":  r["run_id"][:10],
                "score":   r["scores"].get(rank_metric, float("nan")),
                "samples": r["total_samples"],
                "created": (r["created_at"] or "")[:16],
            }
            for r in lb_runs
            if rank_metric in r["scores"]
        ]
        lb_rows.sort(
            key=lambda x: x["score"] if not math.isnan(x["score"]) else -1,
            reverse=(rank_metric != "ece"),
        )

        if lb_rows:
            lb_df = pd.DataFrame(lb_rows)
            lb_df.index = range(1, len(lb_df) + 1)
            st.dataframe(lb_df, width='stretch')

            fig = px.bar(
                lb_df.head(10), x="model", y="score", color="task",
                title=f"Top 10 — {rank_metric}",
            )
            st.plotly_chart(fig, width='stretch')
