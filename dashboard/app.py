"""
dashboard/app.py
LLMBench Streamlit dashboard — visual run comparison, regression view, sample inspector.

Run: streamlit run dashboard/app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make sure llmbench is importable from dashboard/
sys.path.insert(0, str(Path(__file__).parent.parent))

from llmbench.store.db import ResultsStore, RegressionTracker

store   = ResultsStore()
tracker = RegressionTracker(store)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LLMBench",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 LLMBench — LLM Evaluation Dashboard")

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.header("Filters")

runs = store.list_runs(limit=200)
if not runs:
    st.info("No runs in the database yet. Run `llmbench run ...` to get started.")
    st.stop()

datasets  = sorted({r["dataset_name"] for r in runs})
models    = sorted({r["model_slug"]   for r in runs})
tasks     = sorted({r["task_type"]    for r in runs})

sel_dataset = st.sidebar.selectbox("Dataset",  ["All"] + datasets)
sel_model   = st.sidebar.selectbox("Model",    ["All"] + models)
sel_task    = st.sidebar.selectbox("Task",     ["All"] + tasks)

filtered = [
    r for r in runs
    if (sel_dataset == "All" or r["dataset_name"] == sel_dataset)
    and (sel_model   == "All" or r["model_slug"]   == sel_model)
    and (sel_task    == "All" or r["task_type"]    == sel_task)
]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_compare, tab_samples, tab_leaderboard = st.tabs([
    "📊 Overview", "🔁 Compare Runs", "🔍 Sample Inspector", "🏆 Leaderboard"
])

# -----------------------------------------------------------------------
# Tab 1: Overview
# -----------------------------------------------------------------------

with tab_overview:
    st.subheader(f"Recent runs ({len(filtered)} shown)")

    if not filtered:
        st.warning("No runs match the selected filters.")
    else:
        # Build a flat scores DataFrame
        rows = []
        all_metrics: list[str] = []
        for r in filtered:
            for m in r["scores"]:
                if m not in all_metrics:
                    all_metrics.append(m)

        for r in filtered:
            row = {
                "run_id":    r["run_id"][:10],
                "model":     r["model_slug"],
                "dataset":   r["dataset_name"],
                "task":      r["task_type"],
                "samples":   r["total_samples"],
                "failed":    r["failed_samples"],
                "tokens":    r["total_tokens"],
                "created":   (r["created_at"] or "")[:16],
            }
            row.update({m: round(r["scores"].get(m, float("nan")), 4)
                        for m in all_metrics})
            rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(df, width='stretch', hide_index=True)

        # Score trends over time
        if len(filtered) > 1 and all_metrics:
            st.markdown("---")
            sel_metric = st.selectbox("Metric to plot over runs", all_metrics)
            trend_df = df[["run_id", "model", sel_metric]].dropna()
            fig = px.bar(
                trend_df, x="run_id", y=sel_metric, color="model",
                title=f"{sel_metric} across runs",
                labels={"run_id": "Run", sel_metric: sel_metric},
            )
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, width='stretch')

# -----------------------------------------------------------------------
# Tab 2: Compare Runs
# -----------------------------------------------------------------------

with tab_compare:
    st.subheader("Regression check — compare two runs")

    run_ids = [r["run_id"] for r in filtered]
    if len(run_ids) < 2:
        st.info("Need at least 2 runs to compare.")
    else:
        col1, col2, col3 = st.columns([3, 3, 1])
        with col1:
            baseline_id  = st.selectbox("Baseline run",  run_ids, index=0)
        with col2:
            candidate_id = st.selectbox("Candidate run", run_ids, index=1)
        with col3:
            threshold = st.number_input("Threshold", value=0.02, step=0.005, format="%.3f")

        if st.button("Compare", type="primary"):
            try:
                report = tracker.compare(baseline_id, candidate_id, threshold=threshold)
            except Exception as e:
                st.error(str(e))
            else:
                b = report["baseline"]
                c = report["candidate"]

                col_b, col_c = st.columns(2)
                col_b.metric("Baseline model",  b["model"])
                col_c.metric("Candidate model", c["model"])

                rows = []
                for metric in set(b["scores"]) | set(c["scores"]):
                    bv = b["scores"].get(metric, float("nan"))
                    cv = c["scores"].get(metric, float("nan"))
                    dv = report["deltas"].get(metric, float("nan"))
                    if metric in report["regressions"]:
                        status = "🔴 REGRESSION"
                    elif metric in report["improvements"]:
                        status = "🟢 improved"
                    else:
                        status = "⚪ stable"
                    rows.append({"Metric": metric,
                                 "Baseline": round(bv, 4),
                                 "Candidate": round(cv, 4),
                                 "Δ": round(dv, 4),
                                 "Status": status})

                cmp_df = pd.DataFrame(rows)
                st.dataframe(cmp_df, width='stretch', hide_index=True)

                if report["has_regression"]:
                    st.error(f"⚠ {len(report['regressions'])} regression(s) detected")
                else:
                    st.success("✓ No regressions detected")

                # Radar chart
                radar_metrics = [r for r in rows
                                 if not any(map(lambda x: x != x,
                                               [r["Baseline"], r["Candidate"]]))]
                if radar_metrics:
                    cats   = [r["Metric"]    for r in radar_metrics]
                    b_vals = [r["Baseline"]  for r in radar_metrics]
                    c_vals = [r["Candidate"] for r in radar_metrics]
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(r=b_vals, theta=cats, fill="toself",
                                                  name=b["model"]))
                    fig.add_trace(go.Scatterpolar(r=c_vals, theta=cats, fill="toself",
                                                  name=c["model"]))
                    fig.update_layout(title="Metric Radar", polar=dict(radialaxis=dict(range=[0, 1])))
                    st.plotly_chart(fig, width='stretch')

# -----------------------------------------------------------------------
# Tab 3: Sample Inspector
# -----------------------------------------------------------------------

with tab_samples:
    st.subheader("Per-sample breakdown")

    if not run_ids:
        st.info("No runs available.")
    else:
        inspect_id = st.selectbox("Select run", run_ids)
        if inspect_id:
            samples = store.get_samples(inspect_id)
            if not samples:
                st.info("No samples stored for this run.")
            else:
                sample_df = pd.DataFrame(samples)

                # Filter bar
                query = st.text_input("Filter generated output (substring)")
                if query:
                    sample_df = sample_df[
                        sample_df["generated_output"].str.contains(query, case=False, na=False)
                    ]

                # Show only failed
                show_failed = st.checkbox("Show only failed samples")
                if show_failed:
                    sample_df = sample_df[sample_df["error"].notna()]

                st.dataframe(sample_df[[
                    "sample_id", "generated_output", "expected_output",
                    "scores", "judge_reasoning", "latency_ms", "error"
                ]], width='stretch')

                # Latency histogram
                if "latency_ms" in sample_df.columns:
                    fig = px.histogram(sample_df, x="latency_ms",
                                       title="Latency distribution (ms)",
                                       nbins=30)
                    st.plotly_chart(fig, width='stretch')

# -----------------------------------------------------------------------
# Tab 4: Leaderboard
# -----------------------------------------------------------------------

with tab_leaderboard:
    st.subheader("Model leaderboard")

    all_metrics_flat: list[str] = []
    for r in filtered:
        for m in r["scores"]:
            if m not in all_metrics_flat:
                all_metrics_flat.append(m)

    if not all_metrics_flat:
        st.info("No scored runs yet.")
    else:
        rank_metric   = st.selectbox("Rank by metric", all_metrics_flat)
        rank_dataset  = st.selectbox("On dataset", ["All"] + datasets)

        lb_runs = store.list_runs(
            dataset_name=rank_dataset if rank_dataset != "All" else None,
            limit=200,
        )
        lb_rows = [
            {
                "model":   r["model_slug"],
                "dataset": r["dataset_name"],
                "run_id":  r["run_id"][:10],
                "score":   r["scores"].get(rank_metric, float("nan")),
                "samples": r["total_samples"],
                "created": (r["created_at"] or "")[:16],
            }
            for r in lb_runs if rank_metric in r["scores"]
        ]
        lb_rows.sort(key=lambda x: x["score"] if x["score"] == x["score"] else -1,
                     reverse=(rank_metric != "ece"))

        lb_df = pd.DataFrame(lb_rows)
        if not lb_df.empty:
            lb_df.index = range(1, len(lb_df) + 1)
            st.dataframe(lb_df, width='stretch')

            fig = px.bar(lb_df.head(10), x="model", y="score",
                         color="model", title=f"Top 10 — {rank_metric}")
            st.plotly_chart(fig, width='stretch')
