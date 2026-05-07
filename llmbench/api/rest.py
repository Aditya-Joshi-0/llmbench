"""
llmbench/api/rest.py
FastAPI REST API — run evals, query results, compare runs, all async.

Endpoints:
  POST /runs                  Trigger an eval run
  GET  /runs                  List stored runs (with filters)
  GET  /runs/{run_id}         Get run summary + scores
  GET  /runs/{run_id}/samples Per-sample breakdown
  GET  /runs/{run_id}/report  Download HTML report
  POST /runs/compare          Regression diff between two runs
  GET  /tasks                 List registered tasks
  GET  /metrics               List registered metrics
  GET  /health                Health check

Run: uvicorn llmbench.api.rest:app --reload --port 8080
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Trigger metric + task registration
import llmbench.metrics  # noqa: F401
from llmbench.core.registry import registry
from llmbench.core.runner import build_runner
from llmbench.core.schema import ModelConfig, TaskType
from llmbench.loaders import DatasetLoader
from llmbench.store.db import store, tracker
from llmbench.api.reporter import generate_html_report

app = FastAPI(
    title="LLMBench API",
    description="Production-grade LLM evaluation harness — REST interface",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

loader = DatasetLoader()

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ModelSpec(BaseModel):
    provider: str                           # e.g. "groq"
    model_id: str                           # e.g. "llama-3.3-70b-versatile"
    temperature: float = 0.0
    max_tokens: int = 512
    extra_params: dict[str, Any] = Field(default_factory=dict)


class RunRequest(BaseModel):
    source: str                             # file path or HF dataset name
    task_type: str = "open_qa"
    model: ModelSpec
    judge: ModelSpec | None = None
    metrics: list[str] | None = None
    max_samples: int | None = None
    batch_size: int = 10
    concurrency: int = 8
    tags: dict[str, str] = Field(default_factory=dict)


class CompareRequest(BaseModel):
    baseline_run_id: str
    candidate_run_id: str
    threshold: float = 0.02


class RunStatus(BaseModel):
    run_id: str
    status: str                             # "queued" | "running" | "done" | "failed"
    message: str = ""


# ---------------------------------------------------------------------------
# In-memory run state tracker (for async background runs)
# ---------------------------------------------------------------------------

_run_status: dict[str, RunStatus] = {}


# ---------------------------------------------------------------------------
# Background eval task
# ---------------------------------------------------------------------------

async def _execute_run(run_id: str, req: RunRequest) -> None:
    _run_status[run_id] = RunStatus(run_id=run_id, status="running")
    try:
        # Build ModelConfig objects
        provider_cfg = ModelConfig(
            provider=req.model.provider,
            model_id=req.model.model_id,
            temperature=req.model.temperature,
            max_tokens=req.model.max_tokens,
            extra_params=req.model.extra_params,
        )
        judge_cfg = None
        if req.judge:
            judge_cfg = ModelConfig(
                provider=req.judge.provider,
                model_id=req.judge.model_id,
                temperature=req.judge.temperature,
                max_tokens=req.judge.max_tokens,
                extra_params=req.judge.extra_params,
            )

        # Load dataset
        load_kwargs: dict[str, Any] = {}
        if req.max_samples:
            load_kwargs["max_samples"] = req.max_samples

        dataset = loader.load(req.source, task_type=req.task_type, **load_kwargs)

        # Build runner and run
        runner = build_runner(
            provider_cfg,
            judge_config=judge_cfg,
            concurrency=req.concurrency,
        )

        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: runner.run(
                dataset,
                metrics=req.metrics,
                tags=req.tags,
                batch_size=req.batch_size,
                verbose=False,
            ),
        )

        store.save(result)
        _run_status[run_id] = RunStatus(
            run_id=run_id,
            status="done",
            message=f"Completed {result.total_samples} samples",
        )

    except Exception as exc:
        _run_status[run_id] = RunStatus(
            run_id=run_id,
            status="failed",
            message=str(exc),
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/runs", status_code=202)
async def create_run(
    req: RunRequest,
    background_tasks: BackgroundTasks,
) -> RunStatus:
    """
    Kick off an async eval run.
    Returns immediately with run_id + status="queued".
    Poll GET /runs/{run_id} to check progress.
    """
    import uuid
    run_id = str(uuid.uuid4())
    _run_status[run_id] = RunStatus(run_id=run_id, status="queued")
    background_tasks.add_task(_execute_run, run_id, req)
    return _run_status[run_id]


@app.get("/runs/{run_id}/status")
async def get_run_status(run_id: str) -> RunStatus:
    """Poll run execution status."""
    if run_id not in _run_status:
        # Might be an old run already in the DB
        run = store.get_run(run_id)
        if run:
            return RunStatus(run_id=run_id, status="done")
        raise HTTPException(404, f"Run '{run_id}' not found")
    return _run_status[run_id]


@app.get("/runs")
async def list_runs(
    dataset: str | None = Query(None),
    model:   str | None = Query(None),
    task:    str | None = Query(None),
    limit:   int        = Query(50, le=200),
) -> list[dict]:
    """List stored eval runs with optional filters."""
    return store.list_runs(
        dataset_name=dataset,
        model_slug=model,
        task_type=task,
        limit=limit,
    )


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    """Get aggregate scores and metadata for a run."""
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(404, f"Run '{run_id}' not found")
    return run


@app.get("/runs/{run_id}/samples")
async def get_run_samples(
    run_id: str,
    limit: int = Query(100, le=1000),
) -> list[dict]:
    """Get per-sample outputs and scores for a run."""
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(404, f"Run '{run_id}' not found")
    samples = store.get_samples(run_id)
    return samples[:limit]


@app.get("/runs/{run_id}/report", response_class=HTMLResponse)
async def get_run_report(run_id: str) -> HTMLResponse:
    """Download a self-contained HTML report for a run."""
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(404, f"Run '{run_id}' not found")
    samples = store.get_samples(run_id)
    html = generate_html_report(run, samples)
    return HTMLResponse(content=html)


@app.post("/runs/compare")
async def compare_runs(req: CompareRequest) -> dict:
    """
    Regression diff between two runs.
    Returns deltas, regressions, and improvements per metric.
    """
    try:
        report = tracker.compare(
            req.baseline_run_id,
            req.candidate_run_id,
            threshold=req.threshold,
        )
        return report
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@app.get("/runs/leaderboard/{dataset_name}")
async def leaderboard(
    dataset_name: str,
    metric: str = Query("f1"),
    task_type: str | None = Query(None),
    top_n: int = Query(10, le=50),
) -> list[dict]:
    """Model leaderboard ranked by metric on a dataset."""
    return tracker.compare_models(
        dataset_name=dataset_name,
        task_type=task_type,
        metric=metric,
        top_n=top_n,
    )


@app.get("/tasks")
async def list_tasks() -> list[dict]:
    """List all registered task types."""
    return [
        {
            "name": name,
            "description": registry.get_task(name).description,
            "default_metrics": registry.get_task(name).default_metrics,
            "requires_context": registry.get_task(name).requires_context,
        }
        for name in registry.list_tasks()
    ]


@app.get("/metrics")
async def list_metrics() -> list[dict]:
    """List all registered metrics."""
    return [
        {
            "name": name,
            "description": registry.get_metric(name).description,
            "requires_expected": registry.get_metric(name).requires_expected,
            "requires_context": registry.get_metric(name).requires_context,
        }
        for name in registry.list_metrics()
    ]
