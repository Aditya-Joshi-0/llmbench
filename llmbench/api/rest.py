"""
llmbench/api/rest.py
FastAPI REST API — single-turn and multi-turn evals, results, regression, reports.

Single-turn:
  POST /runs                        Trigger async eval run
  GET  /runs                        List runs (filtered)
  GET  /runs/{run_id}               Aggregate scores + metadata
  GET  /runs/{run_id}/samples       Per-sample breakdown
  GET  /runs/{run_id}/report        Self-contained HTML report
  GET  /runs/{run_id}/status        Poll async run status

Multi-turn:                                                        ← NEW
  POST /conversations/runs          Trigger async conversation eval
  GET  /conversations/runs/{run_id}/turns          All turn records
  GET  /conversations/runs/{run_id}/turns/{conv_id} One conversation's turns

Shared:
  POST /runs/compare                Regression diff
  GET  /runs/leaderboard/{dataset}  Model leaderboard
  GET  /tasks                       Registered tasks
  GET  /metrics                     Registered metrics
  GET  /health

Run: uvicorn llmbench.api.rest:app --reload --port 8080
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import llmbench.metrics  # noqa: F401 — trigger registration
from llmbench.core.registry import registry
from llmbench.core.runner import build_runner
from llmbench.core.runner_multiturn import build_conversation_runner
from llmbench.core.schema import ModelConfig
from llmbench.loaders import DatasetLoader
from llmbench.loaders.conversations import (
    load_conversations_json,
    load_conversations_hf,
)
from llmbench.store.db import store, tracker
from llmbench.api.reporter import generate_html_report

app = FastAPI(
    title="LLMBench API",
    description="Production-grade LLM evaluation harness",
    version="0.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_loader = DatasetLoader()

# ---------------------------------------------------------------------------
# Shared request models
# ---------------------------------------------------------------------------

class ModelSpec(BaseModel):
    provider:     str
    model_id:     str
    temperature:  float = 0.0
    max_tokens:   int   = 512
    extra_params: dict[str, Any] = Field(default_factory=dict)


class RunRequest(BaseModel):
    source:      str
    task_type:   str = "open_qa"
    model:       ModelSpec
    judge:       ModelSpec | None = None
    metrics:     list[str] | None = None
    max_samples: int | None = None
    batch_size:  int = 10
    concurrency: int = 8
    tags:        dict[str, str] = Field(default_factory=dict)


class ConversationRunRequest(BaseModel):               # ← NEW
    source:      str                    # local path or HF repo
    model:       ModelSpec
    judge:       ModelSpec | None = None
    metrics:     list[str] | None = None
    max_samples: int | None = None
    concurrency: int = 4
    tags:        dict[str, str] = Field(default_factory=dict)


class CompareRequest(BaseModel):
    baseline_run_id:  str
    candidate_run_id: str
    threshold: float = 0.02


class RunStatus(BaseModel):
    run_id:  str
    status:  str        # queued | running | done | failed
    message: str = ""


# ---------------------------------------------------------------------------
# In-memory status tracker for async background runs
# ---------------------------------------------------------------------------

_run_status: dict[str, RunStatus] = {}


def _model_spec_to_config(spec: ModelSpec) -> ModelConfig:
    return ModelConfig(
        provider=spec.provider,
        model_id=spec.model_id,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        extra_params=spec.extra_params,
    )


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------

async def _execute_single_run(run_id: str, req: RunRequest) -> None:
    _run_status[run_id] = RunStatus(run_id=run_id, status="running")
    try:
        provider_cfg = _model_spec_to_config(req.model)
        judge_cfg    = _model_spec_to_config(req.judge) if req.judge else None

        load_kw: dict[str, Any] = {}
        if req.max_samples:
            load_kw["max_samples"] = req.max_samples

        dataset = _loader.load(req.source, task_type=req.task_type, **load_kw)
        runner  = build_runner(provider_cfg, judge_config=judge_cfg,
                               concurrency=req.concurrency)

        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: runner.run(dataset, metrics=req.metrics,
                               tags=req.tags, batch_size=req.batch_size,
                               verbose=False),
        )
        store.save(result)
        _run_status[run_id] = RunStatus(
            run_id=run_id, status="done",
            message=f"Completed {result.total_samples} samples",
        )
    except Exception as exc:
        _run_status[run_id] = RunStatus(run_id=run_id, status="failed",
                                        message=str(exc))


async def _execute_conversation_run(run_id: str, req: ConversationRunRequest) -> None:
    _run_status[run_id] = RunStatus(run_id=run_id, status="running")
    try:
        from pathlib import Path
        provider_cfg = _model_spec_to_config(req.model)
        judge_cfg    = _model_spec_to_config(req.judge) if req.judge else None

        path = Path(req.source)
        if path.exists():
            dataset = load_conversations_json(req.source,
                                              max_samples=req.max_samples)
        else:
            dataset = load_conversations_hf(req.source,
                                            max_samples=req.max_samples)

        runner = build_conversation_runner(provider_cfg, judge_config=judge_cfg,
                                           concurrency=req.concurrency)
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: runner.run(dataset, metrics=req.metrics,
                               tags=req.tags, verbose=False),
        )
        store.save(result)
        _run_status[run_id] = RunStatus(
            run_id=run_id, status="done",
            message=f"Completed {result.total_samples} conversations",
        )
    except Exception as exc:
        _run_status[run_id] = RunStatus(run_id=run_id, status="failed",
                                        message=str(exc))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.2.0"}


# ---------------------------------------------------------------------------
# Single-turn runs
# ---------------------------------------------------------------------------

@app.post("/runs", status_code=202)
async def create_run(req: RunRequest, bg: BackgroundTasks) -> RunStatus:
    """Kick off a single-turn eval. Returns immediately; poll /runs/{id}/status."""
    run_id = str(uuid.uuid4())
    _run_status[run_id] = RunStatus(run_id=run_id, status="queued")
    bg.add_task(_execute_single_run, run_id, req)
    return _run_status[run_id]


@app.get("/runs/{run_id}/status")
async def get_run_status(run_id: str) -> RunStatus:
    if run_id in _run_status:
        return _run_status[run_id]
    if store.get_run(run_id):
        return RunStatus(run_id=run_id, status="done")
    raise HTTPException(404, f"Run '{run_id}' not found")


@app.get("/runs")
async def list_runs(
    dataset: str | None = Query(None),
    model:   str | None = Query(None),
    task:    str | None = Query(None),
    limit:   int        = Query(50, le=200),
) -> list[dict]:
    return store.list_runs(dataset_name=dataset, model_slug=model,
                           task_type=task, limit=limit)


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(404, f"Run '{run_id}' not found")
    return run


@app.get("/runs/{run_id}/samples")
async def get_samples(
    run_id: str,
    limit:  int = Query(100, le=1000),
) -> list[dict]:
    if not store.get_run(run_id):
        raise HTTPException(404, f"Run '{run_id}' not found")
    return store.get_samples(run_id)[:limit]


@app.get("/runs/{run_id}/report", response_class=HTMLResponse)
async def get_report(run_id: str) -> HTMLResponse:
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(404, f"Run '{run_id}' not found")
    samples = store.get_samples(run_id)
    return HTMLResponse(content=generate_html_report(run, samples))


# ---------------------------------------------------------------------------
# Multi-turn conversation runs                                      ← NEW
# ---------------------------------------------------------------------------

@app.post("/conversations/runs", status_code=202)
async def create_conversation_run(
    req: ConversationRunRequest, bg: BackgroundTasks
) -> RunStatus:
    """
    Kick off a multi-turn conversation eval.
    Returns immediately; poll GET /runs/{id}/status.

    Example body:
        {
          "source": "tasks/sample_conversations.json",
          "model": {"provider": "groq", "model_id": "llama-3.3-70b-versatile"},
          "metrics": ["turn_exact_match", "turn_f1", "turn_rouge_l"]
        }
    """
    run_id = str(uuid.uuid4())
    _run_status[run_id] = RunStatus(run_id=run_id, status="queued")
    bg.add_task(_execute_conversation_run, run_id, req)
    return _run_status[run_id]


@app.get("/conversations/runs/{run_id}/turns")
async def get_all_turns(
    run_id: str,
    limit:  int = Query(500, le=5000),
) -> list[dict]:
    """All turn records across every conversation in a multi-turn run."""
    if not store.get_run(run_id):
        raise HTTPException(404, f"Run '{run_id}' not found")
    return store.get_turn_results(run_id)[:limit]


@app.get("/conversations/runs/{run_id}/turns/{conversation_id}")
async def get_conversation_turns(run_id: str, conversation_id: str) -> list[dict]:
    """Turn-by-turn breakdown for a single conversation."""
    if not store.get_run(run_id):
        raise HTTPException(404, f"Run '{run_id}' not found")
    turns = store.get_turn_results(run_id, conversation_id=conversation_id)
    if not turns:
        raise HTTPException(404, f"Conversation '{conversation_id}' not found in run")
    return turns


# ---------------------------------------------------------------------------
# Shared: compare, leaderboard
# ---------------------------------------------------------------------------

@app.post("/runs/compare")
async def compare_runs(req: CompareRequest) -> dict:
    try:
        return tracker.compare(req.baseline_run_id, req.candidate_run_id,
                               threshold=req.threshold)
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@app.get("/runs/leaderboard/{dataset_name}")
async def leaderboard(
    dataset_name: str,
    metric:    str        = Query("f1"),
    task_type: str | None = Query(None),
    top_n:     int        = Query(10, le=50),
) -> list[dict]:
    return tracker.compare_models(dataset_name=dataset_name, task_type=task_type,
                                  metric=metric, top_n=top_n)


# ---------------------------------------------------------------------------
# Registry info
# ---------------------------------------------------------------------------

@app.get("/tasks")
async def list_tasks() -> list[dict]:
    return [
        {
            "name":             name,
            "description":      registry.get_task(name).description,
            "default_metrics":  registry.get_task(name).default_metrics,
            "requires_context": registry.get_task(name).requires_context,
            "is_multi_turn":    registry.get_task(name).is_multi_turn,
        }
        for name in registry.list_tasks()
    ]


@app.get("/metrics")
async def list_metrics() -> list[dict]:
    return [
        {
            "name":             name,
            "description":      registry.get_metric(name).description,
            "requires_expected":registry.get_metric(name).requires_expected,
            "requires_context": registry.get_metric(name).requires_context,
        }
        for name in registry.list_metrics()
    ]
