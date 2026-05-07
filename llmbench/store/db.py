"""
llmbench/store/db.py
SQLAlchemy-based results store + regression tracker.
Default backend: SQLite (zero setup). Swap to Postgres via DATABASE_URL env var.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text,
    create_engine, desc, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from llmbench.core.schema import RunResult, SampleResult


# ---------------------------------------------------------------------------
# Engine setup
# ---------------------------------------------------------------------------

_DEFAULT_DB = "LLMBench_results.db"
_DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_DEFAULT_DB}")

engine = create_engine(
    _DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in _DATABASE_URL else {},
    echo=False,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase): ...


class RunRecord(Base):
    __tablename__ = "runs"

    run_id          = Column(String, primary_key=True)
    dataset_name    = Column(String, nullable=False, index=True)
    task_type       = Column(String, nullable=False, index=True)
    model_slug      = Column(String, nullable=False, index=True)
    metrics_json    = Column(Text)          # JSON: {metric: score}
    config_json     = Column(Text)          # Full RunConfig as JSON
    total_samples   = Column(Integer, default=0)
    failed_samples  = Column(Integer, default=0)
    total_tokens    = Column(Integer, default=0)
    total_latency_ms = Column(Float, default=0.0)
    tags_json       = Column(Text, default="{}")
    created_at      = Column(DateTime, default=datetime.utcnow)


class SampleRecord(Base):
    __tablename__ = "samples"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    run_id          = Column(String, nullable=False, index=True)
    sample_id       = Column(String, nullable=False)
    model_slug      = Column(String)
    generated_output = Column(Text)
    expected_output = Column(Text)
    scores_json     = Column(Text)          # JSON: {metric: score}
    judge_reasoning = Column(Text)
    prompt_tokens   = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    latency_ms      = Column(Float, default=0.0)
    error           = Column(Text)


Base.metadata.create_all(engine)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class ResultsStore:
    """Persist and query eval run results."""

    def save(self, result: RunResult) -> None:
        with SessionLocal() as session:
            run = RunRecord(
                run_id=result.run_id,
                dataset_name=result.config.dataset_name,
                task_type=result.config.task_type,
                model_slug=result.config.llm_config.slug,
                metrics_json=json.dumps(result.aggregate_scores),
                config_json=result.config.model_dump_json(),
                total_samples=result.total_samples,
                failed_samples=result.failed_samples,
                total_tokens=result.total_tokens,
                total_latency_ms=result.total_latency_ms,
                tags_json=json.dumps(result.config.tags),
                created_at=result.completed_at,
            )
            session.add(run)

            for sr in result.sample_results:
                session.add(SampleRecord(
                    run_id=result.run_id,
                    sample_id=sr.sample_id,
                    model_slug=sr.model_slug,
                    generated_output=sr.generated_output,
                    expected_output=getattr(sr, "expected_output", None),
                    scores_json=json.dumps(sr.scores),
                    judge_reasoning=sr.judge_reasoning,
                    prompt_tokens=sr.prompt_tokens,
                    completion_tokens=sr.completion_tokens,
                    latency_ms=sr.latency_ms,
                    error=sr.error,
                ))
            session.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with SessionLocal() as session:
            rec = session.get(RunRecord, run_id)
            if rec is None:
                return None
            return self._run_to_dict(rec)

    def list_runs(
        self,
        dataset_name: str | None = None,
        model_slug: str | None = None,
        task_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        with SessionLocal() as session:
            q = session.query(RunRecord)
            if dataset_name: q = q.filter(RunRecord.dataset_name == dataset_name)
            if model_slug:   q = q.filter(RunRecord.model_slug   == model_slug)
            if task_type:    q = q.filter(RunRecord.task_type    == task_type)
            rows = q.order_by(desc(RunRecord.created_at)).limit(limit).all()
            return [self._run_to_dict(r) for r in rows]

    def get_samples(self, run_id: str) -> list[dict[str, Any]]:
        with SessionLocal() as session:
            rows = (session.query(SampleRecord)
                    .filter(SampleRecord.run_id == run_id)
                    .all())
            return [self._sample_to_dict(r) for r in rows]

    @staticmethod
    def _run_to_dict(rec: RunRecord) -> dict[str, Any]:
        return {
            "run_id":           rec.run_id,
            "dataset_name":     rec.dataset_name,
            "task_type":        rec.task_type,
            "model_slug":       rec.model_slug,
            "scores":           json.loads(rec.metrics_json or "{}"),
            "total_samples":    rec.total_samples,
            "failed_samples":   rec.failed_samples,
            "total_tokens":     rec.total_tokens,
            "total_latency_ms": rec.total_latency_ms,
            "tags":             json.loads(rec.tags_json or "{}"),
            "created_at":       rec.created_at.isoformat() if rec.created_at else None,
        }

    @staticmethod
    def _sample_to_dict(rec: SampleRecord) -> dict[str, Any]:
        return {
            "sample_id":        rec.sample_id,
            "generated_output": rec.generated_output,
            "expected_output":  rec.expected_output,
            "scores":           json.loads(rec.scores_json or "{}"),
            "judge_reasoning":  rec.judge_reasoning,
            "latency_ms":       rec.latency_ms,
            "error":            rec.error,
        }


# ---------------------------------------------------------------------------
# Regression tracker
# ---------------------------------------------------------------------------

class RegressionTracker:
    """
    Compare two runs and flag metrics that regressed beyond a threshold.

    Usage:
        tracker = RegressionTracker(store)
        report = tracker.compare(run_id_a, run_id_b, threshold=0.02)
    """

    def __init__(self, store: ResultsStore) -> None:
        self.store = store

    def compare(
        self,
        baseline_run_id: str,
        candidate_run_id: str,
        threshold: float = 0.02,    # absolute score drop that counts as a regression
    ) -> dict[str, Any]:
        baseline  = self.store.get_run(baseline_run_id)
        candidate = self.store.get_run(candidate_run_id)

        if baseline is None:
            raise ValueError(f"Run '{baseline_run_id}' not found in store")
        if candidate is None:
            raise ValueError(f"Run '{candidate_run_id}' not found in store")

        b_scores = baseline["scores"]
        c_scores = candidate["scores"]
        all_metrics = set(b_scores) | set(c_scores)

        diffs:       dict[str, float] = {}
        regressions: dict[str, float] = {}
        improvements: dict[str, float] = {}

        for metric in all_metrics:
            b = b_scores.get(metric)
            c = c_scores.get(metric)
            if b is None or c is None:
                continue
            if b != b or c != c:      # NaN guard
                continue

            # ECE: lower is better — flip sign for consistent comparison
            higher_is_better = metric != "ece"
            delta = (c - b) if higher_is_better else (b - c)
            diffs[metric] = c - b     # raw delta always

            if delta < -threshold:
                regressions[metric]  = delta
            elif delta > threshold:
                improvements[metric] = delta

        return {
            "baseline":     {"run_id": baseline_run_id,  "model": baseline["model_slug"],
                             "scores": b_scores},
            "candidate":    {"run_id": candidate_run_id, "model": candidate["model_slug"],
                             "scores": c_scores},
            "deltas":        diffs,
            "regressions":   regressions,      # metric → score change (negative)
            "improvements":  improvements,     # metric → score change (positive)
            "has_regression": len(regressions) > 0,
            "threshold_used": threshold,
        }

    def compare_models(
        self,
        dataset_name: str,
        task_type: str | None = None,
        metric: str = "f1",
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Leaderboard: rank all runs on a dataset by a single metric.
        """
        runs = self.store.list_runs(dataset_name=dataset_name, task_type=task_type)
        ranked = sorted(
            [r for r in runs if metric in r["scores"]],
            key=lambda r: r["scores"].get(metric, 0),
            reverse=(metric != "ece"),
        )
        return ranked[:top_n]


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

store   = ResultsStore()
tracker = RegressionTracker(store)
