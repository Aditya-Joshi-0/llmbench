"""
llmbench/api/cli.py
Typer CLI — primary user interface for LLMBench.

Commands:
  llmbench run          Single-turn eval
  llmbench run-conv     Multi-turn conversation eval       ← NEW
  llmbench list         List stored runs
  llmbench compare      Regression check between two runs
  llmbench show         Inspect a single run
  llmbench metrics      List registered metrics
  llmbench tasks        List registered task types
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app     = typer.Typer(name="llmbench", help="Production-grade LLM evaluation harness",
                      add_completion=False)
console = Console()


# ---------------------------------------------------------------------------
# Shared: parse "provider/model-id" slug → ModelConfig
# ---------------------------------------------------------------------------

def _parse_model(slug: str):
    from llmbench.core.schema import ModelConfig
    parts = slug.split("/", 1)
    if len(parts) != 2:
        raise typer.BadParameter(
            f"Model must be 'provider/model_id' (e.g. groq/llama-3.3-70b-versatile), got: {slug}"
        )
    return ModelConfig(provider=parts[0], model_id=parts[1])


# ---------------------------------------------------------------------------
# run  — single-turn
# ---------------------------------------------------------------------------

@app.command()
def run(
    source: str = typer.Argument(..., help="Dataset path (.json/.csv) or HF repo name"),
    task:   str = typer.Option(..., "--task",    "-t", help="Task type (e.g. open_qa)"),
    model:  str = typer.Option(..., "--model",   "-m", help="Provider/model slug"),
    judge:  Optional[str] = typer.Option(None,   "--judge", "-j", help="Judge model slug"),
    metrics: Optional[str] = typer.Option(None,  "--metrics",
                                          help="Comma-separated metric names (default: task defaults)"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", "-n"),
    batch_size:  int  = typer.Option(10,  "--batch-size"),
    concurrency: int  = typer.Option(8,   "--concurrency"),
    tag: Optional[list[str]] = typer.Option(None, "--tag",
                                             help="Tags as key=value (repeatable)"),
    save:   bool = typer.Option(True, "--save/--no-save"),
    output: Optional[Path] = typer.Option(None, "--output", "-o",
                                           help="Write JSON summary to file"),
) -> None:
    """Run a single-turn evaluation."""
    import llmbench.metrics
    from llmbench.core.runner import build_runner
    from llmbench.loaders import DatasetLoader
    from llmbench.store.db import store as result_store

    provider_cfg = _parse_model(model)
    judge_cfg    = _parse_model(judge) if judge else None

    load_kwargs: dict = {}
    if max_samples:
        load_kwargs["max_samples"] = max_samples

    with console.status(f"Loading [cyan]{source}[/]…"):
        dataset = DatasetLoader().load(source, task_type=task, **load_kwargs)
    console.print(f"Loaded [bold]{len(dataset)}[/] samples")

    tags = {}
    for t in (tag or []):
        if "=" not in t:
            raise typer.BadParameter(f"Tag must be key=value, got: {t}")
        k, v = t.split("=", 1)
        tags[k] = v

    metric_list = [m.strip() for m in metrics.split(",")] if metrics else None

    runner = build_runner(provider_cfg, judge_config=judge_cfg, concurrency=concurrency)
    result = runner.run(dataset, metrics=metric_list, tags=tags,
                        batch_size=batch_size, verbose=True)

    if save:
        result_store.save(result)
        console.print(f"[dim]Saved → run_id: {result.run_id}[/]")

    if output:
        output.write_text(json.dumps(result.summary(), indent=2))
        console.print(f"Report written to [cyan]{output}[/]")


# ---------------------------------------------------------------------------
# run-conv  — multi-turn conversation eval                          ← NEW
# ---------------------------------------------------------------------------

@app.command(name="run-conv")
def run_conv(
    source: str = typer.Argument(...,
        help="Conversation dataset path (.json/.jsonl) or HF repo name"),
    model:  str = typer.Option(..., "--model",  "-m", help="Provider/model slug"),
    judge:  Optional[str] = typer.Option(None,  "--judge", "-j",
        help="Judge model slug for coherence / context-retention metrics"),
    metrics: Optional[str] = typer.Option(None, "--metrics",
        help="Comma-separated metric names (default: turn_exact_match,turn_f1,turn_rouge_l)"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", "-n",
        help="Cap number of conversations"),
    concurrency: int  = typer.Option(4,  "--concurrency",
        help="Max parallel conversations (keep low — each is multi-step)"),
    tag: Optional[list[str]] = typer.Option(None, "--tag",
        help="Tags as key=value (repeatable)"),
    save:   bool = typer.Option(True, "--save/--no-save"),
    output: Optional[Path] = typer.Option(None, "--output", "-o",
        help="Write JSON summary to file"),
    format: str = typer.Option("auto", "--format", "-f",
        help="Conversation format: auto | native | sharegpt | openai"),
) -> None:
    """
    Run a multi-turn conversation evaluation.

    The model is replayed through each conversation turn-by-turn
    with a growing history (mimicking a real chat session).
    Each assistant response is scored against the expected response.

    Example:
        llmbench run-conv tasks/sample_conversations.json \\
            --model groq/llama-3.3-70b-versatile \\
            --judge groq/openai/gpt-oss-120b \\
            --metrics turn_exact_match,turn_f1,conversation_coherence
    """
    import llmbench.metrics
    from llmbench.core.runner_multiturn import build_conversation_runner
    from llmbench.loaders.conversations import (
        load_conversations_json,
        load_conversations_hf,
    )
    from llmbench.store.db import store as result_store

    provider_cfg = _parse_model(model)
    judge_cfg    = _parse_model(judge) if judge else None

    # Load dataset
    path = Path(source)
    with console.status(f"Loading conversations from [cyan]{source}[/]…"):
        if path.exists():
            dataset = load_conversations_json(
                path,
                max_samples=max_samples,
            )
        else:
            dataset = load_conversations_hf(
                source,
                max_samples=max_samples,
            )

    console.print(
        f"Loaded [bold]{len(dataset)}[/] conversations"
    )

    # Validate: warn if no judge supplied for judge metrics
    metric_list = [m.strip() for m in metrics.split(",")] if metrics else None
    judge_metrics = {"conversation_coherence", "context_retention"}
    if metric_list:
        needs_judge = judge_metrics & set(metric_list)
        if needs_judge and not judge_cfg:
            console.print(
                f"[yellow]Warning:[/] metrics {needs_judge} require --judge. "
                f"They will be skipped."
            )

    tags = {}
    for t in (tag or []):
        if "=" not in t:
            raise typer.BadParameter(f"Tag must be key=value, got: {t}")
        k, v = t.split("=", 1)
        tags[k] = v

    runner = build_conversation_runner(
        provider_cfg,
        judge_config=judge_cfg,
        concurrency=concurrency,
    )
    result = runner.run(dataset, metrics=metric_list, tags=tags, verbose=True)

    if save:
        result_store.save(result)
        console.print(f"[dim]Saved → run_id: {result.run_id}[/]")

    if output:
        output.write_text(json.dumps(result.summary(), indent=2))
        console.print(f"Report written to [cyan]{output}[/]")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@app.command(name="list")
def list_runs(
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d"),
    model:   Optional[str] = typer.Option(None, "--model",   "-m"),
    task:    Optional[str] = typer.Option(None, "--task",    "-t"),
    limit:   int = typer.Option(20, "--limit", "-n"),
) -> None:
    """List stored eval runs."""
    from llmbench.store.db import store as result_store

    runs = result_store.list_runs(
        dataset_name=dataset, model_slug=model, task_type=task, limit=limit
    )
    if not runs:
        console.print("[yellow]No runs found.[/]")
        raise typer.Exit()

    all_metrics: list[str] = []
    for r in runs:
        for m in r["scores"]:
            if m not in all_metrics:
                all_metrics.append(m)

    table = Table(title="Stored Runs", show_header=True, header_style="bold")
    table.add_column("Run ID",   style="dim", width=10)
    table.add_column("Model",    style="cyan")
    table.add_column("Dataset")
    table.add_column("Task")
    table.add_column("Samples",  justify="right")
    table.add_column("Created",  style="dim")
    for m in all_metrics:
        table.add_column(m, justify="right")

    for r in runs:
        score_cells = [
            f"{r['scores'].get(m, '—'):.3f}"
            if isinstance(r["scores"].get(m), float) else "—"
            for m in all_metrics
        ]
        table.add_row(
            r["run_id"][:8],
            r["model_slug"],
            r["dataset_name"],
            r["task_type"],
            str(r["total_samples"]),
            (r["created_at"] or "")[:16],
            *score_cells,
        )
    console.print(table)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

@app.command()
def compare(
    baseline:  str   = typer.Argument(..., help="Baseline run_id"),
    candidate: str   = typer.Argument(..., help="Candidate run_id"),
    threshold: float = typer.Option(0.02, "--threshold", "-T"),
) -> None:
    """Regression diff between two runs. Exits with code 1 if regressions found."""
    from llmbench.store.db import tracker

    report = tracker.compare(baseline, candidate, threshold=threshold)

    console.print(f"\n[bold]Baseline :[/] {report['baseline']['model']}  ({baseline[:8]})")
    console.print(f"[bold]Candidate:[/] {report['candidate']['model']}  ({candidate[:8]})")
    console.print(f"[bold]Threshold:[/] {threshold}\n")

    table = Table(title="Metric Comparison", show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Baseline",  justify="right")
    table.add_column("Candidate", justify="right")
    table.add_column("Δ",         justify="right")
    table.add_column("Status")

    b_scores = report["baseline"]["scores"]
    c_scores = report["candidate"]["scores"]

    for metric in set(b_scores) | set(c_scores):
        b     = b_scores.get(metric, float("nan"))
        c     = c_scores.get(metric, float("nan"))
        delta = report["deltas"].get(metric, float("nan"))

        if metric in report["regressions"]:
            status = "[red]▼ REGRESSION[/]"
        elif metric in report["improvements"]:
            status = "[green]▲ improved[/]"
        else:
            status = "[dim]≈ stable[/]"

        table.add_row(
            metric,
            f"{b:.4f}" if b == b else "—",
            f"{c:.4f}" if c == c else "—",
            f"{delta:+.4f}" if delta == delta else "—",
            status,
        )

    console.print(table)

    if report["has_regression"]:
        console.print(f"\n[bold red]⚠ {len(report['regressions'])} regression(s) detected[/]")
        raise typer.Exit(code=1)
    else:
        console.print("\n[bold green]✓ No regressions detected[/]")


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------

@app.command()
def show(
    run_id:  str  = typer.Argument(...),
    samples: bool = typer.Option(False, "--samples", "-s"),
    top:     int  = typer.Option(10,    "--top"),
) -> None:
    """Show aggregate scores for a run. For multi-turn runs shows per-turn breakdown."""
    from llmbench.store.db import store as result_store

    run = result_store.get_run(run_id)
    if not run:
        console.print(f"[red]Run '{run_id}' not found.[/]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Run:[/]      {run['run_id']}")
    console.print(f"[bold]Model:[/]    {run['model_slug']}")
    console.print(f"[bold]Dataset:[/]  {run['dataset_name']}")
    console.print(f"[bold]Task:[/]     {run['task_type']}")
    console.print(f"[bold]Samples:[/]  {run['total_samples']} ({run['failed_samples']} failed)")
    console.print(f"[bold]Tokens:[/]   {run['total_tokens']:,}")

    score_table = Table(title="Aggregate Scores")
    score_table.add_column("Metric")
    score_table.add_column("Score", justify="right")
    for metric, score in run["scores"].items():
        score_table.add_row(
            metric,
            f"{score:.4f}" if isinstance(score, float) else str(score)
        )
    console.print(score_table)

    if samples:
        sample_rows = result_store.get_samples(run_id)[:top]
        is_multiturn = run["task_type"] == "multi_turn"

        if is_multiturn:
            # ── Multi-turn: show per-conversation score breakdown ────────
            st = Table(title=f"Conversation Results (top {top})")
            st.add_column("Conv ID",    width=10, style="dim")
            st.add_column("Turns output (truncated)", overflow="fold", max_width=60)
            st.add_column("Turn scores (avg)")
            st.add_column("Confidence")
            st.add_column("Err")
            for s in sample_rows:
                scores_str = json.dumps(
                    {k: round(v, 3) for k, v in s["scores"].items()}, indent=None
                ) if s["scores"] else "—"
                conf = s.get("confidence")
                st.add_row(
                    (s["sample_id"] or "")[:10],
                    (s["generated_output"] or "")[:120],
                    scores_str,
                    f"{conf:.3f}" if conf is not None else "—",
                    s["error"] or "",
                )
        else:
            # ── Single-turn: standard sample table ───────────────────────
            st = Table(title=f"Sample Results (top {top})")
            st.add_column("ID",         width=8, style="dim")
            st.add_column("Generated",  overflow="fold", max_width=50)
            st.add_column("Expected",   overflow="fold", max_width=30)
            st.add_column("Scores")
            st.add_column("Conf")
            st.add_column("Err")
            for s in sample_rows:
                conf = s.get("confidence")
                st.add_row(
                    (s["sample_id"] or "")[:8],
                    (s["generated_output"] or "")[:100],
                    (s["expected_output"] or "—")[:60],
                    json.dumps({k: round(v, 3) for k, v in s["scores"].items()}),
                    f"{conf:.3f}" if conf is not None else "—",
                    s["error"] or "",
                )

        console.print(st)


# ---------------------------------------------------------------------------
# metrics / tasks
# ---------------------------------------------------------------------------

@app.command(name="metrics")
def list_metrics() -> None:
    """List all registered metrics."""
    import llmbench.metrics
    from llmbench.core.registry import registry

    table = Table(title="Registered Metrics")
    table.add_column("Name",       style="cyan")
    table.add_column("Family")
    table.add_column("Description")
    table.add_column("Ref?",  justify="center")
    table.add_column("Ctx?",  justify="center")

    families = {
        "exact_match": "lexical",   "f1": "lexical",
        "rouge_1": "lexical",       "rouge_2": "lexical",
        "rouge_l": "lexical",       "bleu": "lexical",
        "bertscore": "semantic",    "cosine_similarity": "semantic",
        "llm_faithfulness": "judge","llm_relevance": "judge",
        "llm_coherence": "judge",   "llm_code_quality": "judge",
        "conversation_coherence": "judge", "context_retention": "judge",
        "ece": "calibration",
        "turn_exact_match": "multi-turn", "turn_f1": "multi-turn",
        "turn_rouge_l": "multi-turn",
    }

    for name in registry.list_metrics():
        p = registry.get_metric(name)
        table.add_row(
            name,
            families.get(name, "custom"),
            p.description,
            "✓" if p.requires_expected else "✗",
            "✓" if p.requires_context  else "✗",
        )
    console.print(table)


@app.command(name="tasks")
def list_tasks() -> None:
    """List all registered task types."""
    from llmbench.core.registry import registry

    table = Table(title="Registered Tasks")
    table.add_column("Name",           style="cyan")
    table.add_column("Multi-turn?",    justify="center")
    table.add_column("Description")
    table.add_column("Default metrics")
    for name in registry.list_tasks():
        t = registry.get_task(name)
        table.add_row(
            name,
            "✓" if t.is_multi_turn else "✗",
            t.description,
            ", ".join(t.default_metrics),
        )
    console.print(table)


if __name__ == "__main__":
    app()
