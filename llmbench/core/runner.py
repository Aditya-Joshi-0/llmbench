"""
llmbench/core/runner.py
Async batch eval runner — the execution heart of LLMBench.

Flow:
  EvalDataset + RunConfig
       │
       ▼
  [build prompts]
       │
       ▼
  [async batch inference] ──► Provider
       │
       ▼
  [compute metrics]  ──► Metric engine
       │
       ▼
  RunResult  ──► Store
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from llmbench.core.registry import registry
from llmbench.core.schema import (
    EvalDataset,
    ModelConfig,
    RunConfig,
    RunResult,
    SampleResult,
)
from llmbench.providers.base import BaseProvider, get_provider

log = logging.getLogger("llmbench.runner")
console = Console()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class EvalRunner:
    """
    Orchestrates the full eval loop.

    Args:
        provider:       Instantiated LLM provider for inference.
        judge_provider: Optional separate provider for LLM-as-judge metrics.
        concurrency:    Max parallel inference calls.
    """

    def __init__(
        self,
        provider: BaseProvider,
        judge_provider: BaseProvider | None = None,
        concurrency: int = 8,
    ) -> None:
        self.provider       = provider
        self.judge_provider = judge_provider
        self.concurrency    = concurrency
        self._semaphore     = asyncio.Semaphore(concurrency)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        dataset: EvalDataset,
        metrics: list[str] | None = None,
        tags: dict[str, str] | None = None,
        batch_size: int = 10,
        seed: int = 42,
        verbose: bool = True,
    ) -> RunResult:
        """Synchronous entry point — wraps the async runner."""
        try:
            return asyncio.run(
                self._run_async(dataset, metrics, tags, batch_size, seed, verbose)
            )
        except RuntimeError:
            # Already inside a running loop (Jupyter / FastAPI)
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self._run_async(dataset, metrics, tags, batch_size, seed, verbose)
            )

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------

    async def _run_async(
        self,
        dataset: EvalDataset,
        metrics: list[str] | None,
        tags: dict[str, str] | None,
        batch_size: int,
        seed: int,
        verbose: bool,
    ) -> RunResult:
        task_def = registry.get_task(str(dataset.task_type))
        resolved_metrics = metrics or task_def.default_metrics

        run_config = RunConfig(
            dataset_name=dataset.name,
            task_type=str(dataset.task_type),
            llm_config=self.provider.config,
            metrics=resolved_metrics,
            judge_model=self.judge_provider.config if self.judge_provider else None,
            batch_size=batch_size,
            seed=seed,
            tags=tags or {},
        )

        if verbose:
            console.rule(f"[bold]LLMBench Run — {run_config.run_id[:8]}[/]")
            console.print(f"  Model   : [cyan]{self.provider.config.slug}[/]")
            console.print(f"  Dataset : [cyan]{dataset.name}[/] ({len(dataset)} samples)")
            console.print(f"  Task    : [cyan]{dataset.task_type}[/]")
            console.print(f"  Metrics : [cyan]{', '.join(resolved_metrics)}[/]")
            console.print()

        # --- Inference phase -------------------------------------------
        sample_results = await self._infer_all(dataset, task_def, verbose)

        # Attach expected_output and context onto SampleResult for judge metrics
        sample_map = {s.id: s for s in dataset.samples}
        for sr in sample_results:
            orig = sample_map.get(sr.sample_id)
            if orig:
                sr.expected_output = orig.expected_output
                sr.context = orig.context
                sr.sample_input = orig.input       # type: ignore[attr-defined]

        # --- Metric phase ----------------------------------------------
        if verbose:
            console.print("\n[bold]Computing metrics…[/]")

        aggregate: dict[str, float] = {}
        for metric_name in resolved_metrics:
            try:
                plugin = registry.get_metric(metric_name)
                scores = plugin.fn(
                    sample_results,
                    judge_provider=self.judge_provider,
                )
                aggregate.update(scores)
                # Distribute per-sample scores
                for sr in sample_results:
                    for k, v in scores.items():
                        if k not in sr.scores:
                            sr.scores[k] = v
            except Exception as exc:
                log.warning("Metric '%s' failed: %s", metric_name, exc)
                aggregate[metric_name] = float("nan")

        # --- Aggregate stats -------------------------------------------
        failed = sum(1 for r in sample_results if r.error)
        total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in sample_results)
        total_latency = sum(r.latency_ms for r in sample_results)

        result = RunResult(
            run_id=run_config.run_id,
            config=run_config,
            sample_results=sample_results,
            aggregate_scores=aggregate,
            total_samples=len(sample_results),
            failed_samples=failed,
            total_tokens=total_tokens,
            total_latency_ms=total_latency,
        )

        if verbose:
            self._print_summary(result)

        return result

    async def _infer_all(self, dataset, task_def, verbose: bool) -> list[SampleResult]:
        """Run inference over all samples with concurrency control."""
        results: list[SampleResult] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            disable=not verbose,
        ) as progress:
            task = progress.add_task("Inferring…", total=len(dataset))

            async def _infer_one(sample) -> SampleResult:
                async with self._semaphore:
                    prompt = task_def.build_prompt(sample)
                    inference = await self.provider.async_infer(
                        prompt, system=task_def.system_prompt
                    )
                    progress.advance(task)
                    return SampleResult(
                        sample_id=sample.id,
                        model_slug=self.provider.config.slug,
                        generated_output=inference.text,
                        prompt_tokens=inference.prompt_tokens,
                        completion_tokens=inference.completion_tokens,
                        latency_ms=inference.latency_ms,
                        error=inference.error,
                    )

            tasks = [_infer_one(s) for s in dataset]
            results = list(await asyncio.gather(*tasks))

        return results

    @staticmethod
    def _print_summary(result: RunResult) -> None:
        console.rule("[bold green]Run complete[/]")
        summary = result.summary()
        console.print(f"  Run ID  : [dim]{summary['run_id']}[/]")
        console.print(f"  Samples : {summary['n_samples']}  "
                      f"(failed: [red]{summary['n_failed']}[/])")
        console.print(f"  Tokens  : {result.total_tokens:,}")
        console.print(f"  Latency : {result.total_latency_ms / 1000:.1f}s total")
        console.print("\n  [bold]Scores:[/]")
        for metric, score in summary["scores"].items():
            bar_len = int(score * 20) if not isinstance(score, float) or score == score else 0
            bar = "█" * bar_len + "░" * (20 - bar_len)
            console.print(f"    {metric:<25} {score:.4f}  [{bar}]")
        console.print()


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_runner(
    provider_config: ModelConfig,
    judge_config: ModelConfig | None = None,
    concurrency: int = 8,
    provider_kwargs: dict[str, Any] | None = None,
    judge_kwargs:    dict[str, Any] | None = None,
) -> EvalRunner:
    """
    Build an EvalRunner from configs.

    Example:
        runner = build_runner(
            ModelConfig(provider="groq", model_id="llama-3.3-70b-versatile"),
            judge_config=ModelConfig(provider="groq", model_id="llama-3.3-70b-versatile"),
        )
        result = runner.run(dataset)
    """
    provider = get_provider(provider_config, **(provider_kwargs or {}))
    judge    = get_provider(judge_config,    **(judge_kwargs    or {})) if judge_config else None
    return EvalRunner(provider=provider, judge_provider=judge, concurrency=concurrency)
