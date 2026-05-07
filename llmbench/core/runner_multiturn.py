"""
llmbench/core/runner_multiturn.py
Async multi-turn conversation eval runner.

Strategy:
  For each ConversationSample the runner replays the conversation turn-by-turn:

    Turn 1:  send [system, user_1]                → get assistant_1
    Turn 2:  send [system, user_1, assistant_1, user_2] → get assistant_2
    ...

  Each generated assistant response is scored against expected_content
  using lexical metrics. Judge metrics (coherence, context_retention)
  receive the full conversation history as context.

  Final ConversationResult aggregates scores across all turns.
  A SampleResult is also synthesised so results flow into the normal store.
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
    ConversationDataset,
    ConversationResult,
    ConversationSample,
    ModelConfig,
    RunConfig,
    RunResult,
    SampleResult,
    TurnResult,
)
from llmbench.metrics import _exact_match, _token_f1
from llmbench.providers.base import BaseProvider, get_provider

log     = logging.getLogger("llmbench.runner_multiturn")
console = Console()


# ---------------------------------------------------------------------------
# Per-turn scoring (lexical — no API calls)
# ---------------------------------------------------------------------------

def _score_turn(generated: str, expected: str | None) -> dict[str, float]:
    """Compute lexical scores for one turn without the metric plugin system."""
    if expected is None:
        return {}

    from llmbench.metrics import _norm, _token_f1

    em = float(_norm(generated) == _norm(expected))
    f1 = _token_f1(generated, expected)

    # ROUGE-L
    try:
        from rouge_score import rouge_scorer as rs
        scorer  = rs.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_l = scorer.score(expected, generated)["rougeL"].fmeasure
    except Exception:
        rouge_l = float("nan")

    return {
        "turn_exact_match": em,
        "turn_f1":          f1,
        "turn_rouge_l":     rouge_l,
    }


def _aggregate_turn_scores(turn_results: list[TurnResult]) -> dict[str, float]:
    """Mean each score key across all turns that have it."""
    from collections import defaultdict
    import math

    buckets: dict[str, list[float]] = defaultdict(list)
    for tr in turn_results:
        for k, v in tr.scores.items():
            if not math.isnan(v):
                buckets[k].append(v)

    return {k: sum(vs) / len(vs) for k, vs in buckets.items() if vs}


# ---------------------------------------------------------------------------
# Conversation runner
# ---------------------------------------------------------------------------

class ConversationRunner:
    """
    Runs multi-turn eval over a ConversationDataset.

    Args:
        provider:       LLM provider for generating responses.
        judge_provider: Optional judge for coherence / context_retention metrics.
        concurrency:    Max parallel conversations.
    """

    def __init__(
        self,
        provider: BaseProvider,
        judge_provider: BaseProvider | None = None,
        concurrency: int = 4,
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
        dataset: ConversationDataset,
        metrics: list[str] | None = None,
        tags: dict[str, str] | None = None,
        verbose: bool = True,
    ) -> RunResult:
        """Synchronous entry point."""
        try:
            return asyncio.run(self._run_async(dataset, metrics, tags, verbose))
        except RuntimeError:
            import nest_asyncio; nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self._run_async(dataset, metrics, tags, verbose)
            )

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------

    async def _run_async(
        self,
        dataset: ConversationDataset,
        metrics: list[str] | None,
        tags: dict[str, str] | None,
        verbose: bool,
    ) -> RunResult:
        task_def = registry.get_task("multi_turn")
        resolved_metrics = metrics or task_def.default_metrics

        run_config = RunConfig(
            dataset_name=dataset.name,
            task_type="multi_turn",
            llm_config=self.provider.config,
            metrics=resolved_metrics,
            judge_model=self.judge_provider.config if self.judge_provider else None,
            tags=tags or {},
        )

        if verbose:
            console.rule("[bold]LLMBench Multi-Turn Run[/]")
            console.print(f"  Model        : [cyan]{self.provider.config.slug}[/]")
            console.print(f"  Dataset      : [cyan]{dataset.name}[/] ({len(dataset)} conversations)")
            console.print(f"  Metrics      : [cyan]{', '.join(resolved_metrics)}[/]\n")

        # --- Run all conversations in parallel (with concurrency cap) ----
        conv_results = await self._run_all(dataset, verbose)

        # --- Judge metrics (operate on the full conversation context) ----
        judge_metrics = [m for m in resolved_metrics
                         if m in ("conversation_coherence", "context_retention")]

        if judge_metrics and self.judge_provider:
            if verbose:
                console.print("[bold]Running judge metrics…[/]")
            for cr in conv_results:
                await self._run_judge_metrics(cr, judge_metrics)

        # --- Synthesise SampleResults for the normal store / reporter ----
        sample_results = self._to_sample_results(conv_results)

        # --- Aggregate across all conversations --------------------------
        aggregate: dict[str, float] = {}
        for m in resolved_metrics:
            plugin = registry.get_metric(m)
            try:
                scores = plugin.fn(sample_results, judge_provider=self.judge_provider)
                aggregate.update(scores)
            except Exception as exc:
                log.warning("Metric '%s' failed: %s", m, exc)
                aggregate[m] = float("nan")

        failed  = sum(1 for cr in conv_results if cr.error)
        tokens  = sum(cr.prompt_tokens + cr.completion_tokens for cr in conv_results)
        latency = sum(cr.total_latency_ms for cr in conv_results)

        result = RunResult(
            run_id=run_config.run_id,
            config=run_config,
            sample_results=sample_results,
            aggregate_scores=aggregate,
            total_samples=len(conv_results),
            failed_samples=failed,
            total_tokens=tokens,
            total_latency_ms=latency,
        )

        if verbose:
            self._print_summary(result)

        return result

    async def _run_all(
        self, dataset: ConversationDataset, verbose: bool
    ) -> list[ConversationResult]:
        results: list[ConversationResult] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            disable=not verbose,
        ) as progress:
            task = progress.add_task("Running conversations…", total=len(dataset))

            async def _one(conv: ConversationSample) -> ConversationResult:
                async with self._semaphore:
                    result = await self._run_conversation(conv)
                    progress.advance(task)
                    return result

            results = list(await asyncio.gather(*[_one(c) for c in dataset]))

        return results

    async def _run_conversation(
        self, conv: ConversationSample
    ) -> ConversationResult:
        """
        Replay one conversation turn-by-turn.

        Message history grows with each turn — the model sees the full
        prior context, mimicking a real chat session.
        """
        system    = conv.system_prompt or "You are a helpful assistant."
        history:  list[dict] = [{"role": "system", "content": system}]
        turn_results: list[TurnResult] = []
        total_prompt     = 0
        total_completion = 0
        total_latency    = 0.0
        turn_idx         = 0

        # Walk through turns, skipping the system turn
        turns = [t for t in conv.turns if t.role != "system"]

        i = 0
        while i < len(turns):
            turn = turns[i]

            if turn.role == "user":
                history.append({"role": "user", "content": turn.content})

                # Find the expected assistant response (next turn if it exists)
                expected: str | None = None
                if i + 1 < len(turns) and turns[i + 1].role == "assistant":
                    expected = turns[i + 1].expected_content or turns[i + 1].content

                # Call the model
                t0  = time.perf_counter()
                inf = await self.provider.async_infer_messages(history)
                lat = (time.perf_counter() - t0) * 1000

                total_prompt     += inf.prompt_tokens
                total_completion += inf.completion_tokens
                total_latency    += lat

                generated = inf.text if not inf.error else ""
                scores    = _score_turn(generated, expected)

                turn_results.append(TurnResult(
                    turn_index=turn_idx,
                    generated_output=generated,
                    expected_output=expected,
                    scores=scores,
                    confidence=inf.confidence,
                    latency_ms=lat,
                    error=inf.error,
                ))

                # Append the model's response to history for next turn
                history.append({"role": "assistant", "content": generated})
                turn_idx += 1
                i += 2   # skip the reference assistant turn
            else:
                i += 1

        agg_scores = _aggregate_turn_scores(turn_results)

        return ConversationResult(
            conversation_id=conv.id,
            model_slug=self.provider.config.slug,
            turn_results=turn_results,
            aggregate_scores=agg_scores,
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_latency_ms=total_latency,
        )

    async def _run_judge_metrics(
        self,
        cr: ConversationResult,
        judge_metrics: list[str],
    ) -> None:
        """Run LLM-as-judge metrics on completed conversation results."""
        import json as _json
        from llmbench.metrics import _judge_one

        # Build conversation history string for context
        full_history = ""
        for tr in cr.turn_results:
            full_history += f"User: {tr.generated_output[:200]}\n"
            full_history += f"Assistant: {tr.generated_output[:200]}\n\n"

        # Judge the last assistant turn
        last_turn = cr.turn_results[-1] if cr.turn_results else None
        if not last_turn or last_turn.error:
            return

        for metric in judge_metrics:
            # Temporarily attach context for the judge prompt
            last_turn.context = [full_history]  # type: ignore[attr-defined]
            score, reasoning = await _judge_one(metric, last_turn, self.judge_provider)
            cr.aggregate_scores[metric] = score

    @staticmethod
    def _to_sample_results(conv_results: list[ConversationResult]) -> list[SampleResult]:
        """
        Convert ConversationResults to SampleResults so they can flow
        into the existing store, reporter, and metric aggregation logic.

        One SampleResult per conversation — scores are the mean turn scores.
        generated_output = concatenated assistant responses.
        """
        results = []
        for cr in conv_results:
            generated = " | ".join(
                tr.generated_output for tr in cr.turn_results if not tr.error
            )
            expected = " | ".join(
                tr.expected_output for tr in cr.turn_results
                if tr.expected_output and not tr.error
            ) or None

            # Mean confidence across turns
            confs = [tr.confidence for tr in cr.turn_results if tr.confidence is not None]
            confidence = sum(confs) / len(confs) if confs else None

            sr = SampleResult(
                sample_id=cr.conversation_id,
                model_slug=cr.model_slug,
                generated_output=generated,
                prompt_tokens=cr.prompt_tokens,
                completion_tokens=cr.completion_tokens,
                latency_ms=cr.total_latency_ms,
                scores={**cr.aggregate_scores},
                expected_output=expected,
                confidence=confidence,
                error=cr.error,
            )
            results.append(sr)
        return results

    @staticmethod
    def _print_summary(result: RunResult) -> None:
        console.rule("[bold green]Multi-Turn Run Complete[/]")
        s = result.summary()
        console.print(f"  Run ID  : [dim]{s['run_id']}[/]")
        console.print(f"  Convs   : {s['n_samples']} ({s['n_failed']} failed)")
        console.print(f"  Tokens  : {result.total_tokens:,}")
        console.print("\n  [bold]Scores:[/]")
        for metric, score in s["scores"].items():
            if score != score: continue   # skip NaN
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            console.print(f"    {metric:<30} {score:.4f}  [{bar}]")


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_conversation_runner(
    provider_config: ModelConfig,
    judge_config: ModelConfig | None = None,
    concurrency: int = 4,
    provider_kwargs: dict[str, Any] | None = None,
    judge_kwargs: dict[str, Any] | None = None,
) -> ConversationRunner:
    provider = get_provider(provider_config, **(provider_kwargs or {}))
    judge    = get_provider(judge_config, **(judge_kwargs or {})) if judge_config else None
    return ConversationRunner(provider=provider, judge_provider=judge, concurrency=concurrency)
