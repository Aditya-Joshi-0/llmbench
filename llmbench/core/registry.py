"""
llmbench/core/registry.py
Central registry for task types and metric plugins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from llmbench.core.schema import EvalSample, TaskType


@dataclass
class TaskDefinition:
    name: str
    description: str
    requires_context: bool = False
    requires_expected_output: bool = True
    default_metrics: list[str] = field(default_factory=list)
    prompt_template: str = "{input}"
    system_prompt: str = "You are a helpful, precise assistant."
    is_multi_turn: bool = False     # ← NEW: signals ConversationRunner

    def build_prompt(self, sample: EvalSample) -> str:
        from jinja2 import Template
        ctx = {
            "input": sample.input,
            "context": "\n\n".join(sample.context) if sample.context else "",
            "expected_output": sample.expected_output or "",
            **sample.metadata,
        }
        return Template(self.prompt_template).render(**ctx)


@dataclass
class MetricPlugin:
    name: str
    description: str
    requires_expected: bool = True
    requires_context: bool = False
    fn: Callable[..., dict[str, float]] | None = None


class EvalRegistry:
    def __init__(self) -> None:
        self._tasks:   dict[str, TaskDefinition] = {}
        self._metrics: dict[str, MetricPlugin]   = {}

    def register_task(self, task: TaskDefinition) -> None:
        self._tasks[task.name] = task

    def get_task(self, name: str) -> TaskDefinition:
        if name not in self._tasks:
            raise KeyError(
                f"Task '{name}' not registered. Available: {list(self._tasks)}"
            )
        return self._tasks[name]

    def list_tasks(self) -> list[str]:
        return list(self._tasks)

    def register_metric(
        self,
        name: str,
        description: str,
        fn: Callable,
        requires_expected: bool = True,
        requires_context: bool = False,
    ) -> None:
        self._metrics[name] = MetricPlugin(
            name=name,
            description=description,
            requires_expected=requires_expected,
            requires_context=requires_context,
            fn=fn,
        )

    def get_metric(self, name: str) -> MetricPlugin:
        if name not in self._metrics:
            raise KeyError(
                f"Metric '{name}' not registered. Available: {list(self._metrics)}"
            )
        return self._metrics[name]

    def list_metrics(self) -> list[str]:
        return list(self._metrics)


registry = EvalRegistry()

# ── Single-turn tasks ────────────────────────────────────────────────────────

registry.register_task(TaskDefinition(
    name=TaskType.OPEN_QA,
    description="Single-turn open-domain question answering.",
    default_metrics=["exact_match", "f1", "rouge_l", "bertscore", "ece"],
    system_prompt="Answer the question directly and concisely.",
    prompt_template="{{ input }}",
))

registry.register_task(TaskDefinition(
    name=TaskType.RAG_FAITHFULNESS,
    description="Answer grounded in retrieved context.",
    requires_context=True,
    requires_expected_output=False,
    default_metrics=["llm_faithfulness", "llm_relevance"],
    system_prompt=(
        "Answer using ONLY the provided context. "
        "If the context doesn't contain the answer, say so."
    ),
    prompt_template="Context:\n{{ context }}\n\nQuestion: {{ input }}",
))

registry.register_task(TaskDefinition(
    name=TaskType.SUMMARIZATION,
    description="Abstractive summarization against a reference.",
    default_metrics=["rouge_1", "rouge_2", "rouge_l", "bertscore"],
    system_prompt="Summarize the following text concisely.",
    prompt_template="Summarize the following:\n\n{{ input }}",
))

registry.register_task(TaskDefinition(
    name=TaskType.CODE_CORRECTNESS,
    description="Code generation evaluated for correctness.",
    default_metrics=["exact_match", "rouge_l", "llm_code_quality"],
    system_prompt="You are an expert programmer. Write clean, correct code.",
    prompt_template="{{ input }}\n\nProvide only the code, no explanations.",
))

registry.register_task(TaskDefinition(
    name=TaskType.CLASSIFICATION,
    description="Label classification — checks exact label match.",
    default_metrics=["exact_match", "f1"],
    system_prompt="Classify the input. Output only the label, nothing else.",
    prompt_template="{{ input }}",
))

# ── Multi-turn task ──────────────────────────────────────────────────────────

registry.register_task(TaskDefinition(
    name=TaskType.MULTI_TURN,
    description=(
        "Multi-turn conversation eval. Replays all user turns through the model "
        "and scores each assistant response."
    ),
    requires_expected_output=True,
    is_multi_turn=True,
    default_metrics=[
        "turn_exact_match",
        "turn_f1",
        "turn_rouge_l",
        "conversation_coherence",
        "context_retention",
    ],
    system_prompt="You are a helpful, consistent assistant.",
))
