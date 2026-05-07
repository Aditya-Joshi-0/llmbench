"""
llmbench/core/schema.py
Canonical data models. Every loader, provider, metric, and store
works exclusively with these types — no raw dicts downstream.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    OPEN_QA           = "open_qa"
    RAG_FAITHFULNESS  = "rag_faithfulness"
    SUMMARIZATION     = "summarization"
    CODE_CORRECTNESS  = "code_correctness"
    CLASSIFICATION    = "classification"
    MULTI_TURN        = "multi_turn"
    CUSTOM            = "custom"


class ProviderName(str, Enum):
    OPENAI    = "openai"
    GROQ      = "groq"
    ANTHROPIC = "anthropic"
    VLLM      = "vllm"
    OLLAMA    = "ollama"


# ---------------------------------------------------------------------------
# Single-turn input schema
# ---------------------------------------------------------------------------

class EvalSample(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input: str
    expected_output: str | None = None
    context: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("input")
    @classmethod
    def input_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("EvalSample.input must not be empty")
        return v


class EvalDataset(BaseModel):
    name: str
    task_type: TaskType | str
    samples: list[EvalSample]
    source: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def filter(self, **kwargs) -> "EvalDataset":
        filtered = [
            s for s in self.samples
            if all(s.metadata.get(k) == v for k, v in kwargs.items())
        ]
        return self.model_copy(update={"samples": filtered})


# ---------------------------------------------------------------------------
# Multi-turn conversation schema
# ---------------------------------------------------------------------------

class Turn(BaseModel):
    """A single message in a conversation."""
    role: Literal["system", "user", "assistant"]
    content: str
    expected_content: str | None = None  # ground truth for assistant turns


class ConversationSample(BaseModel):
    """
    A full multi-turn conversation.
    The runner replays all user turns through the model and evaluates
    each assistant response against expected_content.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    turns: list[Turn]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def system_prompt(self) -> str | None:
        if self.turns and self.turns[0].role == "system":
            return self.turns[0].content
        return None

    @property
    def user_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "user"]

    @property
    def assistant_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "assistant"]

    @property
    def num_turns(self) -> int:
        return len(self.user_turns)


class ConversationDataset(BaseModel):
    name: str
    task_type: str = "multi_turn"
    conversations: list[ConversationSample]
    source: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def __len__(self) -> int:
        return len(self.conversations)

    def __iter__(self):
        return iter(self.conversations)

    def __getitem__(self, idx):
        return self.conversations[idx]


# ---------------------------------------------------------------------------
# Provider / model config
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    provider: ProviderName | str
    model_id: str
    temperature: float = 0.0
    max_tokens: int = 512
    extra_params: dict[str, Any] = Field(default_factory=dict)

    @property
    def slug(self) -> str:
        return f"{self.provider}/{self.model_id}"


# ---------------------------------------------------------------------------
# Per-sample output
# ---------------------------------------------------------------------------

class SampleResult(BaseModel):
    sample_id: str
    model_slug: str
    generated_output: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    scores: dict[str, float] = Field(default_factory=dict)
    judge_reasoning: str | None = None
    error: str | None = None
    expected_output: str | None = None
    context: list[str] | None = None
    sample_input: str = ""
    confidence: float | None = None   # extracted from log-probs (0-1)


class TurnResult(BaseModel):
    """Result for one assistant turn in a multi-turn conversation."""
    turn_index: int
    generated_output: str
    expected_output: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = None
    latency_ms: float = 0.0
    error: str | None = None
    context: list[str] | None = None
    


class ConversationResult(BaseModel):
    """Full result for one ConversationSample."""
    conversation_id: str
    model_slug: str
    turn_results: list[TurnResult]
    aggregate_scores: dict[str, float] = Field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_ms: float = 0.0
    error: str | None = None
    context: list[str] | None = None 

# ---------------------------------------------------------------------------
# Run-level aggregation
# ---------------------------------------------------------------------------

class RunConfig(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset_name: str
    task_type: str
    llm_config: ModelConfig
    metrics: list[str]
    judge_model: ModelConfig | None = None
    batch_size: int = 10
    seed: int = 42
    tags: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RunResult(BaseModel):
    run_id: str
    config: RunConfig
    sample_results: list[SampleResult]
    aggregate_scores: dict[str, float] = Field(default_factory=dict)
    total_samples: int = 0
    failed_samples: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    completed_at: datetime = Field(default_factory=datetime.utcnow)

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model": self.config.llm_config.slug,
            "dataset": self.config.dataset_name,
            "task": self.config.task_type,
            "n_samples": self.total_samples,
            "n_failed": self.failed_samples,
            "scores": self.aggregate_scores,
            "completed_at": self.completed_at.isoformat(),
        }
