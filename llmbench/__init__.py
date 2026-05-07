"""LLMBench — Production-grade LLM Evaluation Harness"""
__version__ = "0.2.0"

from llmbench.core.schema import (
    EvalDataset, EvalSample, ModelConfig,
    RunConfig, RunResult, SampleResult, TaskType,
    Turn, ConversationSample, ConversationDataset,
    TurnResult, ConversationResult,
)
from llmbench.core.registry import registry
from llmbench.core.runner import EvalRunner, build_runner
from llmbench.core.runner_multiturn import ConversationRunner, build_conversation_runner
from llmbench.loaders import DatasetLoader, loader
from llmbench.loaders.conversations import (
    load_conversations_json,
    load_conversations_callable,
    load_conversations_hf,
)
import llmbench.metrics  # trigger registration

__all__ = [
    "EvalDataset", "EvalSample", "ModelConfig",
    "RunConfig", "RunResult", "SampleResult", "TaskType",
    "Turn", "ConversationSample", "ConversationDataset",
    "TurnResult", "ConversationResult",
    "registry", "EvalRunner", "build_runner",
    "ConversationRunner", "build_conversation_runner",
    "DatasetLoader", "loader",
    "load_conversations_json", "load_conversations_callable", "load_conversations_hf",
]
