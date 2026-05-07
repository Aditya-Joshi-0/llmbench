"""
llmbench/loaders/conversations.py
Loaders for multi-turn conversation datasets.

Supported formats:
  1. LLMBench native JSON  — list of {turns: [{role, content, expected_content?}]}
  2. ShareGPT format        — list of {conversations: [{from, value}]}
  3. OpenAI format          — list of {messages: [{role, content}]} with last assistant = expected
  4. Callable / generator   — yields dicts in any of the above shapes
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterator

from llmbench.core.schema import ConversationDataset, ConversationSample, Turn


# ---------------------------------------------------------------------------
# Format sniffers + normalizers
# ---------------------------------------------------------------------------

def _turns_from_native(record: dict) -> list[Turn]:
    """LLMBench native: {turns: [{role, content, expected_content?}]}"""
    return [
        Turn(
            role=t["role"],
            content=t["content"],
            expected_content=t.get("expected_content"),
        )
        for t in record["turns"]
    ]


def _turns_from_sharegpt(record: dict) -> list[Turn]:
    """
    ShareGPT format: {conversations: [{from: "human"|"gpt"|"system", value: "..."}]}
    The last GPT turn's value becomes expected_content.
    """
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    raw = record.get("conversations") or record.get("conversation", [])
    turns = []
    for i, msg in enumerate(raw):
        role = role_map.get(msg.get("from", "user"), "user")
        content = msg.get("value", "")
        # Mark last assistant turn as expected
        is_last_assistant = (
            role == "assistant"
            and i == len(raw) - 1
        )
        turns.append(Turn(
            role=role,
            content=content,
            expected_content=content if is_last_assistant else None,
        ))
    return turns


def _turns_from_openai(record: dict) -> list[Turn]:
    """
    OpenAI fine-tune format: {messages: [{role, content}]}.
    All assistant messages get their content set as expected_content.
    """
    turns = []
    for msg in record.get("messages", []):
        role    = msg["role"]
        content = msg["content"]
        turns.append(Turn(
            role=role,
            content=content,
            expected_content=content if role == "assistant" else None,
        ))
    return turns


def _sniff_and_parse(record: dict) -> list[Turn]:
    """Auto-detect format and return normalized turns."""
    if "turns" in record:
        return _turns_from_native(record)
    if "conversations" in record or "conversation" in record:
        return _turns_from_sharegpt(record)
    if "messages" in record:
        return _turns_from_openai(record)
    raise ValueError(
        "Cannot detect conversation format. Expected keys: "
        "'turns' (native), 'conversations' (ShareGPT), or 'messages' (OpenAI)."
    )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_conversations_json(
    path: str | Path,
    name: str | None = None,
    max_samples: int | None = None,
) -> ConversationDataset:
    """
    Load conversations from a JSON or JSONL file.
    Auto-detects native / ShareGPT / OpenAI format.
    """
    path = Path(path)
    raw  = path.read_text(encoding="utf-8")

    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        data    = json.loads(raw)
        records = data if isinstance(data, list) else data.get("data", data)

    if max_samples:
        records = records[:max_samples]

    conversations = []
    for rec in records:
        try:
            turns = _sniff_and_parse(rec)
            conversations.append(ConversationSample(
                turns=turns,
                metadata=rec.get("metadata", {}),
            ))
        except Exception as e:
            import warnings
            warnings.warn(f"Skipping malformed record: {e}")

    return ConversationDataset(
        name=name or path.stem,
        conversations=conversations,
        source=str(path),
    )


def load_conversations_callable(
    generator: Callable[[], Iterator[dict[str, Any]]],
    name: str = "custom_conversations",
) -> ConversationDataset:
    """Load from any Python generator yielding conversation dicts."""
    conversations = []
    for rec in generator():
        turns = _sniff_and_parse(rec)
        conversations.append(ConversationSample(
            turns=turns,
            metadata=rec.get("metadata", {}),
        ))
    return ConversationDataset(name=name, conversations=conversations, source="callable")


def load_conversations_hf(
    dataset_name: str,
    split: str = "train",
    max_samples: int | None = None,
    format: str = "auto",
) -> ConversationDataset:
    """
    Load a conversation dataset from Hugging Face Hub.

    Known compatible datasets:
      - "HuggingFaceH4/ultrachat_200k"  (messages format)
      - "ShareGPT4/sharegpt_gpt4"       (conversations format)
      - "Open-Orca/OpenOrca"            (OpenAI messages format)
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("pip install datasets")

    hf_ds = hf_load(dataset_name, split=split)
    if max_samples:
        hf_ds = hf_ds.select(range(min(max_samples, len(hf_ds))))

    conversations = []
    for record in hf_ds:
        try:
            turns = _sniff_and_parse(dict(record))
            conversations.append(ConversationSample(turns=turns))
        except Exception as e:
            import warnings
            warnings.warn(f"Skipping record: {e}")

    return ConversationDataset(
        name=dataset_name,
        conversations=conversations,
        source=f"hf:{dataset_name}:{split}",
    )
