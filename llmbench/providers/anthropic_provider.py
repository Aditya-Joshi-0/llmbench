"""
llmbench/providers/anthropic_provider.py
Native Anthropic provider using the anthropic SDK.

Key differences from OpenAI-compatible providers:
  - Uses anthropic.AsyncAnthropic (not openai)
  - System prompt is a top-level param, not a message
  - No logprobs API — confidence always returns None
  - Supports multi-turn via messages list natively
  - Model IDs: claude-3-5-haiku-20241022, claude-3-5-sonnet-20241022, etc.
"""

from __future__ import annotations

import time
from typing import Any

from llmbench.core.schema import ModelConfig
from llmbench.providers.base import BaseProvider, InferenceResult


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, config: ModelConfig, api_key: str | None = None) -> None:
        super().__init__(config)
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "pip install anthropic   (required for the anthropic provider)"
            )
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def _call(self, prompt: str, system: str, **kwargs: Any) -> InferenceResult:
        return await self._chat(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            **kwargs,
        )

    async def async_infer_messages(
        self, messages: list[dict], **kwargs
    ) -> InferenceResult:
        """
        Multi-turn inference.
        Anthropic requires system to be a top-level param, not inside messages[].
        We extract it here and pass the rest as the messages list.
        """
        system_content = ""
        filtered: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system_content = m["content"]
            else:
                filtered.append({"role": m["role"], "content": m["content"]})

        return await self._chat(
            messages=filtered,
            system=system_content or "You are a helpful assistant.",
            **kwargs,
        )

    async def _chat(
        self,
        messages: list[dict],
        system: str = "You are a helpful assistant.",
        **kwargs: Any,
    ) -> InferenceResult:
        t0 = time.perf_counter()
        try:
            resp = await self._client.messages.create(
                model=self.config.model_id,
                max_tokens=self.config.max_tokens,
                system=system,
                messages=messages,
                temperature=self.config.temperature,
                **{**self.config.extra_params, **kwargs},
            )
        except Exception as exc:
            return InferenceResult(error=str(exc))

        latency = (time.perf_counter() - t0) * 1000
        text = "".join(
            block.text for block in resp.content
            if hasattr(block, "text")
        )
        return InferenceResult(
            text=text,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            latency_ms=latency,
            confidence=None,    # Anthropic API does not expose log-probs
        )
