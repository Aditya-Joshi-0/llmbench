"""
llmbench/providers/base.py
Abstract LLM provider interface + concrete implementations.

Key addition: log-prob extraction for calibration (ECE).
Every InferenceResult now carries an optional `confidence` float (0-1)
derived from the mean token log-probability of the completion.

Supported:
  - OpenAI  (logprobs=True in chat completions)
  - Groq    (same OpenAI-compatible API, logprobs supported)
  - vLLM / Ollama  (OpenAI-compatible, logprobs supported)
"""

from __future__ import annotations

import asyncio
import math
import time
from abc import ABC, abstractmethod
from typing import Any

from llmbench.core.schema import ModelConfig


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class InferenceResult:
    __slots__ = ("text", "prompt_tokens", "completion_tokens",
                 "latency_ms", "error", "confidence")

    def __init__(
        self,
        text: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        error: str | None = None,
        confidence: float | None = None,   # ← NEW: extracted from log-probs
    ) -> None:
        self.text             = text
        self.prompt_tokens    = prompt_tokens
        self.completion_tokens = completion_tokens
        self.latency_ms       = latency_ms
        self.error            = error
        self.confidence       = confidence


# ---------------------------------------------------------------------------
# Log-prob → confidence helpers                                     ← NEW
# ---------------------------------------------------------------------------

def _logprobs_to_confidence(token_logprobs: list[float]) -> float:
    """
    Convert a list of per-token log-probabilities to a single confidence score.

    Strategy: mean log-prob → exp() → probability in [0, 1].

    A high-confidence, short answer (e.g. "Paris") will have token logprobs
    close to 0 (log(1) = 0), giving confidence near 1.0.
    A low-confidence or rambling answer will have lower mean logprob → lower confidence.

    Args:
        token_logprobs: List of log P(token_i | context) for each generated token.

    Returns:
        float in [0, 1] representing model confidence.
    """
    if not token_logprobs:
        return 0.0
    # Filter out -inf (padding tokens)
    valid = [lp for lp in token_logprobs if lp > -1e9]
    if not valid:
        return 0.0
    mean_lp = sum(valid) / len(valid)
    return float(math.exp(mean_lp))   # always in (0, 1]


def _extract_logprobs_openai(response) -> float | None:
    """Extract confidence from an OpenAI-style chat completion response."""
    try:
        choice = response.choices[0]
        lc = choice.logprobs
        if lc is None:
            return None
        # OpenAI SDK: logprobs.content is a list of ChatCompletionTokenLogprob
        token_logprobs = [t.logprob for t in (lc.content or [])]
        return _logprobs_to_confidence(token_logprobs)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseProvider(ABC):
    MAX_RETRIES     = 3
    RETRY_BASE_DELAY = 1.0

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def _call(
        self, prompt: str, system: str, **kwargs: Any
    ) -> InferenceResult: ...

    async def async_infer(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        **kwargs: Any,
    ) -> InferenceResult:
        delay = self.RETRY_BASE_DELAY
        last_err: str = ""
        for attempt in range(self.MAX_RETRIES):
            try:
                return await self._call(prompt, system, **kwargs)
            except Exception as exc:
                last_err = str(exc)
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
        return InferenceResult(error=f"All {self.MAX_RETRIES} retries failed: {last_err}")

    # Multi-turn variant — sends a full message list instead of a single prompt
    async def async_infer_messages(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> InferenceResult:
        """
        Multi-turn inference. Sends the full message history as-is.
        Default implementation: concatenate history into a single prompt
        (overridden by chat-native providers).
        """
        # Flatten into a single prompt as fallback
        prompt = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
            if m["role"] != "system"
        )
        system = next(
            (m["content"] for m in messages if m["role"] == "system"),
            "You are a helpful assistant.",
        )
        return await self.async_infer(prompt, system=system, **kwargs)


# ---------------------------------------------------------------------------
# OpenAI provider  (also used for Anthropic via openai-compat SDK)
# ---------------------------------------------------------------------------

class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, config: ModelConfig, api_key: str | None = None) -> None:
        super().__init__(config)
        import openai
        self._client = openai.AsyncOpenAI(api_key=api_key)

    async def _call(self, prompt: str, system: str, **kwargs: Any) -> InferenceResult:
        return await self._chat(
            [{"role": "system", "content": system},
             {"role": "user",   "content": prompt}],
            **kwargs,
        )

    async def async_infer_messages(
        self, messages: list[dict], **kwargs
    ) -> InferenceResult:
        return await self._chat(messages, **kwargs)

    async def _chat(self, messages: list[dict], **kwargs) -> InferenceResult:
        t0 = time.perf_counter()
        resp = await self._client.chat.completions.create(
            model=self.config.model_id,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            logprobs=True,          # ← request log-probs
            top_logprobs=1,
            **{**self.config.extra_params, **kwargs},
        )
        latency = (time.perf_counter() - t0) * 1000
        confidence = _extract_logprobs_openai(resp)
        return InferenceResult(
            text=resp.choices[0].message.content or "",
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            latency_ms=latency,
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# Groq provider
# ---------------------------------------------------------------------------

class GroqProvider(BaseProvider):
    name = "groq"

    def __init__(self, config: ModelConfig, api_key: str | None = None) -> None:
        super().__init__(config)
        from groq import AsyncGroq
        self._client = AsyncGroq(api_key=api_key)

    async def _call(self, prompt: str, system: str, **kwargs: Any) -> InferenceResult:
        return await self._chat(
            [{"role": "system", "content": system},
             {"role": "user",   "content": prompt}],
            **kwargs,
        )

    async def async_infer_messages(
        self, messages: list[dict], **kwargs
    ) -> InferenceResult:
        return await self._chat(messages, **kwargs)

    async def _chat(self, messages: list[dict], **kwargs) -> InferenceResult:
        t0 = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(
                model=self.config.model_id,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                logprobs=True,
                top_logprobs=1,
            )
            confidence = _extract_logprobs_openai(resp)
        except Exception:
            # Retry without logprobs if not supported
            resp = await self._client.chat.completions.create(
                model=self.config.model_id,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            confidence = None

        latency = (time.perf_counter() - t0) * 1000
        return InferenceResult(
            text=resp.choices[0].message.content or "",
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            latency_ms=latency,
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# vLLM / Ollama  (OpenAI-compatible)
# ---------------------------------------------------------------------------

class VLLMProvider(BaseProvider):
    """
    Works with any OpenAI-compatible endpoint: vLLM, Ollama, LM Studio, etc.
    Pass base_url in ModelConfig.extra_params: {"base_url": "http://localhost:8000/v1"}
    Log-prob support depends on the serving framework — gracefully falls back to None.
    """
    name = "vllm"

    def __init__(self, config: ModelConfig, api_key: str = "EMPTY") -> None:
        super().__init__(config)
        import openai
        base_url = config.extra_params.get("base_url", "http://localhost:8000/v1")
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def _call(self, prompt: str, system: str, **kwargs: Any) -> InferenceResult:
        return await self._chat(
            [{"role": "system", "content": system},
             {"role": "user",   "content": prompt}],
            **kwargs,
        )

    async def async_infer_messages(
        self, messages: list[dict], **kwargs
    ) -> InferenceResult:
        return await self._chat(messages, **kwargs)

    async def _chat(self, messages: list[dict], **kwargs) -> InferenceResult:
        t0 = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(
                model=self.config.model_id,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                logprobs=True,
                top_logprobs=1,
            )
            confidence = _extract_logprobs_openai(resp)
        except Exception:
            # Retry without logprobs if server doesn't support them
            resp = await self._client.chat.completions.create(
                model=self.config.model_id,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            confidence = None

        latency = (time.perf_counter() - t0) * 1000
        return InferenceResult(
            text=resp.choices[0].message.content or "",
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            latency_ms=latency,
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def _get_anthropic_provider():
    from llmbench.providers.anthropic_provider import AnthropicProvider
    return AnthropicProvider


_PROVIDER_MAP: dict[str, type[BaseProvider]] = {
    "openai":  OpenAIProvider,
    "groq":    GroqProvider,
    "vllm":    VLLMProvider,
    "ollama":  VLLMProvider,
}


def get_provider(config: ModelConfig, **kwargs: Any) -> BaseProvider:
    key = str(config.provider).lower()
    if key == "anthropic":
        cls = _get_anthropic_provider()
        return cls(config, **kwargs)
    cls = _PROVIDER_MAP.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown provider '{config.provider}'. "
            f"Supported: {list(_PROVIDER_MAP) + ['anthropic']}"
        )
    return cls(config, **kwargs)
