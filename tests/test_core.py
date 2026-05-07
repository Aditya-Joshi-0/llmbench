"""
tests/test_core.py
Full LLMBench test suite — no API keys required, all providers mocked.

Coverage:
  v1  — Schema, loaders, registry, lexical metrics
  v2  — Log-prob confidence, ECE end-to-end, Anthropic provider
  v3  — ConversationSample schema, conversation loaders (3 formats),
         ConversationRunner logic, multi-turn metrics, turn scoring
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import llmbench.metrics  # register all metrics
from llmbench.core.schema import (
    ConversationDataset, ConversationResult, ConversationSample,
    EvalDataset, EvalSample, ModelConfig, SampleResult, TaskType,
    Turn, TurnResult,
)
from llmbench.loaders import DatasetLoader, _normalize, validate_dataset
from llmbench.loaders.conversations import (
    _turns_from_native, _turns_from_openai, _turns_from_sharegpt,
    load_conversations_json,
)


# ===========================================================================
# v1 — Schema
# ===========================================================================

class TestEvalSample:
    def test_id_auto_generated(self):
        s = EvalSample(input="Q?")
        assert len(s.id) == 36

    def test_empty_input_raises(self):
        with pytest.raises(Exception):
            EvalSample(input="   ")

    def test_optional_fields_default_to_none(self):
        s = EvalSample(input="Q?")
        assert s.expected_output is None
        assert s.context is None
        assert s.metadata == {}


class TestEvalDataset:
    def _ds(self, n=3):
        return EvalDataset(
            name="t", task_type=TaskType.OPEN_QA,
            samples=[EvalSample(input=f"Q{i}", expected_output=f"A{i}") for i in range(n)],
        )

    def test_len(self):
        assert len(self._ds(5)) == 5

    def test_iter(self):
        ds = self._ds(2)
        inputs = [s.input for s in ds]
        assert inputs == ["Q0", "Q1"]

    def test_filter_by_metadata(self):
        s1 = EvalSample(input="Q1", metadata={"domain": "math"})
        s2 = EvalSample(input="Q2", metadata={"domain": "science"})
        ds = EvalDataset(name="t", task_type="open_qa", samples=[s1, s2])
        assert len(ds.filter(domain="math")) == 1
        assert ds.filter(domain="math")[0].input == "Q1"


# ===========================================================================
# v1 — Loaders
# ===========================================================================

class TestNormalize:
    def test_question_answer_alias(self):
        r = _normalize({"question": "Q?", "answer": "A."})
        assert r["input"] == "Q?" and r["expected_output"] == "A."

    def test_unknown_keys_go_to_metadata(self):
        r = _normalize({"input": "Q", "difficulty": "hard"})
        assert r["metadata"]["difficulty"] == "hard"

    def test_passthrough(self):
        r = _normalize({"input": "Q", "expected_output": "A"})
        assert r == {"input": "Q", "expected_output": "A"}


class TestJSONLoader:
    def test_flat_list(self, tmp_path):
        data = [{"input": "Q1", "expected_output": "A1"},
                {"question": "Q2", "answer": "A2"}]
        f = tmp_path / "d.json"
        f.write_text(json.dumps(data))
        ds = DatasetLoader().load(f, task_type="open_qa", validate=False)
        assert len(ds) == 2
        assert ds[1].input == "Q2"

    def test_nested_list(self, tmp_path):
        data = {"items": [{"input": "Q", "expected_output": "A"}]}
        f = tmp_path / "d.json"
        f.write_text(json.dumps(data))
        ds = DatasetLoader().load(f, task_type="open_qa",
                                  records_key="items", validate=False)
        assert len(ds) == 1

    def test_jsonl(self, tmp_path):
        lines = [json.dumps({"input": f"Q{i}", "expected_output": f"A{i}"}) for i in range(3)]
        f = tmp_path / "d.jsonl"
        f.write_text("\n".join(lines))
        ds = DatasetLoader().load(f, task_type="open_qa", validate=False)
        assert len(ds) == 3


class TestCSVLoader:
    def test_basic(self, tmp_path):
        f = tmp_path / "d.csv"
        f.write_text("input,expected_output\nQ1,A1\nQ2,A2\n")
        ds = DatasetLoader().load(f, task_type="open_qa", validate=False)
        assert len(ds) == 2

    def test_pipe_context(self, tmp_path):
        f = tmp_path / "d.csv"
        f.write_text("input,expected_output,context\nQ,A,c1 ||| c2\n")
        ds = DatasetLoader().load(f, task_type="rag_faithfulness",
                                  context_col="context", validate=False)
        assert ds[0].context == ["c1", "c2"]

    def test_extra_cols_become_metadata(self, tmp_path):
        f = tmp_path / "d.csv"
        f.write_text("input,expected_output,difficulty\nQ,A,hard\n")
        ds = DatasetLoader().load(f, task_type="open_qa", validate=False)
        assert ds[0].metadata["difficulty"] == "hard"


class TestCallableLoader:
    def test_generator(self):
        def gen():
            for i in range(4):
                yield {"input": f"Q{i}", "expected_output": f"A{i}"}
        ds = DatasetLoader().load(gen, task_type="open_qa", validate=False)
        assert len(ds) == 4


# ===========================================================================
# v1 — Validation
# ===========================================================================

class TestValidation:
    def test_warns_missing_expected(self):
        ds = EvalDataset(name="t", task_type="open_qa",
                         samples=[EvalSample(input="Q")])
        assert any("expected_output" in w for w in validate_dataset(ds))

    def test_warns_missing_context_for_rag(self):
        ds = EvalDataset(name="t", task_type="rag_faithfulness",
                         samples=[EvalSample(input="Q", expected_output="A")])
        assert any("context" in w for w in validate_dataset(ds))

    def test_clean_dataset_no_warnings(self):
        ds = EvalDataset(name="t", task_type="open_qa",
                         samples=[EvalSample(input="Q", expected_output="A")])
        assert validate_dataset(ds) == []


# ===========================================================================
# v1 — Registry
# ===========================================================================

class TestRegistry:
    def test_builtin_tasks(self):
        from llmbench.core.registry import registry
        for t in ["open_qa", "rag_faithfulness", "summarization",
                  "code_correctness", "classification", "multi_turn"]:
            assert registry.get_task(t) is not None

    def test_multi_turn_task_is_flagged(self):
        from llmbench.core.registry import registry
        assert registry.get_task("multi_turn").is_multi_turn is True

    def test_all_builtin_metrics(self):
        from llmbench.core.registry import registry
        for m in ["exact_match", "f1", "rouge_l", "bertscore",
                  "cosine_similarity", "llm_faithfulness", "ece",
                  "turn_exact_match", "turn_f1", "turn_rouge_l",
                  "conversation_coherence", "context_retention"]:
            assert registry.get_metric(m) is not None

    def test_total_metric_count(self):
        from llmbench.core.registry import registry
        assert len(registry.list_metrics()) == 18

    def test_unknown_task_raises(self):
        from llmbench.core.registry import registry
        with pytest.raises(KeyError):
            registry.get_task("nonexistent")

    def test_custom_task_registration(self):
        from llmbench.core.registry import registry, TaskDefinition
        registry.register_task(TaskDefinition(
            name="my_task", description="test", default_metrics=["exact_match"],
        ))
        assert registry.get_task("my_task") is not None


# ===========================================================================
# v1 — Lexical metrics
# ===========================================================================

def _sr(generated: str, expected: str, **kw) -> SampleResult:
    return SampleResult(sample_id="x", model_slug="t/m",
                        generated_output=generated,
                        expected_output=expected, **kw)


class TestLexicalMetrics:
    from llmbench.metrics import _exact_match, _f1

    def test_exact_match_hit(self):
        from llmbench.metrics import _exact_match
        assert _exact_match([_sr("Paris", "Paris")])["exact_match"] == 1.0

    def test_exact_match_case_insensitive(self):
        from llmbench.metrics import _exact_match
        assert _exact_match([_sr("paris", "Paris")])["exact_match"] == 1.0

    def test_exact_match_miss(self):
        from llmbench.metrics import _exact_match
        assert _exact_match([_sr("London", "Paris")])["exact_match"] == 0.0

    def test_f1_perfect(self):
        from llmbench.metrics import _f1
        assert _f1([_sr("hello world", "hello world")])["f1"] == 1.0

    def test_f1_partial(self):
        from llmbench.metrics import _f1
        score = _f1([_sr("the cat sat on the mat", "the cat")])["f1"]
        assert 0.0 < score < 1.0

    def test_f1_zero(self):
        from llmbench.metrics import _f1
        assert _f1([_sr("alpha beta", "gamma delta")])["f1"] == 0.0

    def test_skips_errored_results(self):
        from llmbench.metrics import _exact_match
        r = SampleResult(sample_id="y", model_slug="t/m",
                         generated_output="x", expected_output="x",
                         error="timeout")
        assert _exact_match([r])["exact_match"] == 0.0


# ===========================================================================
# v2 — Log-prob confidence extraction
# ===========================================================================

class TestLogprobConfidence:
    def test_high_confidence(self):
        from llmbench.providers.base import _logprobs_to_confidence
        # Near-zero logprobs → confidence near 1.0
        conf = _logprobs_to_confidence([-0.01, -0.02, -0.01])
        assert conf > 0.95

    def test_low_confidence(self):
        from llmbench.providers.base import _logprobs_to_confidence
        conf = _logprobs_to_confidence([-3.0, -4.0, -5.0])
        assert conf < 0.05

    def test_empty_returns_zero(self):
        from llmbench.providers.base import _logprobs_to_confidence
        assert _logprobs_to_confidence([]) == 0.0

    def test_neginf_filtered_out(self):
        from llmbench.providers.base import _logprobs_to_confidence
        conf = _logprobs_to_confidence([-1e10, -0.1, -0.2])
        assert 0.0 < conf < 1.0

    def test_output_always_in_zero_one(self):
        from llmbench.providers.base import _logprobs_to_confidence
        for logprobs in [[-0.001], [-100.0], [-0.5, -1.0, -2.0]]:
            c = _logprobs_to_confidence(logprobs)
            assert 0.0 <= c <= 1.0, f"Out of range for {logprobs}: {c}"

    def test_extract_logprobs_openai_none_when_no_logprobs(self):
        from llmbench.providers.base import _extract_logprobs_openai
        mock_resp = MagicMock()
        mock_resp.choices[0].logprobs = None
        assert _extract_logprobs_openai(mock_resp) is None

    def test_extract_logprobs_openai_computes_value(self):
        from llmbench.providers.base import _extract_logprobs_openai
        tok = MagicMock(); tok.logprob = -0.1
        lp  = MagicMock(); lp.content  = [tok, tok]
        mock_resp = MagicMock()
        mock_resp.choices[0].logprobs = lp
        conf = _extract_logprobs_openai(mock_resp)
        assert conf is not None
        assert abs(conf - math.exp(-0.1)) < 1e-6


# ===========================================================================
# v2 — ECE end-to-end with real confidence values
# ===========================================================================

class TestECE:
    def _results(self, pairs: list[tuple[str, str, float]]) -> list[SampleResult]:
        """(generated, expected, confidence) → list[SampleResult]"""
        return [
            SampleResult(sample_id=str(i), model_slug="t/m",
                         generated_output=g, expected_output=e,
                         confidence=c)
            for i, (g, e, c) in enumerate(pairs)
        ]

    def test_perfect_calibration(self):
        """Model is always right with confidence 1.0 → ECE = 0."""
        from llmbench.metrics import _ece
        pairs = [("Paris", "Paris", 1.0)] * 15
        result = _ece(self._results(pairs))
        assert abs(result["ece"]) < 0.01

    def test_worst_calibration(self):
        """Model always wrong but 100% confident → ECE near 1."""
        from llmbench.metrics import _ece
        pairs = [("wrong", "Paris", 1.0)] * 15
        result = _ece(self._results(pairs))
        assert result["ece"] > 0.8

    def test_nan_when_too_few_samples(self):
        from llmbench.metrics import _ece
        pairs = [("Paris", "Paris", 0.9)] * 3
        result = _ece(self._results(pairs))
        assert math.isnan(result["ece"])

    def test_nan_when_no_confidence_scores(self):
        from llmbench.metrics import _ece
        results = [SampleResult(sample_id=str(i), model_slug="t/m",
                                generated_output="Paris", expected_output="Paris",
                                confidence=None)
                   for i in range(15)]
        result = _ece(results)
        assert math.isnan(result["ece"])

    def test_ece_in_zero_one_range(self):
        from llmbench.metrics import _ece
        import random; random.seed(42)
        pairs = [
            ("Paris" if random.random() > 0.4 else "London",
             "Paris",
             random.uniform(0.3, 0.9))
            for _ in range(20)
        ]
        result = _ece(self._results(pairs))
        assert 0.0 <= result["ece"] <= 1.0


# ===========================================================================
# v2 — Anthropic provider
# ===========================================================================

class TestAnthropicProvider:
    def _make_provider(self):
        from llmbench.providers.anthropic_provider import AnthropicProvider
        mock_client = MagicMock()
        with patch("anthropic.AsyncAnthropic", return_value=mock_client):
            p = AnthropicProvider(
                ModelConfig(provider="anthropic",
                            model_id="claude-3-5-haiku-20241022"),
                api_key="test-key",
            )
        p._client = mock_client
        return p

    def test_confidence_is_always_none(self):
        """Anthropic has no logprobs API — confidence must be None."""
        p = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="Paris")]
        mock_resp.usage.input_tokens  = 10
        mock_resp.usage.output_tokens = 3
        p._client.messages.create = AsyncMock(return_value=mock_resp)

        import asyncio
        result = asyncio.run(p._call("What is the capital of France?", "You are helpful."))
        assert result.confidence is None

    def test_text_extracted_correctly(self):
        p = self._make_provider()
        block = MagicMock(); block.text = "Paris"
        mock_resp = MagicMock()
        mock_resp.content = [block]
        mock_resp.usage.input_tokens  = 10
        mock_resp.usage.output_tokens = 1
        p._client.messages.create = AsyncMock(return_value=mock_resp)

        import asyncio
        result = asyncio.run(p._call("Capital of France?", "Be helpful."))
        assert result.text == "Paris"

    def test_system_extracted_from_messages_in_multiturn(self):
        """System role must be extracted and passed as top-level param."""
        p = self._make_provider()
        calls = []

        async def _mock_create(**kwargs):
            calls.append(kwargs)
            r = MagicMock()
            r.content = [MagicMock(text="ok")]
            r.usage.input_tokens  = 5
            r.usage.output_tokens = 1
            return r

        p._client.messages.create = _mock_create

        import asyncio
        messages = [
            {"role": "system",    "content": "You are a math tutor."},
            {"role": "user",      "content": "What is 2+2?"},
        ]
        asyncio.run(p.async_infer_messages(messages))
        assert calls[0]["system"] == "You are a math tutor."
        # System must NOT be in messages list
        roles = [m["role"] for m in calls[0]["messages"]]
        assert "system" not in roles

    def test_provider_routing(self):
        """get_provider('anthropic') must return AnthropicProvider, not OpenAIProvider."""
        from llmbench.providers.base import get_provider
        mock_sdk = MagicMock()
        mock_sdk.AsyncAnthropic.return_value = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_sdk}):
            p = get_provider(ModelConfig(provider="anthropic",
                                         model_id="claude-3-5-haiku-20241022"),
                              api_key="test")
        assert type(p).__name__ == "AnthropicProvider"


# ===========================================================================
# v3 — ConversationSample schema
# ===========================================================================

class TestConversationSchema:
    def _conv(self) -> ConversationSample:
        return ConversationSample(turns=[
            Turn(role="system",    content="You are helpful."),
            Turn(role="user",      content="Hello"),
            Turn(role="assistant", content="Hi!",   expected_content="Hi!"),
            Turn(role="user",      content="Bye"),
            Turn(role="assistant", content="Bye!",  expected_content="Bye!"),
        ])

    def test_system_prompt_extracted(self):
        assert self._conv().system_prompt == "You are helpful."

    def test_num_turns(self):
        assert self._conv().num_turns == 2

    def test_user_turns(self):
        ut = self._conv().user_turns
        assert len(ut) == 2
        assert ut[0].content == "Hello"

    def test_assistant_turns(self):
        at = self._conv().assistant_turns
        assert len(at) == 2
        assert all(t.expected_content is not None for t in at)

    def test_no_system_returns_none(self):
        conv = ConversationSample(turns=[Turn(role="user", content="Hi")])
        assert conv.system_prompt is None


# ===========================================================================
# v3 — Conversation loaders (format detection)
# ===========================================================================

class TestNativeFormat:
    def test_basic(self):
        record = {"turns": [
            {"role": "user",      "content": "Hi"},
            {"role": "assistant", "content": "Hello", "expected_content": "Hello"},
        ]}
        turns = _turns_from_native(record)
        assert len(turns) == 2
        assert turns[1].expected_content == "Hello"

    def test_missing_expected_content_is_none(self):
        record = {"turns": [{"role": "user", "content": "Hi"}]}
        turns = _turns_from_native(record)
        assert turns[0].expected_content is None


class TestShareGPTFormat:
    def test_role_mapping(self):
        record = {"conversations": [
            {"from": "human", "value": "Question?"},
            {"from": "gpt",   "value": "Answer."},
        ]}
        turns = _turns_from_sharegpt(record)
        assert turns[0].role == "user"
        assert turns[1].role == "assistant"

    def test_last_gpt_gets_expected_content(self):
        record = {"conversations": [
            {"from": "human", "value": "Q?"},
            {"from": "gpt",   "value": "A."},
        ]}
        turns = _turns_from_sharegpt(record)
        assert turns[1].expected_content == "A."

    def test_system_mapped(self):
        record = {"conversations": [{"from": "system", "value": "Be helpful."}]}
        turns = _turns_from_sharegpt(record)
        assert turns[0].role == "system"


class TestOpenAIFormat:
    def test_all_assistant_turns_get_expected(self):
        record = {"messages": [
            {"role": "system",    "content": "Sys."},
            {"role": "user",      "content": "Q?"},
            {"role": "assistant", "content": "A."},
        ]}
        turns = _turns_from_openai(record)
        assistant_turns = [t for t in turns if t.role == "assistant"]
        assert all(t.expected_content is not None for t in assistant_turns)

    def test_user_turns_have_no_expected(self):
        record = {"messages": [
            {"role": "user",      "content": "Q?"},
            {"role": "assistant", "content": "A."},
        ]}
        turns = _turns_from_openai(record)
        assert turns[0].expected_content is None


class TestConversationJSONLoader:
    def test_native_format(self, tmp_path):
        data = [{"turns": [
            {"role": "user",      "content": "Hi"},
            {"role": "assistant", "content": "Hello", "expected_content": "Hello"},
        ]}]
        f = tmp_path / "c.json"
        f.write_text(json.dumps(data))
        ds = load_conversations_json(f)
        assert len(ds) == 1
        assert ds[0].num_turns == 1

    def test_max_samples(self, tmp_path):
        data = [{"turns": [{"role": "user", "content": f"Q{i}"}]}
                for i in range(10)]
        f = tmp_path / "c.json"
        f.write_text(json.dumps(data))
        ds = load_conversations_json(f, max_samples=3)
        assert len(ds) == 3

    def test_sample_conversations_file(self):
        """Smoke test the bundled sample dataset."""
        ds = load_conversations_json("tasks/sample_conversations.json")
        assert len(ds) == 3
        assert ds[0].num_turns == 3
        assert ds[0].system_prompt is not None


# ===========================================================================
# v3 — Turn scoring
# ===========================================================================

class TestTurnScoring:
    def test_exact_match_hit(self):
        from llmbench.core.runner_multiturn import _score_turn
        s = _score_turn("Paris", "Paris")
        assert s["turn_exact_match"] == 1.0
        assert s["turn_f1"] == 1.0

    def test_exact_match_miss(self):
        from llmbench.core.runner_multiturn import _score_turn
        s = _score_turn("London", "Paris")
        assert s["turn_exact_match"] == 0.0
        assert s["turn_f1"] == 0.0

    def test_partial_match(self):
        from llmbench.core.runner_multiturn import _score_turn
        s = _score_turn("the cat sat", "the cat sat on the mat")
        assert 0.0 < s["turn_f1"] < 1.0

    def test_none_expected_returns_empty(self):
        from llmbench.core.runner_multiturn import _score_turn
        assert _score_turn("anything", None) == {}

    def test_aggregate_is_mean(self):
        from llmbench.core.runner_multiturn import _aggregate_turn_scores
        tr1 = TurnResult(turn_index=0, generated_output="A",
                         scores={"turn_exact_match": 1.0, "turn_f1": 1.0})
        tr2 = TurnResult(turn_index=1, generated_output="B",
                         scores={"turn_exact_match": 0.0, "turn_f1": 0.0})
        agg = _aggregate_turn_scores([tr1, tr2])
        assert agg["turn_exact_match"] == 0.5
        assert agg["turn_f1"] == 0.5


# ===========================================================================
# v3 — Multi-turn metrics aggregation
# ===========================================================================

class TestMultiTurnMetrics:
    def _make_sample_result(self, scores: dict) -> SampleResult:
        sr = SampleResult(sample_id="x", model_slug="t/m", generated_output="out")
        sr.scores = scores
        return sr

    def test_turn_exact_match_aggregates(self):
        from llmbench.metrics import _turn_exact_match
        results = [
            self._make_sample_result({"turn_exact_match": 1.0}),
            self._make_sample_result({"turn_exact_match": 0.0}),
        ]
        out = _turn_exact_match(results)
        assert out["turn_exact_match"] == 0.5

    def test_turn_f1_aggregates(self):
        from llmbench.metrics import _turn_f1
        results = [
            self._make_sample_result({"turn_f1": 0.8}),
            self._make_sample_result({"turn_f1": 0.6}),
        ]
        out = _turn_f1(results)
        assert abs(out["turn_f1"] - 0.7) < 1e-6

    def test_nan_values_excluded(self):
        from llmbench.metrics import _turn_exact_match
        results = [
            self._make_sample_result({"turn_exact_match": float("nan")}),
            self._make_sample_result({"turn_exact_match": 1.0}),
        ]
        out = _turn_exact_match(results)
        assert out["turn_exact_match"] == 1.0

    def test_empty_returns_zero(self):
        from llmbench.metrics import _turn_exact_match
        out = _turn_exact_match([])
        assert out["turn_exact_match"] == 0.0


# ===========================================================================
# v3 — ConversationRunner (mocked provider)
# ===========================================================================

class TestConversationRunner:
    def _mock_provider(self, responses: list[str]):
        from llmbench.providers.base import InferenceResult
        provider  = MagicMock()
        provider.config = ModelConfig(provider="groq", model_id="llama-3.3-70b-versatile")
        responses_iter = iter(responses)

        async def _infer_messages(messages, **_):
            text = next(responses_iter, "fallback")
            return InferenceResult(
                text=text,
                prompt_tokens=10,
                completion_tokens=5,
                latency_ms=100.0,
                confidence=0.85,
            )

        provider.async_infer_messages = _infer_messages
        return provider

    def test_single_conversation_two_turns(self):
        """Runner replays 2 user turns and scores each response."""
        from llmbench.core.runner_multiturn import ConversationRunner

        conv = ConversationSample(turns=[
            Turn(role="system",    content="You are helpful."),
            Turn(role="user",      content="What is 2+2?"),
            Turn(role="assistant", content="4", expected_content="4"),
            Turn(role="user",      content="And 3+3?"),
            Turn(role="assistant", content="6", expected_content="6"),
        ])
        ds = ConversationDataset(name="test", conversations=[conv])

        provider = self._mock_provider(["4", "6"])
        runner   = ConversationRunner(provider=provider, concurrency=1)
        result   = runner.run(ds, verbose=False)

        assert result.total_samples == 1
        assert result.failed_samples == 0
        # Both turns answered correctly
        sr = result.sample_results[0]
        assert sr.scores.get("turn_exact_match") == 1.0

    def test_wrong_answers_score_zero(self):
        from llmbench.core.runner_multiturn import ConversationRunner

        conv = ConversationSample(turns=[
            Turn(role="user",      content="Capital of France?"),
            Turn(role="assistant", content="Paris", expected_content="Paris"),
        ])
        ds = ConversationDataset(name="test", conversations=[conv])

        provider = self._mock_provider(["Berlin"])   # wrong answer
        runner   = ConversationRunner(provider=provider, concurrency=1)
        result   = runner.run(ds, verbose=False)

        sr = result.sample_results[0]
        assert sr.scores.get("turn_exact_match") == 0.0

    def test_confidence_propagated(self):
        from llmbench.core.runner_multiturn import ConversationRunner

        conv = ConversationSample(turns=[
            Turn(role="user",      content="Q?"),
            Turn(role="assistant", content="A.", expected_content="A."),
        ])
        ds = ConversationDataset(name="test", conversations=[conv])

        provider = self._mock_provider(["A."])
        runner   = ConversationRunner(provider=provider, concurrency=1)
        result   = runner.run(ds, verbose=False)

        sr = result.sample_results[0]
        assert sr.confidence == pytest.approx(0.85)

    def test_multiple_conversations_parallel(self):
        from llmbench.core.runner_multiturn import ConversationRunner

        convs = [
            ConversationSample(turns=[
                Turn(role="user",      content=f"Q{i}?"),
                Turn(role="assistant", content=f"A{i}", expected_content=f"A{i}"),
            ])
            for i in range(3)
        ]
        ds = ConversationDataset(name="multi", conversations=convs)

        provider = self._mock_provider([f"A{i}" for i in range(3)])
        runner   = ConversationRunner(provider=provider, concurrency=2)
        result   = runner.run(ds, verbose=False)

        assert result.total_samples == 3

    def test_result_stored_as_sample_result(self):
        """ConversationResult is converted to SampleResult for the store."""
        from llmbench.core.runner_multiturn import ConversationRunner

        conv = ConversationSample(turns=[
            Turn(role="user",      content="Q?"),
            Turn(role="assistant", content="A.", expected_content="A."),
        ])
        ds = ConversationDataset(name="test", conversations=[conv])

        provider = self._mock_provider(["A."])
        runner   = ConversationRunner(provider=provider)
        result   = runner.run(ds, verbose=False)

        assert len(result.sample_results) == 1
        assert isinstance(result.sample_results[0], SampleResult)

    def test_cli_run_conv_command_exists(self):
        """Verify the run-conv CLI command is registered."""
        from llmbench.api.cli import app
        command_names = [c.name for c in app.registered_commands]
        assert "run-conv" in command_names

    def test_cli_metrics_shows_family_column(self):
        """Sanity-check that the CLI metrics table includes multi-turn entries."""
        from typer.testing import CliRunner
        from llmbench.api.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["metrics"])
        assert result.exit_code == 0
        assert "turn_exact_match" in result.output
        assert "multi-turn" in result.output
