"""
llmbench/metrics/__init__.py
All metric families — auto-registered into the global registry.

  Lexical      : exact_match, f1, rouge_1, rouge_2, rouge_l, bleu
  Semantic     : bertscore, cosine_similarity
  LLM-judge    : llm_faithfulness, llm_relevance, llm_coherence, llm_code_quality
  Calibration  : ece  (uses log-prob confidence extracted by provider)
  Multi-turn   : turn_exact_match, turn_f1, turn_rouge_l,
                 conversation_coherence, context_retention
"""

from __future__ import annotations

import math
import string
from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llmbench.core.schema import SampleResult, TurnResult


# ---------------------------------------------------------------------------
# Shared text helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _token_f1(pred: str, gold: str) -> float:
    pred_toks = _norm(pred).split()
    gold_toks = _norm(gold).split()
    common    = Counter(pred_toks) & Counter(gold_toks)
    n_same    = sum(common.values())
    if n_same == 0:
        return 0.0
    p = n_same / len(pred_toks)
    r = n_same / len(gold_toks)
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Lexical metrics
# ---------------------------------------------------------------------------

def _exact_match(results: list["SampleResult"], **_) -> dict[str, float]:
    scores = [
        float(_norm(r.generated_output) == _norm(r.expected_output))
        for r in results
        if not r.error and r.expected_output is not None
    ]
    return {"exact_match": sum(scores) / len(scores) if scores else 0.0}


def _f1(results: list["SampleResult"], **_) -> dict[str, float]:
    scores = [
        _token_f1(r.generated_output, r.expected_output)
        for r in results
        if not r.error and r.expected_output is not None
    ]
    return {"f1": sum(scores) / len(scores) if scores else 0.0}


def _rouge(results, variant="rougeL", **_) -> dict[str, float]:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError("pip install rouge-score")
    scorer = rouge_scorer.RougeScorer([variant], use_stemmer=True)
    scores = [
        scorer.score(r.expected_output, r.generated_output)[variant].fmeasure
        for r in results
        if not r.error and r.expected_output is not None
    ]
    key = variant.lower().replace("rouge", "rouge_")
    return {key: sum(scores) / len(scores) if scores else 0.0}


def _bleu(results, **_) -> dict[str, float]:
    try:
        import evaluate as hf_eval
    except ImportError:
        raise ImportError("pip install evaluate")
    bleu  = hf_eval.load("bleu")
    valid = [(r.generated_output, r.expected_output)
             for r in results if not r.error and r.expected_output]
    if not valid:
        return {"bleu": 0.0}
    preds, refs = zip(*valid)
    result = bleu.compute(predictions=list(preds), references=[[r] for r in refs])
    return {"bleu": result["bleu"]}


# ---------------------------------------------------------------------------
# Semantic metrics
# ---------------------------------------------------------------------------

def _bertscore(results, lang="en", **_) -> dict[str, float]:
    try:
        from bert_score import score as bs
    except ImportError:
        raise ImportError("pip install bert-score")
    valid = [(r.generated_output, r.expected_output)
             for r in results if not r.error and r.expected_output]
    if not valid:
        return {"bertscore_f1": 0.0}
    preds, refs = zip(*valid)
    _, _, f1 = bs(list(preds), list(refs), lang=lang, verbose=False)
    return {"bertscore_f1": float(f1.mean())}


def _cosine_similarity(results, **_) -> dict[str, float]:
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        raise ImportError("pip install sentence-transformers")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    valid = [(r.generated_output, r.expected_output)
             for r in results if not r.error and r.expected_output]
    if not valid:
        return {"cosine_similarity": 0.0}
    preds, refs = zip(*valid)
    pe  = model.encode(list(preds), normalize_embeddings=True)
    re_ = model.encode(list(refs),  normalize_embeddings=True)
    return {"cosine_similarity": float((pe * re_).sum(axis=1).mean())}


# ---------------------------------------------------------------------------
# Calibration: ECE — now uses provider-extracted confidence         ← UPDATED
# ---------------------------------------------------------------------------

def _ece(results: list["SampleResult"], n_bins: int = 10, **_) -> dict[str, float]:
    """
    Expected Calibration Error using log-prob confidence scores.

    Confidence is extracted directly from token log-probs by the provider
    (stored in SampleResult.confidence). Correctness = exact match.

    ECE near 0  → well calibrated.
    ECE near 1  → maximally miscalibrated.
    Returns NaN if fewer than n_bins samples have confidence scores.
    """
    import numpy as np

    valid = [
        r for r in results
        if not r.error
        and r.expected_output is not None
        and r.confidence is not None
    ]

    if len(valid) < n_bins:
        return {"ece": float("nan")}

    confidences = np.array([r.confidence for r in valid])
    correctness = np.array([
        float(_norm(r.generated_output) == _norm(r.expected_output))
        for r in valid
    ])

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n   = len(valid)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi if hi < 1.0 else confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc  = correctness[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)

    return {"ece": float(ece)}


# ---------------------------------------------------------------------------
# LLM-as-judge metrics
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are a precise evaluation judge. Score the output on the requested "
    "dimension. Respond ONLY with valid JSON — no markdown, no preamble."
)

_JUDGE_PROMPTS: dict[str, str] = {
    "llm_faithfulness": (
        "Given the CONTEXT and MODEL OUTPUT, score faithfulness "
        "(1=hallucinated, 5=fully grounded).\n"
        "Context: {context}\nModel output: {generated}\n"
        'Return JSON: {{"score": <1-5>, "reasoning": "<one sentence>"}}'
    ),
    "llm_relevance": (
        "Given the QUESTION and MODEL OUTPUT, score relevance "
        "(1=irrelevant, 5=perfectly relevant).\n"
        "Question: {input}\nModel output: {generated}\n"
        'Return JSON: {{"score": <1-5>, "reasoning": "<one sentence>"}}'
    ),
    "llm_coherence": (
        "Score coherence and fluency of the MODEL OUTPUT "
        "(1=incoherent, 5=perfectly clear).\n"
        "Model output: {generated}\n"
        'Return JSON: {{"score": <1-5>, "reasoning": "<one sentence>"}}'
    ),
    "llm_code_quality": (
        "Score correctness and quality of the generated CODE "
        "(1=broken, 5=excellent).\n"
        "Code: {generated}\nReference: {expected}\n"
        'Return JSON: {{"score": <1-5>, "reasoning": "<one sentence>"}}'
    ),
    "conversation_coherence": (
        "Score the overall coherence of this ASSISTANT RESPONSE within "
        "the conversation (1=incoherent/off-topic, 5=perfectly coherent).\n"
        "Conversation so far: {context}\n"
        "Latest assistant response: {generated}\n"
        'Return JSON: {{"score": <1-5>, "reasoning": "<one sentence>"}}'
    ),
    "context_retention": (
        "Score how well the ASSISTANT RESPONSE retains and correctly uses "
        "information from earlier in the conversation "
        "(1=ignores prior context, 5=perfectly consistent).\n"
        "Full conversation history: {context}\n"
        "Latest assistant response: {generated}\n"
        'Return JSON: {{"score": <1-5>, "reasoning": "<one sentence>"}}'
    ),
}


async def _judge_one(metric: str, result, judge_provider) -> tuple[float, str]:
    import json as _json
    prompt = _JUDGE_PROMPTS[metric].format(
        input=getattr(result, "sample_input", ""),
        generated=result.generated_output,
        expected=result.expected_output or "",
        context=(
            "\n".join(result.context)
            if getattr(result, "context", None)
            else ""
        ),
    )
    inf = await judge_provider.async_infer(prompt, system=_JUDGE_SYSTEM)
    if inf.error:
        return 0.0, f"judge error: {inf.error}"
    try:
        data  = _json.loads(inf.text)
        score = (float(data.get("score", 1)) - 1) / 4   # normalise 1-5 → 0-1
        return score, data.get("reasoning", "")
    except Exception:
        return 0.0, f"parse error: {inf.text[:120]}"


def _make_judge_fn(metric: str):
    def _fn(results, judge_provider=None, **_):
        import asyncio
        if judge_provider is None:
            raise ValueError(f"Metric '{metric}' requires judge_provider")

        async def _batch():
            return await asyncio.gather(*[
                _judge_one(metric, r, judge_provider)
                for r in results if not r.error
            ])

        try:
            pairs = asyncio.run(_batch())
        except RuntimeError:
            import nest_asyncio; nest_asyncio.apply()
            pairs = asyncio.get_event_loop().run_until_complete(_batch())

        scores = [s for s, _ in pairs]
        for r, (_, reasoning) in zip([r for r in results if not r.error], pairs):
            r.judge_reasoning = reasoning
        return {metric: sum(scores) / len(scores) if scores else 0.0}

    _fn.__name__ = metric
    return _fn


# ---------------------------------------------------------------------------
# Multi-turn metrics                                                 ← NEW
# ---------------------------------------------------------------------------

def _turn_exact_match(results: list["SampleResult"], **_) -> dict[str, float]:
    """
    For multi-turn runs, SampleResult.scores already contains per-turn
    exact_match values (set by ConversationRunner). We aggregate here.
    """
    per_turn = [
        r.scores.get("turn_exact_match", float("nan"))
        for r in results if not r.error
    ]
    valid = [s for s in per_turn if not math.isnan(s)]
    return {"turn_exact_match": sum(valid) / len(valid) if valid else 0.0}


def _turn_f1(results: list["SampleResult"], **_) -> dict[str, float]:
    per_turn = [
        r.scores.get("turn_f1", float("nan"))
        for r in results if not r.error
    ]
    valid = [s for s in per_turn if not math.isnan(s)]
    return {"turn_f1": sum(valid) / len(valid) if valid else 0.0}


def _turn_rouge_l(results: list["SampleResult"], **_) -> dict[str, float]:
    per_turn = [
        r.scores.get("turn_rouge_l", float("nan"))
        for r in results if not r.error
    ]
    valid = [s for s in per_turn if not math.isnan(s)]
    return {"turn_rouge_l": sum(valid) / len(valid) if valid else 0.0}


# ---------------------------------------------------------------------------
# Auto-register all metrics
# ---------------------------------------------------------------------------

def _register_all() -> None:
    from llmbench.core.registry import registry

    # Lexical
    registry.register_metric("exact_match", "Exact string match (normalised)",
                              _exact_match)
    registry.register_metric("f1", "Token-level F1",
                              _f1)
    registry.register_metric("rouge_1", "ROUGE-1 F-measure",
                              lambda r, **kw: _rouge(r, "rouge1", **kw))
    registry.register_metric("rouge_2", "ROUGE-2 F-measure",
                              lambda r, **kw: _rouge(r, "rouge2", **kw))
    registry.register_metric("rouge_l", "ROUGE-L F-measure",
                              lambda r, **kw: _rouge(r, "rougeL", **kw))
    registry.register_metric("bleu", "Corpus BLEU",
                              _bleu)

    # Semantic
    registry.register_metric("bertscore", "BERTScore F1",
                              _bertscore)
    registry.register_metric("cosine_similarity",
                              "Sentence-transformer cosine similarity",
                              _cosine_similarity)

    # LLM-as-judge (single-turn)
    for name in ["llm_faithfulness", "llm_relevance",
                 "llm_coherence", "llm_code_quality"]:
        registry.register_metric(
            name,
            f"LLM-as-judge: {name.replace('llm_', '').replace('_', ' ')}",
            _make_judge_fn(name),
            requires_expected="code" in name,
            requires_context="faithfulness" in name,
        )

    # Calibration
    registry.register_metric(
        "ece",
        "Expected Calibration Error from log-prob confidence (lower=better)",
        _ece,
    )

    # Multi-turn aggregation metrics
    registry.register_metric("turn_exact_match",
                              "Mean exact match across conversation turns",
                              _turn_exact_match)
    registry.register_metric("turn_f1",
                              "Mean token F1 across conversation turns",
                              _turn_f1)
    registry.register_metric("turn_rouge_l",
                              "Mean ROUGE-L across conversation turns",
                              _turn_rouge_l)

    # LLM-as-judge: multi-turn specific
    for name in ["conversation_coherence", "context_retention"]:
        registry.register_metric(
            name,
            f"LLM-as-judge (multi-turn): {name.replace('_', ' ')}",
            _make_judge_fn(name),
            requires_expected=False,
            requires_context=True,
        )


_register_all()
