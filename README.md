# 🔬 LLMBench

**Production-grade LLM evaluation harness built from scratch.**

LLMBench runs structured, reproducible evaluations across four metric families — lexical, semantic, LLM-as-judge, and calibration — stores every run, and diffs any two runs to detect regressions. Supports both single-turn and multi-turn conversation evals.

---

## Why LLMBench?

Most teams swap models or update prompts and then *eyeball* outputs. LLMBench gives you:

- **18 metrics across 4 families** — lexical, semantic, LLM-as-judge, and ECE calibration
- **Multi-turn conversation eval** — replays full chat histories, scores each turn
- **Log-prob confidence extraction** — extracts calibrated confidence from token log-probs (OpenAI, Groq, vLLM)
- **Regression tracking** — diff any two run IDs, fail CI if metrics drop > threshold
- **Full run history** — every run stored with config, scores, and per-sample outputs
- **Plugin architecture** — register custom tasks and metrics in 10 lines
- **FastAPI + CLI + Dashboard** — use it however you want

---

## Quickstart

```bash
pip install -e .
export GROQ_API_KEY=your_key

# Single-turn eval
llmbench run tasks/sample_qa.json \
    --task open_qa \
    --model groq/llama-3.3-70b-versatile \
    --metrics exact_match,f1,rouge_l,ece

# Multi-turn conversation eval
llmbench run-conv tasks/sample_conversations.json \
    --model groq/llama-3.3-70b-versatile \
    --judge groq/openai/gpt-oss-120b \
    --metrics turn_exact_match,turn_f1,conversation_coherence

# List stored runs
llmbench list

# Regression check (exits 1 if regression > 2%)
llmbench compare <baseline_run_id> <candidate_run_id>

# Per-sample inspection
llmbench show <run_id> --samples --top 10

# HTML report
python -m llmbench.api.reporter <run_id> report.html

# Dashboard
streamlit run dashboard/app.py

# REST API
uvicorn llmbench.api.rest:app --reload --port 8080
```

---

## Python API

```python
from llmbench import build_runner, loader, ModelConfig

# Single-turn
dataset = loader.load("squad", task_type="open_qa", max_samples=100)
runner  = build_runner(
    ModelConfig(provider="groq", model_id="llama-3.3-70b-versatile"),
    judge_config=ModelConfig(provider="groq", model_id="llama-3.3-70b-versatile"),
)
result = runner.run(dataset, metrics=["exact_match", "f1", "bertscore", "ece"])

# Multi-turn
from llmbench import build_conversation_runner, load_conversations_json

dataset = load_conversations_json("tasks/sample_conversations.json")
runner  = build_conversation_runner(
    ModelConfig(provider="groq", model_id="llama-3.3-70b-versatile"),
    judge_config=ModelConfig(provider="groq", model_id="llama-3.3-70b-versatile"),
)
result = runner.run(dataset, metrics=["turn_exact_match", "turn_f1", "conversation_coherence"])

# Save + compare
from llmbench.store.db import store, tracker
store.save(result)
report = tracker.compare(baseline_id, result.run_id, threshold=0.02)
print(report["regressions"])
```

---

## Metric Reference

| Metric | Family | Notes |
|--------|--------|-------|
| `exact_match` | Lexical | Case + punctuation normalised |
| `f1` | Lexical | Token-level F1 |
| `rouge_1`, `rouge_2`, `rouge_l` | Lexical | ROUGE variants |
| `bleu` | Lexical | Corpus BLEU |
| `bertscore` | Semantic | BERTScore F1 |
| `cosine_similarity` | Semantic | `all-MiniLM-L6-v2` |
| `llm_faithfulness` | LLM-judge | Requires context |
| `llm_relevance` | LLM-judge | Question vs output |
| `llm_coherence` | LLM-judge | Fluency + structure |
| `llm_code_quality` | LLM-judge | Code correctness |
| `ece` | Calibration | Expected Calibration Error from log-probs |
| `turn_exact_match` | Multi-turn | Mean EM across conversation turns |
| `turn_f1` | Multi-turn | Mean token F1 across turns |
| `turn_rouge_l` | Multi-turn | Mean ROUGE-L across turns |
| `conversation_coherence` | Multi-turn judge | LLM scores overall coherence |
| `context_retention` | Multi-turn judge | LLM scores use of prior context |

### Custom metric in 10 lines

```python
from llmbench.core.registry import registry

def avg_length(results, **_):
    avg = sum(len(r.generated_output.split()) for r in results) / len(results)
    return {"avg_output_words": avg}

registry.register_metric("avg_output_words", "Avg word count", fn=avg_length,
                          requires_expected=False)
```

---

## Supported Providers

| Provider | Slug | Log-probs | Notes |
|----------|------|-----------|-------|
| Groq | `groq/llama-3.3-70b-versatile` | ✅ | Free tier, fast |
| Groq (namespaced) | `groq/openai/gpt-oss-120b` | ✅ | Model ID with slash |
| OpenAI | `openai/gpt-4o-mini` | ✅ | |
| Anthropic | `anthropic/claude-3-5-haiku-20241022` | ❌ | No logprobs API |
| vLLM | `vllm/mistral-7b` | ✅ | Set `base_url` in extra_params |
| Ollama | `ollama/llama3` | ✅ | Same as vLLM |

---

## Conversation Dataset Formats

LLMBench auto-detects format from the JSON structure:

```jsonc
// Native (llmbench)
{"turns": [{"role": "user", "content": "Hi"},
           {"role": "assistant", "content": "Hello", "expected_content": "Hello"}]}

// ShareGPT
{"conversations": [{"from": "human", "value": "Hi"},
                   {"from": "gpt",   "value": "Hello"}]}

// OpenAI fine-tune
{"messages": [{"role": "user",      "content": "Hi"},
              {"role": "assistant", "content": "Hello"}]}
```

HuggingFace Hub datasets also supported:

```python
from llmbench.loaders.conversations import load_conversations_hf
dataset = load_conversations_hf("HuggingFaceH4/ultrachat_200k", max_samples=50)
```

---

## REST API

```
POST /runs                               Single-turn eval (async)
POST /conversations/runs                 Multi-turn eval (async)
GET  /runs/{id}/status                   Poll run status
GET  /runs                               List runs (filter by dataset/model/task)
GET  /runs/{id}                          Aggregate scores
GET  /runs/{id}/samples                  Per-sample breakdown
GET  /runs/{id}/report                   Download HTML report
GET  /conversations/runs/{id}/turns      All turn records for a multi-turn run
GET  /conversations/runs/{id}/turns/{conv_id}  One conversation's turns
POST /runs/compare                       Regression diff
GET  /runs/leaderboard/{dataset}         Model ranking
GET  /tasks                              Registered tasks
GET  /metrics                            Registered metrics
GET  /health
```

Interactive docs at `http://localhost:8080/docs`.

---

## Dashboard Tabs

| Tab | Contents |
|-----|----------|
| 📊 Overview | Run table + metric trend bar chart |
| 🔁 Compare Runs | Regression diff table + radar chart |
| 🔍 Sample Inspector | Per-sample table, confidence histogram, latency histogram |
| 💬 Conversation View | Per-turn drill-down with expandable turns + score trend line |
| 📈 Calibration | Reliability diagram (confidence vs accuracy) + ECE |
| 🏆 Leaderboard | Model ranking by any metric |

---

## CI/CD Eval Gating

```yaml
# .github/workflows/eval_ci.yml already included
# Add to your repo and set GROQ_API_KEY in GitHub Secrets.
# On every PR:
#   1. Runs unit tests (no API key needed)
#   2. Runs eval on sample dataset
#   3. Compares to cached baseline — fails PR if metric drops > 2%
#   4. Posts score table as PR comment
#   5. Uploads HTML report as build artifact
```

---

## Project Structure

```
llmbench/
├── llmbench/
│   ├── core/
│   │   ├── schema.py          EvalSample, ConversationSample, RunResult, …
│   │   ├── registry.py        Task + metric plugin registry (6 tasks, 18 metrics)
│   │   ├── runner.py          Async single-turn eval runner
│   │   └── runner_multiturn.py Async multi-turn conversation runner
│   ├── loaders/
│   │   ├── __init__.py        JSON, CSV, HF Hub, callable loaders
│   │   └── conversations.py   Conversation loaders (native/ShareGPT/OpenAI)
│   ├── providers/
│   │   ├── base.py            Abstract provider + log-prob extraction
│   │   └── anthropic_provider.py  Native Anthropic SDK implementation
│   ├── metrics/
│   │   └── __init__.py        All 18 metrics, auto-registered
│   ├── store/
│   │   └── db.py              SQLAlchemy store: runs + samples + turn_results
│   └── api/
│       ├── cli.py             Typer CLI: run, run-conv, list, compare, show, …
│       ├── rest.py            FastAPI: single-turn + conversation endpoints
│       └── reporter.py        HTML report with reliability diagram
├── dashboard/
│   └── app.py                 Streamlit: 6 tabs including Calibration + Conv view
├── tasks/
│   ├── sample_qa.json         5-sample QA test set
│   └── sample_conversations.json  3-conversation multi-turn test set
├── tests/
│   └── test_core.py           79 unit tests, all offline
└── .github/workflows/
    └── eval_ci.yml            CI eval gating pipeline
```

---

## Running Tests

```bash
pip install pytest
pytest tests/test_core.py -v   # 79 tests, no API keys needed
```