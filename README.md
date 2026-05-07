# 🔬 LLMBench

**LLM evaluation harness built from scratch.**

LLMBench runs structured, reproducible evaluations across four metric families — lexical, semantic, LLM-as-judge, and calibration — stores every run, and diffs any two runs to surface regressions. Plug in any task type or metric in under 10 lines.

---

## Why LLMBench?

Most teams swap models or update prompts and then *eyeball* outputs to decide if things got better. LLMBench gives you:

- **Four metric families** — not just ROUGE, but semantic similarity, LLM-as-judge scoring, and calibration (ECE)
- **Regression tracking** — diff any two run IDs, get a structured report of what regressed
- **Full run history** — every run stored with config, scores, and per-sample outputs
- **Plugin architecture** — register custom tasks and metrics in 10 lines
- **Any provider** — OpenAI, Groq, Anthropic, vLLM, Ollama via one unified interface
- **FastAPI + CLI + Dashboard** — use it however you want

---

## Quickstart

```bash
pip install -e .

# Set your provider API key
export GROQ_API_KEY=your_key_here

# Run an eval
llmbench run tasks/sample_qa.json \
    --task open_qa \
    --model groq/llama-3.3-70b-versatile \
    --metrics exact_match,f1,rouge_l

# List all stored runs
llmbench list

# Compare two runs (regression check)
llmbench compare <baseline_run_id> <candidate_run_id>

# Launch dashboard
streamlit run dashboard/app.py
```

---

## Python API

```python
from llmbench import build_runner, loader, ModelConfig

# Load a dataset (JSON, CSV, HF Hub, or callable)
dataset = loader.load("squad", task_type="open_qa", max_samples=100)

# Build a runner
runner = build_runner(
    ModelConfig(provider="groq", model_id="llama-3.3-70b-versatile"),
    judge_config=ModelConfig(provider="groq", model_id="llama-3.3-70b-versatile"),
)

# Run the eval
result = runner.run(dataset, metrics=["exact_match", "f1", "bertscore", "llm_relevance"])
print(result.aggregate_scores)

# Save and compare
from llmbench.store.db import store, tracker
store.save(result)

# Later: compare to a previous run
report = tracker.compare(baseline_run_id, result.run_id, threshold=0.02)
print(report["regressions"])   # metrics that dropped > 2%
```

---

## Metric Reference

| Metric | Family | Notes |
|--------|--------|-------|
| `exact_match` | Lexical | Case/punct normalised |
| `f1` | Lexical | Token-level F1 |
| `rouge_1`, `rouge_2`, `rouge_l` | Lexical | ROUGE variants |
| `bleu` | Lexical | Corpus BLEU |
| `bertscore` | Semantic | BERTScore F1 |
| `cosine_similarity` | Semantic | `all-MiniLM-L6-v2` |
| `llm_faithfulness` | LLM-judge | Requires context |
| `llm_relevance` | LLM-judge | Question vs output |
| `llm_coherence` | LLM-judge | Fluency + structure |
| `llm_code_quality` | LLM-judge | Code correctness |
| `ece` | Calibration | Requires confidence scores |

### Register a custom metric

```python
from llmbench.core.registry import registry

def my_length_metric(results, **_):
    avg_len = sum(len(r.generated_output.split()) for r in results) / len(results)
    return {"avg_output_length": avg_len}

registry.register_metric(
    "avg_output_length",
    "Average word count of generated outputs",
    fn=my_length_metric,
    requires_expected=False,
)
```

---

## Supported Datasets

| Source | Example |
|--------|---------|
| Local JSON/JSONL | `loader.load("data.json", task_type="open_qa")` |
| Local CSV/TSV | `loader.load("data.csv", task_type="open_qa")` |
| HF Hub (preset) | `loader.load("squad", task_type="open_qa", preset="squad")` |
| HF Hub (custom cols) | `loader.load("my/repo", task_type="open_qa", input_col="q", output_col="a")` |
| Python callable | `loader.load(my_generator_fn, task_type="open_qa")` |

Field aliases are resolved automatically: `question/query/prompt` → `input`, `answer/label/target` → `expected_output`.

---

## Supported Providers

| Provider | Slug format | Notes |
|----------|-------------|-------|
| Groq | `groq/llama-3.3-70b-versatile` | Recommended (fast + free tier) |
| OpenAI | `openai/gpt-4o-mini` | |
| Anthropic | `anthropic/claude-3-5-haiku-20241022` | |
| vLLM | `vllm/my-model` | Set `base_url` in extra_params |
| Ollama | `ollama/llama3` | Same as vLLM |

---

## CLI Reference

```
llmbench run      --task --model [--judge] [--metrics] [--max-samples] [--tag key=val]
llmbench list     [--dataset] [--model] [--task] [--limit]
llmbench compare  <baseline_id> <candidate_id> [--threshold]
llmbench show     <run_id> [--samples] [--top]
llmbench metrics  List all registered metrics
llmbench tasks    List all registered task types
```

---

## Project Structure

```
llmbench/
├── llmbench/
│   ├── core/
│   │   ├── schema.py       # EvalSample, EvalDataset, RunResult, ModelConfig
│   │   ├── registry.py     # Task + metric plugin registry
│   │   └── runner.py       # Async batch eval runner
│   ├── loaders/            # JSON, CSV, HF Hub, callable loaders
│   ├── providers/          # OpenAI, Groq, vLLM provider abstraction
│   ├── metrics/            # Lexical, semantic, LLM-judge, calibration
│   ├── store/              # SQLAlchemy results store + regression tracker
│   └── api/                # FastAPI REST + Typer CLI
├── dashboard/              # Streamlit dashboard
├── tasks/                  # Built-in task YAML configs + sample datasets
└── tests/                  # Pytest unit tests
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

All tests run without API keys — providers are mocked at the metric layer.

---

## Roadmap

- [ ] Confidence score extraction from log-probs (for ECE on open-source models)
- [ ] HTML report export (Jinja2 template)
- [ ] GitHub Actions workflow for CI eval gating
- [ ] Multi-turn conversation eval support
- [ ] Async FastAPI REST endpoint
