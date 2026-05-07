"""
llmbench/loaders/__init__.py
Unified dataset loading interface. All loaders normalize to EvalDataset.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Iterator

from llmbench.core.schema import EvalDataset, EvalSample, TaskType


# ---------------------------------------------------------------------------
# Field alias map — common naming variants → canonical field names
# ---------------------------------------------------------------------------

_ALIASES: dict[str, str] = {
    "question": "input", "query": "input", "prompt": "input", "text": "input",
    "answer": "expected_output", "label": "expected_output",
    "response": "expected_output", "target": "expected_output",
    "reference": "expected_output", "gold": "expected_output",
    "passages": "context", "documents": "context",
    "chunks": "context", "retrieved": "context",
}

_CORE_FIELDS = {"id", "input", "expected_output", "context", "metadata"}


def _normalize(record: dict[str, Any]) -> dict[str, Any]:
    """Apply alias map and bucket unknown keys into metadata."""
    normalized: dict[str, Any] = {}
    metadata: dict[str, Any] = {}

    for key, value in record.items():
        mapped = _ALIASES.get(key.lower(), key)
        if mapped in _CORE_FIELDS:
            normalized[mapped] = value
        else:
            metadata[key] = value

    if metadata:
        existing = normalized.get("metadata", {})
        normalized["metadata"] = {**existing, **metadata}

    return normalized


# ---------------------------------------------------------------------------
# JSON loader
# ---------------------------------------------------------------------------

def load_json(
    path: str | Path,
    task_type: str,
    name: str | None = None,
    records_key: str | None = None,
) -> EvalDataset:
    """
    Load from a JSON file (flat list or nested dict).

    Args:
        path:        Path to the .json or .jsonl file.
        task_type:   Registered task type name.
        name:        Dataset name (defaults to filename stem).
        records_key: If the list is nested: {"data": [...]}, pass "data".
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")

    # JSONL support
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        data = json.loads(raw)
        records = data[records_key] if records_key else data

    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list, got {type(records).__name__}")

    samples = [EvalSample(**_normalize(r)) for r in records]
    return EvalDataset(
        name=name or path.stem,
        task_type=task_type,
        samples=samples,
        source=str(path),
    )


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_csv(
    path: str | Path,
    task_type: str,
    input_col: str = "input",
    output_col: str = "expected_output",
    context_col: str | None = None,
    name: str | None = None,
    delimiter: str = ",",
) -> EvalDataset:
    """
    Load from a CSV/TSV file.

    Context chunks can be pipe-separated in a single cell: chunk1 ||| chunk2
    All extra columns are automatically captured in sample.metadata.
    """
    path = Path(path)
    samples: list[EvalSample] = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            context = None
            if context_col and context_col in row and row[context_col].strip():
                context = [c.strip() for c in row[context_col].split("|||")]

            reserved = {input_col, output_col, context_col} - {None}
            metadata = {k: v for k, v in row.items() if k not in reserved}

            samples.append(EvalSample(
                input=row[input_col],
                expected_output=row.get(output_col) or None,
                context=context,
                metadata=metadata,
            ))

    return EvalDataset(
        name=name or path.stem,
        task_type=task_type,
        samples=samples,
        source=str(path),
    )


# ---------------------------------------------------------------------------
# Hugging Face loader
# ---------------------------------------------------------------------------

# Pre-built field mappings for popular public datasets
_HF_PRESETS: dict[str, dict[str, Any]] = {
    "squad": {
        "input_fn":   lambda r: r["question"],
        "output_fn":  lambda r: r["answers"]["text"][0] if r["answers"]["text"] else None,
        "context_fn": lambda r: [r["context"]],
    },
    "squad_v2": {
        "input_fn":   lambda r: r["question"],
        "output_fn":  lambda r: r["answers"]["text"][0] if r["answers"]["text"] else None,
        "context_fn": lambda r: [r["context"]],
    },
    "truthful_qa": {
        "input_fn":   lambda r: r["question"],
        "output_fn":  lambda r: r["best_answer"],
        "context_fn": None,
    },
    "openai_humaneval": {
        "input_fn":   lambda r: r["prompt"],
        "output_fn":  lambda r: r["canonical_solution"],
        "context_fn": None,
    },
    "cnn_dailymail": {
        "input_fn":   lambda r: r["article"],
        "output_fn":  lambda r: r["highlights"],
        "context_fn": None,
    },
}


def load_hf(
    dataset_name: str,
    task_type: str,
    split: str = "validation",
    max_samples: int | None = None,
    input_col: str = "input",
    output_col: str = "expected_output",
    context_col: str | None = None,
    preset: str | None = None,
    hf_config: str | None = None,
) -> EvalDataset:
    """
    Load a dataset from Hugging Face Hub.

    Args:
        dataset_name: HF repo name, e.g. "squad" or "rajpurkar/squad".
        task_type:    Registered task type.
        split:        HF split ("train", "validation", "test").
        max_samples:  Cap the dataset size (useful for fast dev runs).
        preset:       Use a built-in field mapping (see _HF_PRESETS).
        hf_config:    HF dataset config name if required (e.g. "3.0.0" for CNN/DM).
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    hf_ds = hf_load(dataset_name, hf_config, split=split)
    if max_samples:
        hf_ds = hf_ds.select(range(min(max_samples, len(hf_ds))))

    # Resolve preset key: try exact name, then the last segment of the repo path
    preset_key = preset or dataset_name.split("/")[-1]
    mapping = _HF_PRESETS.get(preset_key)

    samples: list[EvalSample] = []
    for record in hf_ds:
        if mapping:
            samples.append(EvalSample(
                input=mapping["input_fn"](record),
                expected_output=mapping["output_fn"](record),
                context=mapping["context_fn"](record) if mapping["context_fn"] else None,
            ))
        else:
            # Generic column-name mode
            context = None
            if context_col and context_col in record:
                raw_ctx = record[context_col]
                context = raw_ctx if isinstance(raw_ctx, list) else [raw_ctx]

            samples.append(EvalSample(
                input=str(record[input_col]),
                expected_output=str(record[output_col]) if output_col in record else None,
                context=context,
            ))

    return EvalDataset(
        name=dataset_name,
        task_type=task_type,
        samples=samples,
        source=f"hf:{dataset_name}:{split}",
    )


# ---------------------------------------------------------------------------
# Callable / generator loader
# ---------------------------------------------------------------------------

def load_callable(
    generator: Callable[[], Iterator[dict[str, Any]]],
    task_type: str,
    name: str = "custom",
) -> EvalDataset:
    """
    Load from any Python generator that yields dicts.
    Field aliasing is applied so you can yield {"question": ..., "answer": ...}.

    Example:
        def my_db_gen():
            for row in db.execute("SELECT q, a FROM qa"):
                yield {"input": row.q, "expected_output": row.a}

        dataset = load_callable(my_db_gen, task_type="open_qa")
    """
    samples = [EvalSample(**_normalize(record)) for record in generator()]
    return EvalDataset(name=name, task_type=task_type, samples=samples, source="callable")


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_dataset(
    dataset: EvalDataset,
    strict: bool = False,
) -> list[str]:
    """
    Return a list of warning strings. Empty list = all clear.
    If strict=True, raises ValueError on the first issue instead.
    """
    from llmbench.core.registry import registry

    warnings: list[str] = []

    try:
        task_def = registry.get_task(str(dataset.task_type))
    except KeyError:
        warnings.append(f"Task '{dataset.task_type}' is not registered — skipping task-specific validation")
        task_def = None

    for i, sample in enumerate(dataset.samples):
        if not sample.input.strip():
            msg = f"Sample {i} ({sample.id}): empty input"
            if strict:
                raise ValueError(msg)
            warnings.append(msg)

        if task_def and task_def.requires_expected_output and sample.expected_output is None:
            warnings.append(f"Sample {i} ({sample.id}): task '{dataset.task_type}' "
                            f"requires expected_output but it is missing")

        if task_def and task_def.requires_context and not sample.context:
            warnings.append(f"Sample {i} ({sample.id}): task '{dataset.task_type}' "
                            f"requires context but none provided")

    return warnings


# ---------------------------------------------------------------------------
# Unified DatasetLoader facade
# ---------------------------------------------------------------------------

class DatasetLoader:
    """
    Single entry point for all dataset sources.

        loader = DatasetLoader()
        ds = loader.load("data.json",  task_type="open_qa")
        ds = loader.load("data.csv",   task_type="open_qa")
        ds = loader.load("squad",      task_type="open_qa", max_samples=100)
        ds = loader.load(my_gen_fn,    task_type="open_qa")
    """

    def load(
        self,
        source: str | Path | Callable,
        task_type: str,
        validate: bool = True,
        **kwargs,
    ) -> EvalDataset:
        if callable(source):
            dataset = load_callable(source, task_type, **kwargs)

        elif isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists():
                suffix = path.suffix.lower()
                if suffix in (".json", ".jsonl"):
                    dataset = load_json(path, task_type, **kwargs)
                elif suffix in (".csv", ".tsv"):
                    dataset = load_csv(
                        path, task_type,
                        delimiter="\t" if suffix == ".tsv" else ",",
                        **kwargs,
                    )
                else:
                    raise ValueError(f"Unsupported file extension: {suffix}")
            else:
                # Treat as a HF dataset repo name
                dataset = load_hf(str(source), task_type, **kwargs)
        else:
            raise TypeError(f"source must be a path, HF repo name, or callable — got {type(source)}")

        if validate:
            issues = validate_dataset(dataset)
            if issues:
                import warnings as _w
                for msg in issues:
                    _w.warn(f"[LLMBench] {msg}", stacklevel=2)

        return dataset


# Module-level singleton
loader = DatasetLoader()
