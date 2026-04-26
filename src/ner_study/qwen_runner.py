"""Prompt-only evaluation for Qwen through local Ollama."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import ollama

from ner_study.config import QwenConfig
from ner_study.instrumentation import current_rss_mb, timed_block
from ner_study.io_utils import ensure_dir, write_json, write_text
from ner_study.metrics import compute_span_metrics
from ner_study.prompts import SYSTEM_PROMPT, build_extraction_prompt
from ner_study.schemas import AggregateMetrics, SampleMetrics, SentenceExample
from ner_study.span_utils import parse_json_text_label_spans


QWEN_TEXT_LABEL_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["text", "label"],
        "properties": {
            "text": {"type": "string"},
            "label": {"type": "string", "enum": ["PER", "ORG", "LOC", "MISC"]},
        },
        "additionalProperties": False,
    },
}


def ensure_ollama_model(model_name: str, fallback_model_name: str | None = None) -> str:
    show = subprocess.run(["ollama", "show", model_name], capture_output=True, text=True)
    if show.returncode == 0:
        return model_name
    pull = subprocess.run(["ollama", "pull", model_name], capture_output=True, text=True)
    if pull.returncode == 0:
        return model_name
    if fallback_model_name:
        fallback_show = subprocess.run(["ollama", "show", fallback_model_name], capture_output=True, text=True)
        if fallback_show.returncode == 0:
            return fallback_model_name
    raise RuntimeError(f"Unable to use requested Ollama model {model_name!r}.")


def run_qwen(
    examples: list[SentenceExample],
    config: QwenConfig,
    output_dir: Path,
) -> tuple[AggregateMetrics, Path, str]:
    model_name = ensure_ollama_model(config.model_name, config.fallback_model_name)
    raw_dir = ensure_dir(output_dir / "raw_outputs")
    checkpoint_path = output_dir / "qwen_predictions.jsonl"
    checkpoint_rows = read_checkpoint_rows(checkpoint_path)

    predictions = []
    sample_metrics = []
    serializable_rows = []

    for example in examples:
        checkpoint_row = checkpoint_rows.get(example.id)
        if checkpoint_row is not None:
            serializable_rows.append(checkpoint_row)
            predictions.append(spans_from_row(checkpoint_row))
            sample_metrics.append(sample_metrics_from_row(checkpoint_row))
            continue

        prompt = build_extraction_prompt(example.sentence)
        with timed_block() as timer:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                think=config.think_level,
                format=QWEN_TEXT_LABEL_SCHEMA,
                options={"temperature": config.temperature},
                keep_alive="30m",
            )
        raw_content = response.message.content.strip()
        raw_thinking = getattr(response.message, "thinking", None)
        spans, parse_errors = parse_json_text_label_spans(raw_content, example.sentence)
        predictions.append(spans)

        raw_output_path = raw_dir / f"{example.id}.txt"
        rendered = raw_content
        if raw_thinking:
            rendered += "\n\nTHINKING_VISIBLE_BUT_TOKEN_COUNT_NOT_OBSERVABLE:\n" + raw_thinking
        if parse_errors:
            rendered += "\n\nPARSE_ERRORS:\n" + "\n".join(parse_errors)
        write_text(raw_output_path, rendered)

        prompt_tokens = getattr(response, "prompt_eval_count", None)
        completion_tokens = getattr(response, "eval_count", None)
        total_tokens = None
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = int(prompt_tokens) + int(completion_tokens)

        metric = SampleMetrics(
            sample_id=example.id,
            latency_seconds=timer.duration_seconds,
            prompt_tokens=int(prompt_tokens) if prompt_tokens is not None else None,
            completion_tokens=int(completion_tokens) if completion_tokens is not None else None,
            total_tokens=total_tokens,
            reasoning_tokens=None,
            reasoning_tokens_observable=False,
            prompt_eval_duration_seconds=nanos_to_seconds(getattr(response, "prompt_eval_duration", None)),
            eval_duration_seconds=nanos_to_seconds(getattr(response, "eval_duration", None)),
            load_duration_seconds=nanos_to_seconds(getattr(response, "load_duration", None)),
            memory_rss_mb=current_rss_mb(),
            raw_output_path=str(raw_output_path),
        )
        sample_metrics.append(metric)

        row = {
            "sample_id": example.id,
            "sentence": example.sentence,
            "gold_spans": [span.model_dump() for span in example.spans],
            "predicted_spans": [span.model_dump() for span in spans],
            "parse_errors": parse_errors,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "thinking_visible": raw_thinking is not None,
            "sample_metrics": metric.model_dump(),
        }
        serializable_rows.append(row)
        append_checkpoint_row(checkpoint_path, row)

    metrics = compute_span_metrics(
        system_name=f"qwen_ollama_{model_name.replace(':', '_').replace('/', '_')}",
        gold_batches=[example.spans for example in examples],
        pred_batches=predictions,
        sample_metrics=sample_metrics,
    )
    predictions_path = output_dir / "qwen_predictions.json"
    write_json(predictions_path, serializable_rows)
    return metrics, predictions_path, model_name


def nanos_to_seconds(value: int | None) -> float | None:
    if value is None:
        return None
    return value / 1_000_000_000


def read_checkpoint_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["sample_id"]] = row
    return rows


def append_checkpoint_row(path: Path, row: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=True) + "\n")


def spans_from_row(row: dict) -> list:
    from ner_study.schemas import Span

    return [Span.model_validate(span) for span in row.get("predicted_spans", [])]


def sample_metrics_from_row(row: dict) -> SampleMetrics:
    metrics = row.get("sample_metrics")
    if metrics is not None:
        return SampleMetrics.model_validate(metrics)
    return SampleMetrics(
        sample_id=row["sample_id"],
        latency_seconds=0.0,
        prompt_tokens=row.get("prompt_tokens"),
        completion_tokens=row.get("completion_tokens"),
        total_tokens=(row.get("prompt_tokens") or 0) + (row.get("completion_tokens") or 0),
    )
