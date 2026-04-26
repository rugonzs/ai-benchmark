"""Pydantic models shared by all benchmark stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ner_study.constants import LABELS


class Span(BaseModel):
    """Character-level span in the reconstructed sentence."""

    model_config = ConfigDict(extra="forbid")

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str
    label: str

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        if value not in LABELS:
            raise ValueError(f"Unsupported label: {value}")
        return value

    @field_validator("end")
    @classmethod
    def validate_offsets(cls, value: int, info: Any) -> int:
        start = info.data.get("start")
        if start is not None and value <= start:
            raise ValueError("Span end must be greater than start.")
        return value


class SentenceExample(BaseModel):
    """Single sentence from CoNLL with multiple synchronized representations."""

    model_config = ConfigDict(extra="forbid")

    split: str
    id: str
    tokens: list[str]
    ner_tags: list[str]
    sentence: str
    spans: list[Span]


class SampleMetrics(BaseModel):
    """Per-sample runtime and generation metrics."""

    model_config = ConfigDict(extra="allow")

    sample_id: str
    latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    reasoning_tokens_observable: bool = False
    prompt_eval_duration_seconds: float | None = None
    eval_duration_seconds: float | None = None
    load_duration_seconds: float | None = None
    memory_rss_mb: float | None = None
    raw_output_path: str | None = None


class AggregateMetrics(BaseModel):
    """Common accuracy and runtime metrics."""

    model_config = ConfigDict(extra="allow")

    system_name: str
    micro_precision: float
    micro_recall: float
    micro_f1: float
    exact_match_rate: float
    positive_exact_match_rate: float | None = None
    empty_exact_match_rate: float | None = None
    gold_entity_sentence_count: int | None = None
    predicted_entity_sentence_count: int | None = None
    per_label_f1: dict[str, float]
    samples: int
    latency_mean_seconds: float | None = None
    latency_median_seconds: float | None = None
    latency_p95_seconds: float | None = None
    throughput_samples_per_second: float | None = None
    prompt_tokens_total: int | None = None
    completion_tokens_total: int | None = None
    total_tokens_total: int | None = None
    reasoning_tokens_total: int | None = None
    reasoning_tokens_observable: bool = False
    memory_peak_rss_mb: float | None = None
    train_runtime_seconds: float | None = None
    inference_runtime_seconds: float | None = None


class AwsInstanceEstimate(BaseModel):
    """AWS cost estimate for a single workload profile."""

    model_config = ConfigDict(extra="forbid")

    workload: str
    instance_type: str
    accelerator: str
    hourly_cost_usd: float
    estimated_runtime_hours: float
    estimated_cost_usd: float
    notes: str = ""


class BenchmarkArtifacts(BaseModel):
    """Paths written during a benchmark run."""

    model_config = ConfigDict(extra="forbid")

    output_dir: Path
    metrics_json: Path
    metrics_md: Path
    comparison_json: Path
    comparison_md: Path
