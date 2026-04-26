"""Utilities to serialize the benchmark outputs."""

from __future__ import annotations

import json
from pathlib import Path

from ner_study.io_utils import write_json, write_text
from ner_study.schemas import AggregateMetrics, AwsInstanceEstimate


def write_metrics_json(path: Path, metrics: AggregateMetrics) -> Path:
    return write_json(path, metrics.model_dump())


def render_comparison_table(metrics: list[AggregateMetrics], cost_rows: list[dict]) -> str:
    header = [
        "| System | Micro F1 | Exact Match | Positive Exact | Mean Latency (s) | Prompt Tokens | Completion Tokens | Cost Estimate (USD) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    cost_by_system = {row["system_name"]: row["estimated_cost_usd"] for row in cost_rows}
    rows = []
    for metric in metrics:
        rows.append(
            "| {system} | {f1:.4f} | {em:.4f} | {pem} | {latency} | {prompt} | {completion} | {cost} |".format(
                system=metric.system_name,
                f1=metric.micro_f1,
                em=metric.exact_match_rate,
                pem=format_optional(metric.positive_exact_match_rate),
                latency=format_optional(metric.latency_mean_seconds),
                prompt=format_optional(metric.prompt_tokens_total),
                completion=format_optional(metric.completion_tokens_total),
                cost=format_optional(cost_by_system.get(metric.system_name)),
            )
        )
    return "\n".join(header + rows) + "\n"


def write_comparison_bundle(
    output_dir: Path,
    metrics: list[AggregateMetrics],
    cost_rows: list[dict],
) -> tuple[Path, Path]:
    comparison_json = output_dir / "comparison.json"
    comparison_md = output_dir / "comparison.md"
    write_json(
        comparison_json,
        {
            "systems": [metric.model_dump() for metric in metrics],
            "cost_estimates": cost_rows,
        },
    )
    write_text(comparison_md, render_comparison_table(metrics, cost_rows))
    return comparison_json, comparison_md


def write_cost_estimates(path: Path, estimates: list[AwsInstanceEstimate]) -> Path:
    return write_json(path, [estimate.model_dump() for estimate in estimates])


def summarize_costs_by_system(estimates: list[AwsInstanceEstimate]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for estimate in estimates:
        system_name = estimate.workload.split(":", 1)[0]
        row = grouped.setdefault(
            system_name,
            {
                "system_name": system_name,
                "estimated_cost_usd": 0.0,
                "hourly_costs_usd": [],
                "workloads": [],
            },
        )
        row["estimated_cost_usd"] += estimate.estimated_cost_usd
        row["hourly_costs_usd"].append(estimate.hourly_cost_usd)
        row["workloads"].append(estimate.model_dump())
    for row in grouped.values():
        row["estimated_cost_usd"] = round(row["estimated_cost_usd"], 6)
    return list(grouped.values())


def format_optional(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return json.dumps(value)
