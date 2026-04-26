"""Benchmark orchestration."""

from __future__ import annotations

from pathlib import Path

from ner_study.aws_pricing import make_estimate, ratio_against_minimum
from ner_study.baseline_bert import train_baseline
from ner_study.config import StudyConfig
from ner_study.data import examples_to_dicts, load_sentence_examples
from ner_study.io_utils import ensure_dir, write_json
from ner_study.lfm2 import train_lfm2
from ner_study.qwen_runner import run_qwen
from ner_study.reporting import summarize_costs_by_system, write_comparison_bundle, write_cost_estimates, write_metrics_json
from ner_study.schemas import AggregateMetrics, AwsInstanceEstimate


def run_benchmark(config: StudyConfig, skip_lfm2_train: bool = False) -> dict[str, Path]:
    output_dir = ensure_dir(config.output_dir)
    dataset_dir = ensure_dir(output_dir / "dataset")
    split_examples = load_sentence_examples(config.data)
    for split_name, examples in split_examples.items():
        write_json(dataset_dir / f"{split_name}.json", examples_to_dicts(examples))

    metrics: list[AggregateMetrics] = []
    estimates: list[AwsInstanceEstimate] = []

    baseline_dir = ensure_dir(output_dir / "baseline")
    baseline_metrics, baseline_predictions_path = train_baseline(
        train_examples=split_examples["train"],
        validation_examples=split_examples["validation"],
        test_examples=split_examples["test"],
        config=config.baseline,
        output_dir=baseline_dir,
    )
    metrics.append(baseline_metrics)
    estimates.extend(estimate_costs_for_system("baseline_bert", baseline_metrics, config))
    write_metrics_json(baseline_dir / "metrics.json", baseline_metrics)

    if not skip_lfm2_train:
        lfm2_dir = ensure_dir(output_dir / "lfm2")
        lfm2_metrics, lfm2_predictions_path = train_lfm2(
            train_examples=split_examples["train"],
            validation_examples=split_examples["validation"],
            test_examples=split_examples["test"],
            config=config.lfm2,
            output_dir=lfm2_dir,
        )
        metrics.append(lfm2_metrics)
        estimates.extend(estimate_costs_for_system("lfm2_350m_sft", lfm2_metrics, config))
        write_metrics_json(lfm2_dir / "metrics.json", lfm2_metrics)
    else:
        lfm2_predictions_path = output_dir / "lfm2" / "lfm2_predictions.json"

    qwen_dir = ensure_dir(output_dir / "qwen")
    qwen_metrics, qwen_predictions_path, qwen_model_name = run_qwen(
        examples=split_examples["test"],
        config=config.qwen,
        output_dir=qwen_dir,
    )
    metrics.append(qwen_metrics)
    estimates.extend(estimate_costs_for_system(qwen_metrics.system_name, qwen_metrics, config))
    write_metrics_json(qwen_dir / "metrics.json", qwen_metrics)
    write_json(qwen_dir / "model_used.json", {"model_name": qwen_model_name})

    write_cost_estimates(output_dir / "cost_estimates.json", estimates)
    cost_rows = summarize_costs_by_system(estimates)
    add_cost_ratios(cost_rows)
    comparison_json, comparison_md = write_comparison_bundle(output_dir, metrics, cost_rows)
    return {
        "output_dir": output_dir,
        "comparison_json": comparison_json,
        "comparison_md": comparison_md,
        "baseline_predictions": baseline_predictions_path,
        "lfm2_predictions": lfm2_predictions_path,
        "qwen_predictions": qwen_predictions_path,
    }


def estimate_costs_for_system(system_name: str, metrics: AggregateMetrics, config: StudyConfig) -> list[AwsInstanceEstimate]:
    runtime_candidates: list[AwsInstanceEstimate] = []
    if system_name == "baseline_bert":
        if metrics.train_runtime_seconds is not None:
            runtime_candidates.append(
                make_estimate(
                    workload="baseline_train",
                    instance_type="c7i.xlarge",
                    accelerator="CPU",
                    runtime_seconds=metrics.train_runtime_seconds,
                    config=config.cost,
                    notes="Conservative CPU estimate for bert-base-cased fine-tuning on CoNLL-2003.",
                )
            )
        if metrics.inference_runtime_seconds is not None:
            runtime_candidates.append(
                make_estimate(
                    workload="baseline_inference",
                    instance_type="c7i.large",
                    accelerator="CPU",
                    runtime_seconds=metrics.inference_runtime_seconds,
                    config=config.cost,
                    notes="CPU inference is sufficient for bert-base-sized NER.",
                )
            )
    elif system_name == "lfm2_350m_sft":
        if metrics.train_runtime_seconds is not None:
            runtime_candidates.append(
                make_estimate(
                    workload="lfm2_train",
                    instance_type="g5.xlarge",
                    accelerator="NVIDIA A10G 24GB",
                    runtime_seconds=metrics.train_runtime_seconds,
                    config=config.cost,
                    notes="Single-GPU LoRA fine-tuning estimate for LFM2-350M.",
                )
            )
        if metrics.inference_runtime_seconds is not None:
            runtime_candidates.append(
                make_estimate(
                    workload="lfm2_inference",
                    instance_type="g5.xlarge",
                    accelerator="NVIDIA A10G 24GB",
                    runtime_seconds=metrics.inference_runtime_seconds,
                    config=config.cost,
                    notes="Single-GPU generative extraction estimate for LFM2-350M.",
                )
            )
    elif system_name.startswith("qwen_ollama_") and metrics.inference_runtime_seconds is not None:
        if "qwen3_30b" in system_name:
            runtime_candidates.append(
                make_estimate(
                    workload="qwen_inference",
                    instance_type="g5.xlarge",
                    accelerator="NVIDIA A10G 24GB",
                    runtime_seconds=metrics.inference_runtime_seconds,
                    config=config.cost,
                    notes=(
                        "Inference estimate for the quantized Ollama qwen3:30b run. "
                        "Uses a single A10G 24GB profile to keep serving costs aligned "
                        "with the actual 18GB local model artifact."
                    ),
                )
            )
        else:
            runtime_candidates.append(
                make_estimate(
                    workload="qwen_inference",
                    instance_type="g6e.12xlarge",
                    accelerator="NVIDIA L40S 48GB",
                    runtime_seconds=metrics.inference_runtime_seconds,
                    config=config.cost,
                    notes="Conservative single-GPU inference estimate for larger Qwen variants such as Qwen3.6-35B-A3B MXFP8.",
                )
            )

    for item in runtime_candidates:
        item.workload = f"{system_name}:{item.workload}"
    return runtime_candidates


def add_cost_ratios(rows: list[dict]) -> list[dict]:
    costs = [row["estimated_cost_usd"] for row in rows]
    ratios = ratio_against_minimum(costs)
    for row, ratio in zip(rows, ratios, strict=True):
        row["cost_ratio_vs_cheapest"] = ratio
    return rows
