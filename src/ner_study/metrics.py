"""Common scoring utilities used across all systems."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, median

from ner_study.constants import LABELS
from ner_study.schemas import AggregateMetrics, SampleMetrics, Span


def span_key(span: Span) -> tuple[int, int, str]:
    return (span.start, span.end, span.label)


def compute_span_metrics(
    system_name: str,
    gold_batches: list[list[Span]],
    pred_batches: list[list[Span]],
    sample_metrics: list[SampleMetrics] | None = None,
    train_runtime_seconds: float | None = None,
) -> AggregateMetrics:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    exact_matches = 0
    positive_exact_matches = 0
    positive_sentence_count = 0
    empty_exact_matches = 0
    empty_sentence_count = 0
    predicted_entity_sentence_count = 0

    per_label_tp: dict[str, int] = defaultdict(int)
    per_label_fp: dict[str, int] = defaultdict(int)
    per_label_fn: dict[str, int] = defaultdict(int)

    for gold_spans, pred_spans in zip(gold_batches, pred_batches, strict=True):
        gold_keys = {span_key(span) for span in gold_spans}
        pred_keys = {span_key(span) for span in pred_spans}
        if gold_keys == pred_keys:
            exact_matches += 1
        if gold_keys:
            positive_sentence_count += 1
            if gold_keys == pred_keys:
                positive_exact_matches += 1
        else:
            empty_sentence_count += 1
            if gold_keys == pred_keys:
                empty_exact_matches += 1
        if pred_keys:
            predicted_entity_sentence_count += 1
        intersection = gold_keys & pred_keys
        true_positives += len(intersection)
        false_positives += len(pred_keys - gold_keys)
        false_negatives += len(gold_keys - pred_keys)

        for label in LABELS:
            label_gold = {key for key in gold_keys if key[2] == label}
            label_pred = {key for key in pred_keys if key[2] == label}
            label_intersection = label_gold & label_pred
            per_label_tp[label] += len(label_intersection)
            per_label_fp[label] += len(label_pred - label_gold)
            per_label_fn[label] += len(label_gold - label_pred)

    precision = safe_divide(true_positives, true_positives + false_positives)
    recall = safe_divide(true_positives, true_positives + false_negatives)
    f1 = safe_f1(precision, recall)

    per_label_f1 = {
        label: safe_f1(
            safe_divide(per_label_tp[label], per_label_tp[label] + per_label_fp[label]),
            safe_divide(per_label_tp[label], per_label_tp[label] + per_label_fn[label]),
        )
        for label in LABELS
    }

    aggregate = AggregateMetrics(
        system_name=system_name,
        micro_precision=precision,
        micro_recall=recall,
        micro_f1=f1,
        exact_match_rate=safe_divide(exact_matches, len(gold_batches)),
        positive_exact_match_rate=safe_divide(positive_exact_matches, positive_sentence_count)
        if positive_sentence_count
        else None,
        empty_exact_match_rate=safe_divide(empty_exact_matches, empty_sentence_count) if empty_sentence_count else None,
        gold_entity_sentence_count=positive_sentence_count,
        predicted_entity_sentence_count=predicted_entity_sentence_count,
        per_label_f1=per_label_f1,
        samples=len(gold_batches),
        train_runtime_seconds=train_runtime_seconds,
    )

    if sample_metrics:
        latencies = [item.latency_seconds for item in sample_metrics]
        prompt_tokens = [item.prompt_tokens for item in sample_metrics if item.prompt_tokens is not None]
        completion_tokens = [item.completion_tokens for item in sample_metrics if item.completion_tokens is not None]
        total_tokens = [item.total_tokens for item in sample_metrics if item.total_tokens is not None]
        reasoning_tokens = [item.reasoning_tokens for item in sample_metrics if item.reasoning_tokens is not None]
        memory_mb = [item.memory_rss_mb for item in sample_metrics if item.memory_rss_mb is not None]

        aggregate.latency_mean_seconds = mean(latencies)
        aggregate.latency_median_seconds = median(latencies)
        aggregate.latency_p95_seconds = percentile(latencies, 95)
        total_runtime = sum(latencies)
        aggregate.inference_runtime_seconds = total_runtime
        aggregate.throughput_samples_per_second = safe_divide(len(sample_metrics), total_runtime)
        aggregate.prompt_tokens_total = sum(prompt_tokens) if prompt_tokens else None
        aggregate.completion_tokens_total = sum(completion_tokens) if completion_tokens else None
        aggregate.total_tokens_total = sum(total_tokens) if total_tokens else None
        aggregate.reasoning_tokens_total = sum(reasoning_tokens) if reasoning_tokens else None
        aggregate.reasoning_tokens_observable = any(item.reasoning_tokens_observable for item in sample_metrics)
        aggregate.memory_peak_rss_mb = max(memory_mb) if memory_mb else None

    return aggregate


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def safe_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((p / 100) * (len(ordered) - 1)))
    return ordered[index]
