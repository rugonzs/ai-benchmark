"""Command-line interface for the NER comparison study."""

from __future__ import annotations

import argparse
from pathlib import Path

from ner_study.benchmark import run_benchmark
from ner_study.config import StudyConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NER comparison study benchmark runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark", help="Run the full study.")
    add_common_arguments(benchmark)
    benchmark.add_argument("--skip-lfm2-train", action="store_true")
    benchmark.set_defaults(command_fn=handle_benchmark)

    smoke = subparsers.add_parser("smoke", help="Run a tiny smoke pass with reduced sample counts.")
    add_common_arguments(smoke)
    smoke.add_argument("--smoke-samples", type=int, default=2)
    smoke.add_argument("--skip-lfm2-train", action="store_true")
    smoke.set_defaults(command_fn=handle_smoke)
    return parser


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/full"))
    parser.add_argument("--qwen-model", type=str, default=None)
    parser.add_argument("--qwen-fallback-model", type=str, default=None)
    parser.add_argument("--qwen-think", type=str, default=None, choices=["false", "low", "medium", "high"])
    parser.add_argument("--baseline-model", type=str, default=None)
    parser.add_argument("--lfm2-model", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--baseline-epochs", type=float, default=None)
    parser.add_argument("--lfm2-epochs", type=float, default=None)


def build_config_from_args(args: argparse.Namespace) -> StudyConfig:
    config = StudyConfig(output_dir=args.output_dir)
    if args.qwen_model:
        config.qwen.model_name = args.qwen_model
    if args.qwen_fallback_model:
        config.qwen.fallback_model_name = args.qwen_fallback_model
    if args.qwen_think:
        config.qwen.think_level = False if args.qwen_think == "false" else args.qwen_think
    if args.baseline_model:
        config.baseline.model_name = args.baseline_model
    if args.lfm2_model:
        config.lfm2.model_name = args.lfm2_model
    if args.max_train_samples is not None:
        config.data.max_train_samples = args.max_train_samples
    if args.max_validation_samples is not None:
        config.data.max_validation_samples = args.max_validation_samples
    if args.max_test_samples is not None:
        config.data.max_test_samples = args.max_test_samples
    if args.baseline_epochs is not None:
        config.baseline.num_train_epochs = args.baseline_epochs
    if args.lfm2_epochs is not None:
        config.lfm2.num_train_epochs = args.lfm2_epochs
    return config


def handle_benchmark(args: argparse.Namespace) -> None:
    config = build_config_from_args(args)
    run_benchmark(config=config, skip_lfm2_train=args.skip_lfm2_train)


def handle_smoke(args: argparse.Namespace) -> None:
    config = build_config_from_args(args)
    config.data.max_train_samples = args.smoke_samples
    config.data.max_validation_samples = args.smoke_samples
    config.data.max_test_samples = args.smoke_samples
    config.baseline.num_train_epochs = 1
    config.lfm2.num_train_epochs = 1
    config.lfm2.gradient_accumulation_steps = 1
    run_benchmark(config=config, skip_lfm2_train=args.skip_lfm2_train)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.command_fn(args)


if __name__ == "__main__":
    main()
