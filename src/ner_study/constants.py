"""Shared constants used across the benchmark."""

LABELS = ("PER", "ORG", "LOC", "MISC")
IOB_LABELS = (
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
)
DATASET_NAME = "tomaarsen/conll2003"
DEFAULT_BASELINE_MODEL = "bert-base-cased"
DEFAULT_LFM2_MODEL = "LiquidAI/LFM2-350M"
DEFAULT_QWEN_MODEL = "qwen3.6:35b-a3b-mxfp8"
DEFAULT_QWEN_FALLBACK_MODEL = "qwen3:30b"
DEFAULT_AWS_REGION = "us-east-1"
