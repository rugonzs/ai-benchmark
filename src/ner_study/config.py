"""Configuration models and defaults for the study."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ner_study.constants import (
    DATASET_NAME,
    DEFAULT_AWS_REGION,
    DEFAULT_BASELINE_MODEL,
    DEFAULT_LFM2_MODEL,
    DEFAULT_QWEN_FALLBACK_MODEL,
    DEFAULT_QWEN_MODEL,
)


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str = DATASET_NAME
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    max_test_samples: int | None = None


class BaselineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = DEFAULT_BASELINE_MODEL
    output_dir: str = "artifacts/baseline"
    learning_rate: float = 2e-5
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_train_epochs: float = 3.0
    weight_decay: float = 0.01
    max_length: int = 256
    logging_steps: int = 25
    early_stopping_patience: int = 2
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    fp16: bool = False


class Lfm2Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = DEFAULT_LFM2_MODEL
    output_dir: str = "artifacts/lfm2"
    learning_rate: float = 1e-4
    train_batch_size: int = 2
    eval_batch_size: int = 2
    num_train_epochs: float = 3.0
    max_source_length: int = 512
    max_target_length: int = 256
    logging_steps: int = 10
    gradient_accumulation_steps: int = 8
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    fp16: bool = False


class QwenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = DEFAULT_QWEN_MODEL
    fallback_model_name: str = DEFAULT_QWEN_FALLBACK_MODEL
    temperature: float = 0.0
    think_level: bool | str = False
    timeout_seconds: int = 600


class CostConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aws_region: str = DEFAULT_AWS_REGION
    ec2_operating_system: str = "Linux"
    tenancy: str = "Shared"
    preinstalled_software: str = "NA"
    capacity_status: str = "Used"


class StudyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path = Path("artifacts/full")
    data: DataConfig = Field(default_factory=DataConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    lfm2: Lfm2Config = Field(default_factory=Lfm2Config)
    qwen: QwenConfig = Field(default_factory=QwenConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
