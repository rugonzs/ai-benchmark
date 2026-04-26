"""Shared helpers for model execution."""

from __future__ import annotations

from collections.abc import Iterable
import os

import torch


def get_torch_device() -> torch.device:
    requested = os.environ.get("NER_STUDY_TORCH_DEVICE", "cpu").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("NER_STUDY_TORCH_DEVICE=cuda but CUDA is not available.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("NER_STUDY_TORCH_DEVICE=mps but MPS is not available.")
        return torch.device("mps")
    return torch.device("cpu")


def get_torch_dtype(device: torch.device) -> torch.dtype:
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved: dict = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def batched(items: list, batch_size: int) -> Iterable[list]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]
