"""Helpers for reading/writing benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def write_json(path: str | Path, payload: Any) -> Path:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    path_obj.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path_obj


def write_text(path: str | Path, content: str) -> Path:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    path_obj.write_text(content, encoding="utf-8")
    return path_obj

