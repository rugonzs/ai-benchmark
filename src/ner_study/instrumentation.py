"""Runtime instrumentation helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import psutil


def current_rss_mb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


@dataclass(slots=True)
class TimedBlock:
    started_at: float
    finished_at: float
    duration_seconds: float
    rss_mb: float


@contextmanager
def timed_block() -> TimedBlock:
    start = time.perf_counter()
    holder = TimedBlock(started_at=start, finished_at=start, duration_seconds=0.0, rss_mb=current_rss_mb())
    try:
        yield holder
    finally:
        end = time.perf_counter()
        holder.finished_at = end
        holder.duration_seconds = end - start
        holder.rss_mb = current_rss_mb()

