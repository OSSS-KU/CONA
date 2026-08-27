#!/usr/bin/env python3
"""CONA's fallback for sustained GNS decreases.

Loaded by path on both sides -- the runner imports it, the training loop loads
this same file through ``CONA_FALLBACK_RULE`` -- so both apply the same rule.
"""

from __future__ import annotations

from math import isfinite
from typing import Optional

__all__ = [
    "DEFAULT_EMA_ALPHA",
    "DEFAULT_PATIENCE",
    "GnsFallbackMonitor",
    "previous_is_preferable",
    "reversal_gns_threshold",
    "throughput_of",
]

DEFAULT_EMA_ALPHA = 0.1
DEFAULT_PATIENCE = 100


def throughput_of(gbs: float, iteration_time_sec: float) -> float:
    if not (isfinite(gbs) and gbs > 0):
        raise ValueError("gbs must be finite and positive")
    if not (isfinite(iteration_time_sec) and iteration_time_sec > 0):
        raise ValueError("iteration_time_sec must be finite and positive")
    return gbs / iteration_time_sec


def reversal_gns_threshold(
    prev_gbs: float,
    prev_time_sec: float,
    cur_gbs: float,
    cur_time_sec: float,
) -> Optional[float]:
    if not (isfinite(prev_gbs) and isfinite(cur_gbs)):
        return None
    prev_throughput = throughput_of(prev_gbs, prev_time_sec)
    cur_throughput = throughput_of(cur_gbs, cur_time_sec)
    if cur_throughput <= prev_throughput:
        return None
    numerator = prev_throughput * cur_gbs - cur_throughput * prev_gbs
    if numerator <= 0:
        return None
    return numerator / (cur_throughput - prev_throughput)


def previous_is_preferable(
    prev_gbs: float,
    prev_time_sec: float,
    cur_gbs: float,
    cur_time_sec: float,
    gns: float,
) -> bool:
    if not (isfinite(gns) and gns >= 0):
        raise ValueError("gns must be finite and non-negative")
    threshold = reversal_gns_threshold(
        prev_gbs, prev_time_sec, cur_gbs, cur_time_sec
    )
    return threshold is not None and gns < threshold


class GnsFallbackMonitor:
    def __init__(
        self,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        patience: int = DEFAULT_PATIENCE,
    ):
        if not (isfinite(ema_alpha) and 0.0 < ema_alpha <= 1.0):
            raise ValueError("ema_alpha must lie in (0, 1]")
        if patience < 1:
            raise ValueError("patience must be at least 1")
        self.ema_alpha = float(ema_alpha)
        self.patience = int(patience)
        self.ema: Optional[float] = None
        self.consecutive_decreases = 0
        self.observations = 0

    def observe(self, gns: Optional[float]) -> bool:

        if gns is None:
            return False
        value = float(gns)
        if not (isfinite(value) and value >= 0):
            return False

        self.observations += 1
        previous = self.ema
        if previous is None:
            self.ema = value
            return False

        self.ema = self.ema_alpha * value + (1.0 - self.ema_alpha) * previous
        if self.ema < previous:
            self.consecutive_decreases += 1
        else:
            self.consecutive_decreases = 0
        return self.consecutive_decreases >= self.patience

    def reset(self) -> None:
        self.ema = None
        self.consecutive_decreases = 0

    def __repr__(self) -> str:
        ema = "None" if self.ema is None else f"{self.ema:.4g}"
        return (
            f"GnsFallbackMonitor(alpha={self.ema_alpha}, "
            f"patience={self.patience}, ema={ema}, "
            f"streak={self.consecutive_decreases})"
        )
