#!/usr/bin/env python3

import json
import os
from typing import Optional

__all__ = [
    "ratio_from_scalars",
    "write_candidate",
    "read_candidate",
]


def ratio_from_scalars(
    gbs_cur: float,
    t_cur: float,
    gbs_next: float,
    t_next: float,
    gns: float,
) -> float:

    if not (t_cur > 0 and t_next > 0):
        raise ValueError("iteration times must be positive")
    if not (gbs_cur > 0 and gbs_next > 0):
        raise ValueError("batch sizes must be positive")
    if gns < 0:
        raise ValueError("gns must be non-negative")
    phi_cur = (gbs_cur / t_cur) / (gbs_cur + gns)
    phi_next = (gbs_next / t_next) / (gbs_next + gns)
    return phi_next / phi_cur


def write_candidate(path: str, candidate: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as handle:
        json.dump(candidate, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_candidate(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    try:
        with open(path) as handle:
            candidate = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(candidate, dict):
        return None
    try:
        gbs = float(candidate["gbs"])
        t_next = float(candidate["t_next_sec"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (gbs > 0 and t_next > 0):
        return None
    return candidate
