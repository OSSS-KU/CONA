#!/usr/bin/env python3

import math
from typing import Optional


def linear_scale_lr(base_lr: float, base_gbs: int, new_gbs: int) -> float:

    if base_gbs <= 0 or new_gbs <= 0:
        return base_lr

    return base_lr * (new_gbs / base_gbs)


def sqrt_scale_lr(base_lr: float, base_gbs: int, new_gbs: int) -> float:

    if base_gbs <= 0 or new_gbs <= 0:
        return base_lr

    return base_lr * math.sqrt(new_gbs / base_gbs)


def adascale_scale_lr(
    base_lr: float,
    base_gbs: int,
    new_gbs: int,
    gns: float,
) -> float:
    if base_gbs <= 0 or new_gbs <= 0:
        return base_lr
    if gns is None or not math.isfinite(gns) or gns < 0:
        raise ValueError("adascale needs a finite, non-negative gns")
    return base_lr * (gns + new_gbs) / (gns + base_gbs)


def power_scale_lr(
    base_lr: float,
    base_gbs: int,
    new_gbs: int,
    exp: float = 0.25,
    lr_cap: float = 0.0,
) -> float:

    if base_gbs <= 0 or new_gbs <= 0:
        return base_lr
    lr = base_lr * math.exp(exp * math.log(new_gbs / base_gbs))
    if lr_cap and lr_cap > 0 and lr > lr_cap:
        lr = lr_cap
    return lr


def piecewise_scale_lr(
    base_lr: float,
    base_gbs: int,
    new_gbs: int,
    pivot_gbs: int = 32,
    low_exp: float = 0.5,
    high_exp: float = 0.25,
    lr_cap: float = 0.0,
) -> float:

    if base_gbs <= 0 or new_gbs <= 0:
        return base_lr
    if pivot_gbs <= 0:
        pivot_gbs = base_gbs

    if new_gbs <= pivot_gbs:
        lr = base_lr * math.exp(low_exp * math.log(new_gbs / base_gbs))
    else:
        lr_pivot = base_lr * math.exp(low_exp * math.log(pivot_gbs / base_gbs))
        lr = lr_pivot * math.exp(high_exp * math.log(new_gbs / pivot_gbs))

    if lr_cap and lr_cap > 0 and lr > lr_cap:
        lr = lr_cap
    return lr


def scale_lr(
    strategy: str,
    base_lr: float,
    base_gbs: int,
    new_gbs: int,
    *,
    gns: Optional[float] = None,
    exp: float = 0.25,
    pivot_gbs: int = 32,
    low_exp: float = 0.5,
    high_exp: float = 0.25,
    lr_cap: float = 0.0,
) -> float:

    s = (strategy or "linear").lower()
    if s == "linear":
        return linear_scale_lr(base_lr, base_gbs, new_gbs)
    if s == "sqrt":
        return sqrt_scale_lr(base_lr, base_gbs, new_gbs)
    if s == "adascale":
        if new_gbs == base_gbs:
            return base_lr
        if gns is None:
            raise ValueError(
                "lr_scale_strategy 'adascale' needs the gradient noise scale. "
                "The online search takes it from the stage that just ran; a "
                "chain config supplies it per step as \"gns\", and a fixed "
                "strategy takes it from --gns"
            )
        return adascale_scale_lr(base_lr, base_gbs, new_gbs, gns)
    if s == "power":
        return power_scale_lr(base_lr, base_gbs, new_gbs, exp=exp, lr_cap=lr_cap)
    if s == "piecewise":
        return piecewise_scale_lr(
            base_lr,
            base_gbs,
            new_gbs,
            pivot_gbs=pivot_gbs,
            low_exp=low_exp,
            high_exp=high_exp,
            lr_cap=lr_cap,
        )
    raise ValueError(f"Unknown lr_scale_strategy: {strategy!r}")
