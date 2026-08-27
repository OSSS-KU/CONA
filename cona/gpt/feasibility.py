#!/usr/bin/env python3


from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Sequence

GIB = 1024 ** 3

DEFAULT_STATE_BYTES_PER_PARAM = 16
DEFAULT_ACT_BYTES_PER_TOKEN_HIDDEN = 2.0


@dataclass(frozen=True)
class ClusterSpec:
    """Hardware the strategy space is enumerated against."""

    num_gpus: int
    gpu_memory_bytes: int

    def __post_init__(self) -> None:
        if self.num_gpus <= 0:
            raise ValueError("num_gpus must be positive")
        if self.gpu_memory_bytes <= 0:
            raise ValueError("gpu_memory_bytes must be positive")

    @classmethod
    def from_gib(cls, num_gpus: int, gpu_memory_gib: float) -> "ClusterSpec":
        return cls(num_gpus=num_gpus, gpu_memory_bytes=int(gpu_memory_gib * GIB))


@dataclass(frozen=True)
class Strategy:
    """A parallelism strategy ``s = (dp, tp, pp, mbs, gbs)``."""

    dp: int
    tp: int
    pp: int
    mbs: int
    gbs: int

    def __post_init__(self) -> None:
        if min(self.dp, self.tp, self.pp, self.mbs, self.gbs) <= 0:
            raise ValueError("strategy dimensions must be positive")

    @property
    def world_size(self) -> int:
        return self.dp * self.tp * self.pp

    @property
    def num_microbatches(self) -> int:
        """Micro-batches per data-parallel replica per optimizer step."""
        return self.gbs // (self.dp * self.mbs)

    def __str__(self) -> str:
        return (
            f"dp{self.dp}_tp{self.tp}_pp{self.pp}_mbs{self.mbs}_gbs{self.gbs}"
        )


def _params_per_layer(hidden: int, ffn_hidden: int, heads: int, attn_size: int) -> int:
    """Parameter count of one transformer layer."""
    p = 2 * hidden * ffn_hidden                # MLP weights
    p += 4 * hidden * heads * attn_size        # attention weights
    p += hidden + ffn_hidden                   # MLP biases
    p += 3 * heads * attn_size + hidden        # attention biases
    p += 2 * 2 * hidden                        # two layer norms
    return p


def model_state_bytes(
    model,
    strategy: Strategy,
    vocab_size: int,
    state_bytes_per_param: int = DEFAULT_STATE_BYTES_PER_PARAM,
) -> float:
    """Largest per-stage model-state memory.

    ``M_state ~ (sigma_state / tp) * (h * v + (L / pp) * P_layer)``
    """
    hidden = int(model.hidden)
    heads = int(model.heads)
    layers = int(model.layers)
    ffn_hidden = int(getattr(model, "ffn_hidden", None) or 4 * hidden)
    attn_size = hidden // heads
    per_layer = _params_per_layer(hidden, ffn_hidden, heads, attn_size)
    params_on_stage = hidden * vocab_size + (layers / strategy.pp) * per_layer
    return state_bytes_per_param * params_on_stage / strategy.tp


def activation_bytes(
    model,
    strategy: Strategy,
    act_bytes_per_token_hidden: float = DEFAULT_ACT_BYTES_PER_TOKEN_HIDDEN,
) -> float:
    """Per-device activation memory.

    ``M_act ~ sigma_act * mbs * s * h * L / (tp * pp)``
    """
    layers_on_stage = int(model.layers) / (strategy.tp * strategy.pp)
    return (
        act_bytes_per_token_hidden
        * strategy.mbs
        * int(model.seq)
        * int(model.hidden)
        * layers_on_stage
    )


def memory_bytes(
    model,
    strategy: Strategy,
    vocab_size: int,
    state_bytes_per_param: int = DEFAULT_STATE_BYTES_PER_PARAM,
    act_bytes_per_token_hidden: float = DEFAULT_ACT_BYTES_PER_TOKEN_HIDDEN,
) -> float:
    """Estimated per-GPU memory footprint ``M_state + M_act``."""
    return model_state_bytes(
        model, strategy, vocab_size, state_bytes_per_param
    ) + activation_bytes(model, strategy, act_bytes_per_token_hidden)


def is_feasible(
    model,
    cluster: ClusterSpec,
    strategy: Strategy,
    vocab_size: int,
    state_bytes_per_param: int = DEFAULT_STATE_BYTES_PER_PARAM,
    act_bytes_per_token_hidden: float = DEFAULT_ACT_BYTES_PER_TOKEN_HIDDEN,
) -> bool:
    """Check one candidate against every constraint."""
    if strategy.world_size != cluster.num_gpus:
        return False
    if strategy.gbs % strategy.dp != 0:
        return False
    if (strategy.gbs // strategy.dp) % strategy.mbs != 0:
        return False
    if int(model.hidden) % strategy.tp != 0:
        return False
    if int(model.heads) % strategy.tp != 0:
        return False
    if int(model.layers) % strategy.pp != 0:
        return False
    footprint = memory_bytes(
        model,
        strategy,
        vocab_size,
        state_bytes_per_param,
        act_bytes_per_token_hidden,
    )
    return footprint <= cluster.gpu_memory_bytes


def _divisors(value: int) -> Iterator[int]:
    for candidate in range(1, value + 1):
        if value % candidate == 0:
            yield candidate


def enumerate_strategies(
    model,
    cluster: ClusterSpec,
    gbs: int,
    vocab_size: int,
    max_mbs: int = 32,
    state_bytes_per_param: int = DEFAULT_STATE_BYTES_PER_PARAM,
    act_bytes_per_token_hidden: float = DEFAULT_ACT_BYTES_PER_TOKEN_HIDDEN,
) -> List[Strategy]:
    """Return the feasible strategy set ``S(gbs)``, ordered deterministically."""
    if gbs <= 0:
        raise ValueError("gbs must be positive")
    out: List[Strategy] = []
    for tp in _divisors(cluster.num_gpus):
        remaining = cluster.num_gpus // tp
        for pp in _divisors(remaining):
            dp = remaining // pp
            if gbs % dp != 0:
                continue
            local_batch = gbs // dp
            for mbs in _divisors(local_batch):
                if mbs > max_mbs:
                    break
                candidate = Strategy(dp=dp, tp=tp, pp=pp, mbs=mbs, gbs=gbs)
                if is_feasible(
                    model,
                    cluster,
                    candidate,
                    vocab_size,
                    state_bytes_per_param,
                    act_bytes_per_token_hidden,
                ):
                    out.append(candidate)
    out.sort(key=lambda s: (s.dp, s.tp, s.pp, s.mbs))
    return out


def next_frontier_gbs(current_gbs: int) -> int:
    """Next larger global batch size on the power-of-two ladder.

    CONA advances only to the adjacent larger batch size and never revisits
    smaller ones.
    """
    if current_gbs <= 0:
        raise ValueError("current_gbs must be positive")
    return 2 * current_gbs


def feasible_gbs_ladder(
    model,
    cluster: ClusterSpec,
    initial_gbs: int,
    vocab_size: int,
    max_gbs: int,
    **kwargs,
) -> List[int]:
    """Power-of-two batch sizes from ``initial_gbs`` that admit a strategy."""
    ladder: List[int] = []
    gbs = initial_gbs
    while gbs <= max_gbs:
        if enumerate_strategies(model, cluster, gbs, vocab_size, **kwargs):
            ladder.append(gbs)
        gbs = next_frontier_gbs(gbs)
    return ladder


__all__: Sequence[str] = [
    "ClusterSpec",
    "Strategy",
    "activation_bytes",
    "enumerate_strategies",
    "feasible_gbs_ladder",
    "is_feasible",
    "memory_bytes",
    "model_state_bytes",
    "next_frontier_gbs",
]
