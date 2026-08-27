#!/usr/bin/env python3

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from feasibility import Strategy

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CALCULON_ROOT = _REPO_ROOT / "utils" / "calculon"
_SYSTEM_ROOT = Path(__file__).resolve().parents[1] / "config" / "calculon" / "systems"


class PerfModelError(RuntimeError):
    """Raised when the analytical model cannot score a strategy."""


def _import_calculon():
    """Import the vendored Calculon package, adding it to ``sys.path`` once."""
    if not _CALCULON_ROOT.is_dir():
        raise PerfModelError(
            f"Calculon not found at {_CALCULON_ROOT}. The analytical iteration-time "
            "model of Isaev et al. (2023) is required by the strategy search."
        )
    root = str(_CALCULON_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from calculon.llm import Llm
        from calculon.system import System
    except ImportError as exc:  # pragma: no cover - environment problem
        raise PerfModelError(f"Failed to import Calculon: {exc}") from exc
    return Llm, System


def resolve_system_path(system: str) -> Path:
    """Resolve a system name or path to a Calculon system JSON file."""
    candidate = Path(system)
    if candidate.is_file():
        return candidate
    for root in (_SYSTEM_ROOT, _CALCULON_ROOT / "systems"):
        for name in (system, f"{system}.json"):
            path = root / name
            if path.is_file():
                return path
    available = sorted(
        p.stem
        for root in (_SYSTEM_ROOT, _CALCULON_ROOT / "systems")
        if root.is_dir()
        for p in root.glob("*.json")
    )
    raise PerfModelError(
        f"Unknown Calculon system {system!r}. Available: {', '.join(available)}"
    )


def application_from_model(model, vocab_size: int) -> Dict[str, Any]:
    """Build a Calculon application description from CONA's ``ModelConfig``.

    ``vocab_size`` is passed through for callers that log it; Calculon's own
    parameter count assumes the Megatron GPT-2 vocabulary, so the value only
    affects CONA's memory screen in :mod:`feasibility`.
    """
    hidden = int(model.hidden)
    heads = int(model.heads)
    if hidden % heads != 0:
        raise PerfModelError(
            f"hidden ({hidden}) must be divisible by heads ({heads})"
        )
    return {
        "hidden": hidden,
        "feedforward": int(getattr(model, "ffn_hidden", None) or 4 * hidden),
        "seq_size": int(model.seq),
        "attn_heads": heads,
        "attn_size": hidden // heads,
        "num_blocks": int(model.layers),
    }


@dataclass(frozen=True)
class ExecutionOptions:

    datatype: str = "float16"
    fused_activation: bool = True
    attention_type: str = "multihead"
    activation_recompute: str = "full"
    tensor_par_comm_type: str = "ar"
    optimizer_sharding: bool = True
    data_par_overlap: bool = False
    tensor_par_overlap: str = "none"
    tensor_par_net: int = 0
    pipeline_par_net: int = 1
    data_par_net: int = 1


class CalculonTimeModel:
    """Estimate ``T(s)``, the per-iteration time of a strategy, analytically.

    The model never touches a GPU: every value comes from Calculon's
    roofline-style performance model, which is what lets CONA refine its
    Gaussian-process surrogate while training runs undisturbed.
    """

    def __init__(
        self,
        model,
        system: str = "a6000_48g",
        vocab_size: int = 50257,
        options: Optional[ExecutionOptions] = None,
    ):
        self._llm_cls, self._system_cls = _import_calculon()
        self.options = options or ExecutionOptions()
        self.system_path = resolve_system_path(system)
        self.application = application_from_model(model, vocab_size)
        self._system_json = json.loads(self.system_path.read_text())
        self._logger = logging.getLogger("cona.perf_model")
        if not self._logger.handlers:
            self._logger.addHandler(logging.NullHandler())
        self._cache: Dict[Strategy, float] = {}

    @property
    def device_memory_bytes(self) -> int:
        """Per-GPU memory capacity declared by the selected system file."""
        return int(self._system_json["mem1"]["GiB"]) * 1024 ** 3

    def _execution_json(self, strategy: Strategy) -> Dict[str, Any]:
        opt = self.options
        return {
            "num_procs": strategy.world_size,
            "tensor_par": strategy.tp,
            "pipeline_par": strategy.pp,
            "data_par": strategy.dp,
            "tensor_par_net": opt.tensor_par_net,
            "pipeline_par_net": opt.pipeline_par_net,
            "data_par_net": opt.data_par_net,
            "batch_size": strategy.gbs,
            "microbatch_size": strategy.mbs,
            "datatype": opt.datatype,
            "fused_activation": opt.fused_activation,
            "attention_type": opt.attention_type,
            "activation_recompute": opt.activation_recompute,
            "pipeline_interleaving": 1,
            "optimizer_sharding": opt.optimizer_sharding and strategy.dp > 1,
            "tensor_par_comm_type": opt.tensor_par_comm_type,
            "tensor_par_overlap": opt.tensor_par_overlap if strategy.tp > 1 else "none",
            "seq_par_ag_redo": False,
            "data_par_overlap": opt.data_par_overlap and strategy.dp > 1,
            "weight_offload": False,
            "activations_offload": False,
            "optimizer_offload": False,
            "training": True,
        }

    def iteration_time(self, strategy: Strategy) -> float:
        """Return ``T(s)`` in seconds, or raise :class:`PerfModelError`."""
        cached = self._cache.get(strategy)
        if cached is not None:
            return cached

        llm_cls, system_cls = self._llm_cls, self._system_cls
        try:
            app = llm_cls.Application(dict(self.application))
            exe = llm_cls.Execution.from_json(self._execution_json(strategy))
            syst = system_cls(json.loads(json.dumps(self._system_json)))
            estimator = llm_cls(app, self._logger)
            estimator.compile(syst, exe)
            estimator.run(syst)
            seconds = float(estimator.get_total_time())
        except Exception as exc:  # Calculon raises Llm.Error and AssertionError
            raise PerfModelError(f"Calculon rejected {strategy}: {exc}") from exc

        if not (seconds > 0):
            raise PerfModelError(f"Calculon returned non-positive T(s) for {strategy}")
        self._cache[strategy] = seconds
        return seconds

    def try_iteration_time(self, strategy: Strategy) -> Optional[float]:
        """Same as :meth:`iteration_time` but returns ``None`` on rejection."""
        try:
            return self.iteration_time(strategy)
        except PerfModelError:
            return None


@lru_cache(maxsize=None)
def available_systems() -> tuple:
    """Names of the Calculon system files CONA can score against."""
    names = set()
    for root in (_SYSTEM_ROOT, _CALCULON_ROOT / "systems"):
        if root.is_dir():
            names.update(p.stem for p in root.glob("*.json"))
    return tuple(sorted(names))


__all__ = [
    "CalculonTimeModel",
    "ExecutionOptions",
    "PerfModelError",
    "application_from_model",
    "available_systems",
    "resolve_system_path",
]
