#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isfinite, log2
from typing import Dict, List, Optional, Sequence, Tuple

from fallback import GnsFallbackMonitor
from feasibility import (
    ClusterSpec,
    Strategy,
    enumerate_strategies,
    next_frontier_gbs,
)
from perf_model import CalculonTimeModel, PerfModelError
from switch_rule import ratio_from_scalars

BOOTSTRAP_SAMPLES = 10
EI_STOP_FRACTION = 0.10
MAX_GP_UPDATES = 100


def throughput(strategy: Strategy, iteration_time_sec: float) -> float:
    """``Throughput(s) = gbs(s) / T(s)`` in samples per second."""
    if not isfinite(iteration_time_sec) or iteration_time_sec <= 0:
        raise ValueError("iteration_time_sec must be finite and positive")
    return strategy.gbs / iteration_time_sec


def surrogate_phi(strategy: Strategy, iteration_time_sec: float, gns: float) -> float:
    """The GNS-aware surrogate ``Phi_t(s)``."""
    if not isfinite(gns) or gns < 0:
        raise ValueError("gns must be finite and non-negative")
    return throughput(strategy, iteration_time_sec) / (strategy.gbs + gns)


def switch_ratio(
    current: Strategy,
    current_time: float,
    candidate: Strategy,
    candidate_time: float,
    gns: float,
) -> float:
    """``R_t``: the surrogate ratio of candidate over current.

    The training loop applies the same rule to the published candidate, so both
    sides go through :func:`switch_rule.ratio_from_scalars`.
    """
    if not isfinite(gns) or gns < 0:
        raise ValueError("gns must be finite and non-negative")
    return ratio_from_scalars(
        current.gbs, current_time, candidate.gbs, candidate_time, gns
    )


@dataclass(frozen=True)
class SearchDecision:
    """Outcome of one search iteration."""

    iteration: int
    gns: float
    current: Strategy
    current_time: float
    candidate: Optional[Strategy]
    candidate_time: Optional[float]
    ratio: Optional[float]
    switched: bool
    gp_updates: int
    acquisition_active: bool
    fell_back: bool = False

    def summary(self) -> str:
        if self.fell_back:
            return (
                f"iter={self.iteration} gns={self.gns:.1f} "
                f"cur={self.current} FALLBACK (sustained GNS decrease)"
            )
        if self.candidate is None:
            return (
                f"iter={self.iteration} gns={self.gns:.1f} cur={self.current} "
                "candidate=<no larger feasible frontier>"
            )
        return (
            f"iter={self.iteration} gns={self.gns:.1f} cur={self.current} "
            f"T_cur={self.current_time * 1e3:.1f}ms next={self.candidate} "
            f"T_next={self.candidate_time * 1e3:.1f}ms R={self.ratio:.4f} "
            f"{'SWITCH' if self.switched else 'keep'}"
        )


@dataclass
class _Frontier:
    """BO state for one target batch size ``S(gbs)``."""

    gbs: int
    candidates: List[Strategy]
    evaluated: Dict[Strategy, float] = field(default_factory=dict)
    updates: int = 0
    acquisition_active: bool = True

    @property
    def best(self) -> Optional[Tuple[Strategy, float]]:
        if not self.evaluated:
            return None
        strategy = min(self.evaluated, key=self.evaluated.get)
        return strategy, self.evaluated[strategy]

    @property
    def unevaluated(self) -> List[Strategy]:
        return [s for s in self.candidates if s not in self.evaluated]


def _features(strategy: Strategy) -> List[float]:
    """Log2 parallelism degrees; the GP input space of the search."""
    return [
        log2(strategy.dp),
        log2(strategy.tp),
        log2(strategy.pp),
        log2(strategy.mbs),
    ]


def latin_hypercube_subset(
    candidates: Sequence[Strategy], count: int, seed: int = 0
) -> List[Strategy]:
    import numpy as np

    if count <= 0 or not candidates:
        return []
    if count >= len(candidates):
        return list(candidates)

    points = np.asarray([_features(s) for s in candidates], dtype=float)
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    normalized = (points - lo) / span

    rng = np.random.default_rng(seed)
    dims = normalized.shape[1]
    strata = (rng.permuted(
        np.tile(np.arange(count), (dims, 1)), axis=1
    ).T + rng.random((count, dims))) / count

    picked: List[Strategy] = []
    taken: set = set()
    for target in strata:
        distances = np.linalg.norm(normalized - target, axis=1)
        for index in np.argsort(distances):
            if index not in taken:
                taken.add(int(index))
                picked.append(candidates[int(index)])
                break
    return picked


class OnlineStrategySearch:

    def __init__(
        self,
        model,
        cluster: ClusterSpec,
        time_model: CalculonTimeModel,
        initial_gbs: int,
        vocab_size: int = 50257,
        max_gbs: Optional[int] = None,
        bootstrap_samples: int = BOOTSTRAP_SAMPLES,
        ei_stop_fraction: float = EI_STOP_FRACTION,
        max_gp_updates: int = MAX_GP_UPDATES,
        max_mbs: int = 32,
        seed: int = 0,
        use_botorch: bool = True,
        fallback: Optional[GnsFallbackMonitor] = None,
    ):
        self.model = model
        self.cluster = cluster
        self.time_model = time_model
        self.vocab_size = vocab_size
        self.max_gbs = max_gbs
        self.bootstrap_samples = bootstrap_samples
        self.ei_stop_fraction = ei_stop_fraction
        self.max_gp_updates = max_gp_updates
        self.max_mbs = max_mbs
        self.seed = seed
        self.use_botorch = use_botorch
        self.fallback = fallback

        self.iteration = 0
        self.chain: List[Strategy] = []
        self.current: Strategy
        self.current_time: float
        self._frontier: Optional[_Frontier] = None
        self._passed: List[Tuple[Strategy, float]] = []

        self._bootstrap(initial_gbs)

    def _feasible(self, gbs: int) -> List[Strategy]:
        return enumerate_strategies(
            self.model,
            self.cluster,
            gbs,
            self.vocab_size,
            max_mbs=self.max_mbs,
        )

    def _score(self, candidates: Sequence[Strategy]) -> Dict[Strategy, float]:
        """Analytical ``T(s)`` for each candidate, dropping any Calculon rejects."""
        scored: Dict[Strategy, float] = {}
        for candidate in candidates:
            seconds = self.time_model.try_iteration_time(candidate)
            if seconds is not None:
                scored[candidate] = seconds
        return scored

    def _bootstrap(self, initial_gbs: int) -> None:

        candidates = self._feasible(initial_gbs)
        if not candidates:
            raise ValueError(
                f"No feasible strategy at gbs={initial_gbs} on "
                f"{self.cluster.num_gpus} GPUs"
            )
        sampled = latin_hypercube_subset(
            candidates, self.bootstrap_samples, seed=self.seed
        )
        scored = self._score(sampled)
        if not scored:
            raise PerfModelError(
                f"Calculon rejected every bootstrap sample at gbs={initial_gbs}"
            )
        self.current = min(scored, key=scored.get)
        self.current_time = scored[self.current]
        self.chain = [self.current]
        self.init_search(next_frontier_gbs(initial_gbs))

    def init_search(self, gbs: int) -> None:
        """``INITSEARCH(S(gbs))``: open a fresh BO frontier at ``gbs``."""
        if self.max_gbs is not None and gbs > self.max_gbs:
            self._frontier = None
            return
        candidates = self._feasible(gbs)
        if not candidates:
            self._frontier = None
            return
        frontier = _Frontier(gbs=gbs, candidates=candidates)
        sampled = latin_hypercube_subset(
            candidates, self.bootstrap_samples, seed=self.seed + gbs
        )
        frontier.evaluated.update(self._score(sampled))
        if not frontier.evaluated:
            self._frontier = None
            return
        self._frontier = frontier

    @property
    def previous(self) -> Optional[Strategy]:
        """``s_prev``: the strategy a fallback would revert to."""
        return self._passed[-1][0] if self._passed else None

    @property
    def previous_time(self) -> Optional[float]:
        return self._passed[-1][1] if self._passed else None

    @property
    def candidate(self) -> Optional[Strategy]:
        """``s_next``: current argmin-T belief on the target frontier."""
        if self._frontier is None:
            return None
        best = self._frontier.best
        return None if best is None else best[0]

    @property
    def candidate_time(self) -> Optional[float]:
        if self._frontier is None:
            return None
        best = self._frontier.best
        return None if best is None else best[1]

    def search_by_bo(self) -> Optional[Strategy]:

        frontier = self._frontier
        if frontier is None:
            return None
        if frontier.acquisition_active:
            if frontier.updates >= self.max_gp_updates or not frontier.unevaluated:
                frontier.acquisition_active = False
            else:
                self._acquire(frontier)
        return self.candidate

    def _acquire(self, frontier: _Frontier) -> None:
        """Evaluate the highest-EI unevaluated candidate and refit the GP."""
        pending = frontier.unevaluated
        best = frontier.best
        assert best is not None  # a frontier always carries its bootstrap points
        best_time = best[1]

        picked, expected_improvement = self._expected_improvement_argmax(
            frontier, pending
        )
        if picked is None:
            frontier.acquisition_active = False
            return

        if (
            expected_improvement is not None
            and expected_improvement < self.ei_stop_fraction * best_time
        ):
            frontier.acquisition_active = False
            return

        seconds = self.time_model.try_iteration_time(picked)
        frontier.updates += 1
        if seconds is None:
            frontier.candidates = [s for s in frontier.candidates if s != picked]
            return
        frontier.evaluated[picked] = seconds

    def _expected_improvement_argmax(
        self, frontier: _Frontier, pending: Sequence[Strategy]
    ) -> Tuple[Optional[Strategy], Optional[float]]:

        if not pending:
            return None, None
        if not self.use_botorch:
            return pending[0], None

        try:
            import torch
            from botorch.acquisition.analytic import LogNoisyExpectedImprovement
            from botorch.fit import fit_gpytorch_mll
            from botorch.models import SingleTaskGP
            from botorch.models.transforms.outcome import Standardize
            from gpytorch.mlls import ExactMarginalLogLikelihood
        except ImportError:
            return pending[0], None

        observed = list(frontier.evaluated.items())
        if len(observed) < 2:
            return pending[0], None

        train_x = torch.tensor(
            [_features(s) for s, _ in observed], dtype=torch.double
        )
        train_y = torch.tensor(
            [[-seconds] for _, seconds in observed], dtype=torch.double
        )
        test_x = torch.tensor([_features(s) for s in pending], dtype=torch.double)

        try:
            gp = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
            acquisition = LogNoisyExpectedImprovement(
                model=gp, X_observed=train_x, maximize=True
            )
            log_ei = acquisition(test_x.unsqueeze(1)).detach().flatten()
        except Exception:
            return pending[0], None

        index = int(torch.argmax(log_ei).item())
        try:
            expected_improvement = float(exp(float(log_ei[index].item())))
        except OverflowError:
            expected_improvement = float("inf")
        return pending[index], expected_improvement

    def observe(
        self, gns: float, measured_iteration_time: Optional[float] = None
    ) -> SearchDecision:

        self.iteration += 1
        if measured_iteration_time is not None:
            if not (measured_iteration_time > 0):
                raise ValueError("measured_iteration_time must be positive")
            self.current_time = float(measured_iteration_time)

        if self.fallback is not None and self.fallback.observe(gns):
            reverted = self.fall_back()
            if reverted is not None:
                return SearchDecision(
                    iteration=self.iteration,
                    gns=gns,
                    current=self.current,
                    current_time=self.current_time,
                    candidate=None,
                    candidate_time=None,
                    ratio=None,
                    switched=False,
                    gp_updates=0,
                    acquisition_active=False,
                    fell_back=True,
                )

        candidate = self.search_by_bo()
        candidate_time = self.candidate_time
        frontier = self._frontier

        if candidate is None or candidate_time is None:
            return SearchDecision(
                iteration=self.iteration,
                gns=gns,
                current=self.current,
                current_time=self.current_time,
                candidate=None,
                candidate_time=None,
                ratio=None,
                switched=False,
                gp_updates=0 if frontier is None else frontier.updates,
                acquisition_active=False,
            )

        ratio = switch_ratio(
            self.current, self.current_time, candidate, candidate_time, gns
        )
        switched = ratio > 1.0
        decision = SearchDecision(
            iteration=self.iteration,
            gns=gns,
            current=self.current,
            current_time=self.current_time,
            candidate=candidate,
            candidate_time=candidate_time,
            ratio=ratio,
            switched=switched,
            gp_updates=frontier.updates,
            acquisition_active=frontier.acquisition_active,
        )

        if switched:
            self.adopt(candidate, candidate_time)

        return decision

    def adopt(self, candidate: Strategy, candidate_time: float) -> None:

        self._passed.append((self.current, self.current_time))
        self.current = candidate
        self.current_time = candidate_time
        self.chain.append(self.current)
        self.init_search(next_frontier_gbs(self.current.gbs))

    def fall_back(self) -> Optional[Strategy]:
        if not self._passed:
            return None
        strategy, seconds = self._passed.pop()
        print(f"[FALLBACK] reverting {self.current} -> {strategy}")
        self.current = strategy
        self.current_time = seconds
        self.chain.append(strategy)
        if self.fallback is not None:
            self.fallback.reset()
        self.init_search(next_frontier_gbs(strategy.gbs))
        return strategy


__all__ = [
    "BOOTSTRAP_SAMPLES",
    "EI_STOP_FRACTION",
    "MAX_GP_UPDATES",
    "OnlineStrategySearch",
    "SearchDecision",
    "latin_hypercube_subset",
    "surrogate_phi",
    "switch_ratio",
    "throughput",
]
