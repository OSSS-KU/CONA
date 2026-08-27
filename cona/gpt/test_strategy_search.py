#!/usr/bin/env python3

import unittest

from config import ModelConfig
from feasibility import (
    ClusterSpec,
    Strategy,
    enumerate_strategies,
    feasible_gbs_ladder,
    is_feasible,
    memory_bytes,
    next_frontier_gbs,
)
from fallback import (
    GnsFallbackMonitor,
    previous_is_preferable,
    reversal_gns_threshold,
)
from lr_scaling import adascale_scale_lr, scale_lr
from perf_model import CalculonTimeModel, PerfModelError
from strategy_search import (
    OnlineStrategySearch,
    latin_hypercube_subset,
    surrogate_phi,
    switch_ratio,
)

VOCAB = 50257


def gpt2_medium() -> ModelConfig:
    """The .conaconfig default: 24 layers, hidden 1024, 16 heads, seq 1024."""
    return ModelConfig()


class FeasibilityTests(unittest.TestCase):
    def setUp(self):
        self.model = gpt2_medium()
        self.cluster = ClusterSpec.from_gib(num_gpus=4, gpu_memory_gib=80)

    def test_world_size_must_match_gpu_count(self):
        self.assertFalse(
            is_feasible(
                self.model, self.cluster, Strategy(2, 1, 1, 2, 8), VOCAB
            )
        )
        self.assertTrue(
            is_feasible(
                self.model, self.cluster, Strategy(4, 1, 1, 2, 8), VOCAB
            )
        )

    def test_divisibility_constraints(self):
        # gbs must be divisible by dp.
        self.assertFalse(
            is_feasible(self.model, self.cluster, Strategy(4, 1, 1, 1, 6), VOCAB)
        )
        # heads (16) is not divisible by tp=3, and 3 does not divide 4 GPUs.
        self.assertFalse(
            is_feasible(self.model, self.cluster, Strategy(1, 3, 1, 1, 8), VOCAB)
        )
        # layers (24) is not divisible by pp=5.
        self.assertFalse(
            is_feasible(self.model, self.cluster, Strategy(1, 1, 5, 1, 8), VOCAB)
        )

    def test_local_batch_must_split_into_microbatches(self):
        # dp=2 leaves a local batch of 4, which mbs=3 cannot tile.
        self.assertFalse(
            is_feasible(self.model, self.cluster, Strategy(2, 2, 1, 3, 8), VOCAB)
        )

    def test_memory_screen_rejects_oversized_activations(self):
        small = ClusterSpec.from_gib(num_gpus=4, gpu_memory_gib=8)
        strategy = Strategy(dp=1, tp=1, pp=1, mbs=32, gbs=32)
        self.assertGreater(
            memory_bytes(self.model, strategy, VOCAB), small.gpu_memory_bytes
        )
        self.assertFalse(is_feasible(self.model, small, strategy, VOCAB))

    def test_enumeration_yields_only_feasible_strategies(self):
        candidates = enumerate_strategies(self.model, self.cluster, 16, VOCAB)
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertEqual(candidate.world_size, 4)
            self.assertEqual(candidate.gbs, 16)
            self.assertEqual(
                candidate.num_microbatches * candidate.mbs * candidate.dp, 16
            )
            self.assertTrue(
                is_feasible(self.model, self.cluster, candidate, VOCAB)
            )

    def test_frontier_ladder_is_powers_of_two(self):
        self.assertEqual(next_frontier_gbs(8), 16)
        ladder = feasible_gbs_ladder(
            self.model, self.cluster, 8, VOCAB, max_gbs=64
        )
        self.assertEqual(ladder, [8, 16, 32, 64])


class SurrogateTests(unittest.TestCase):
    def test_surrogate_favors_large_batch_when_gns_is_large(self):
        small = Strategy(1, 1, 1, 1, 8)
        large = Strategy(1, 1, 1, 2, 16)
        # Larger batch is only 1.2x slower per iteration, so at high GNS it wins.
        self.assertGreater(
            surrogate_phi(large, 1.2, gns=64), surrogate_phi(small, 1.0, gns=64)
        )
        self.assertLess(
            surrogate_phi(large, 1.2, gns=0), surrogate_phi(small, 1.0, gns=0)
        )

    def test_switch_ratio_matches_equation_10(self):
        current, candidate = Strategy(4, 1, 1, 2, 8), Strategy(4, 1, 1, 4, 16)
        t_cur, t_next, gns = 1.0, 1.2, 8.0
        expected = (t_cur / t_next) * (16 / 8) * ((8 + gns) / (16 + gns))
        self.assertAlmostEqual(
            switch_ratio(current, t_cur, candidate, t_next, gns), expected
        )

    def test_switch_ratio_crosses_one_as_gns_grows(self):
        current, candidate = Strategy(4, 1, 1, 2, 8), Strategy(4, 1, 1, 4, 16)
        low = switch_ratio(current, 1.0, candidate, 1.9, gns=0.0)
        high = switch_ratio(current, 1.0, candidate, 1.9, gns=512.0)
        self.assertLess(low, 1.0)
        self.assertGreater(high, 1.0)
        self.assertGreater(high, low)


class LatinHypercubeTests(unittest.TestCase):
    def test_returns_distinct_candidates_and_is_deterministic(self):
        model, cluster = gpt2_medium(), ClusterSpec.from_gib(16, 48)
        candidates = enumerate_strategies(model, cluster, 64, VOCAB)
        self.assertGreater(len(candidates), 10)
        first = latin_hypercube_subset(candidates, 10, seed=7)
        second = latin_hypercube_subset(candidates, 10, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(len(set(first)), 10)

    def test_returns_everything_when_count_exceeds_pool(self):
        model, cluster = gpt2_medium(), ClusterSpec.from_gib(4, 80)
        candidates = enumerate_strategies(model, cluster, 8, VOCAB)
        self.assertEqual(
            latin_hypercube_subset(candidates, len(candidates) + 5), candidates
        )


class CalculonTimeModelTests(unittest.TestCase):
    def setUp(self):
        self.model = gpt2_medium()
        try:
            self.time_model = CalculonTimeModel(
                self.model, system="a100_80g_2gpu_node"
            )
        except PerfModelError as exc:  # pragma: no cover
            self.skipTest(f"Calculon unavailable: {exc}")

    def test_iteration_time_is_positive_and_cached(self):
        strategy = Strategy(dp=2, tp=1, pp=2, mbs=2, gbs=8)
        first = self.time_model.iteration_time(strategy)
        self.assertGreater(first, 0.0)
        self.assertEqual(first, self.time_model.iteration_time(strategy))

    def test_larger_batch_costs_more_time_per_iteration(self):
        small = self.time_model.iteration_time(Strategy(4, 1, 1, 2, 8))
        large = self.time_model.iteration_time(Strategy(4, 1, 1, 2, 32))
        self.assertGreater(large, small)

    def test_declared_device_memory_matches_system_file(self):
        self.assertEqual(self.time_model.device_memory_bytes, 80 * 1024 ** 3)


class OnlineStrategySearchTests(unittest.TestCase):
    def setUp(self):
        self.model = gpt2_medium()
        self.cluster = ClusterSpec.from_gib(num_gpus=4, gpu_memory_gib=80)
        try:
            self.time_model = CalculonTimeModel(
                self.model, system="a100_80g_2gpu_node"
            )
        except PerfModelError as exc:  # pragma: no cover
            self.skipTest(f"Calculon unavailable: {exc}")

    def _search(self, **kwargs):
        kwargs.setdefault("use_botorch", False)
        kwargs.setdefault("initial_gbs", 8)
        kwargs.setdefault("max_gbs", 64)
        return OnlineStrategySearch(
            self.model, self.cluster, self.time_model, **kwargs
        )

    def test_bootstrap_sets_current_and_candidate(self):
        search = self._search()
        self.assertEqual(search.current.gbs, 8)
        self.assertEqual(search.chain, [search.current])
        self.assertIsNotNone(search.candidate)
        self.assertEqual(search.candidate.gbs, 16)

    def test_zero_gns_keeps_the_small_batch(self):
        # With GNS_t = 0 the surrogate reduces to 1/T(s), so a slower
        # larger-batch strategy must not be adopted.
        search = self._search()
        decision = search.observe(gns=0.0)
        self.assertGreater(decision.candidate_time, 0.0)
        self.assertLess(decision.ratio, 1.0)
        self.assertFalse(decision.switched)
        self.assertEqual(search.current.gbs, 8)

    def test_large_gns_switches_and_advances_the_frontier(self):
        search = self._search()
        decision = search.observe(gns=1e6)
        self.assertTrue(decision.switched)
        self.assertEqual(search.current.gbs, 16)
        self.assertEqual([s.gbs for s in search.chain], [8, 16])
        # The target frontier moved to the next power of two.
        self.assertEqual(search.candidate.gbs, 32)

    def test_chain_climbs_the_ladder_and_stops_at_max_gbs(self):
        search = self._search()
        for _ in range(10):
            search.observe(gns=1e6)
        self.assertEqual([s.gbs for s in search.chain], [8, 16, 32, 64])
        # 128 is past max_gbs, so no candidate remains.
        self.assertIsNone(search.candidate)
        final = search.observe(gns=1e6)
        self.assertFalse(final.switched)
        self.assertIsNone(final.ratio)

    def test_measured_iteration_time_overrides_the_estimate(self):
        search = self._search()
        estimated = search.current_time
        # A measured T(s_cur) far above the estimate makes the current strategy
        # look slow, so the ratio favours the candidate even at GNS_t = 0.
        decision = search.observe(gns=0.0, measured_iteration_time=5.0)
        self.assertEqual(decision.current_time, 5.0)
        self.assertNotAlmostEqual(estimated, 5.0)
        self.assertTrue(decision.switched)
        with self.assertRaises(ValueError):
            search.observe(gns=0.0, measured_iteration_time=0.0)

    def test_acquisition_stops_after_the_update_budget(self):
        search = self._search(max_gp_updates=0)
        search.observe(gns=0.0)
        self.assertFalse(search._frontier.acquisition_active)
        self.assertEqual(search._frontier.updates, 0)
        # A stopped frontier is still queryable.
        self.assertIsNotNone(search.candidate)


class LrScalingTests(unittest.TestCase):
    BASE_LR = 3e-4

    def test_linear_and_sqrt_follow_the_batch_size_ratio(self):
        self.assertAlmostEqual(scale_lr("linear", 1.0, 8, 32), 4.0)
        self.assertAlmostEqual(scale_lr("sqrt", 1.0, 8, 32), 2.0)

    def test_linear_is_the_default_rule(self):
        self.assertEqual(scale_lr("", 1.0, 8, 32), scale_lr("linear", 1.0, 8, 32))
        self.assertEqual(scale_lr(None, 1.0, 8, 32), scale_lr("linear", 1.0, 8, 32))

    def test_adascale_interpolates_between_no_scaling_and_linear(self):
        linear = scale_lr("linear", 1.0, 8, 16)
        noiseless = adascale_scale_lr(1.0, 8, 16, gns=0.0)
        all_noise = adascale_scale_lr(1.0, 8, 16, gns=1e12)
        self.assertAlmostEqual(noiseless, linear)
        self.assertAlmostEqual(all_noise, 1.0, places=6)
        middle = adascale_scale_lr(1.0, 8, 16, gns=8.0)
        self.assertLess(middle, linear)
        self.assertGreater(middle, 1.0)

    def test_adascale_matches_the_gain_ratio(self):
        # r = (gns + new_gbs) / (gns + base_gbs)
        self.assertAlmostEqual(
            adascale_scale_lr(self.BASE_LR, 8, 16, gns=16.0),
            self.BASE_LR * (16.0 + 16) / (16.0 + 8),
        )

    def test_adascale_is_monotone_in_batch_size(self):
        previous = 0.0
        for gbs in (8, 16, 32, 64):
            lr = adascale_scale_lr(1.0, 8, gbs, gns=12.0)
            self.assertGreater(lr, previous)
            previous = lr

    def test_adascale_needs_a_gns_only_when_the_batch_size_changes(self):
        self.assertEqual(scale_lr("adascale", self.BASE_LR, 8, 8), self.BASE_LR)
        with self.assertRaises(ValueError):
            scale_lr("adascale", self.BASE_LR, 8, 16)
        with self.assertRaises(ValueError):
            adascale_scale_lr(self.BASE_LR, 8, 16, gns=float("nan"))
        with self.assertRaises(ValueError):
            adascale_scale_lr(self.BASE_LR, 8, 16, gns=-1.0)

    def test_unknown_rule_is_rejected(self):
        with self.assertRaises(ValueError):
            scale_lr("adascale-v2", self.BASE_LR, 8, 16, gns=1.0)


class ReversalConditionTests(unittest.TestCase):
    def test_threshold_is_where_the_two_surrogates_meet(self):
        threshold = reversal_gns_threshold(8, 1.0, 16, 1.2)
        self.assertIsNotNone(threshold)
        prev = Strategy(2, 1, 1, 4, 8)
        cur = Strategy(2, 1, 1, 8, 16)
        self.assertAlmostEqual(
            surrogate_phi(prev, 1.0, threshold),
            surrogate_phi(cur, 1.2, threshold),
        )
        self.assertTrue(previous_is_preferable(8, 1.0, 16, 1.2, threshold * 0.9))
        self.assertFalse(previous_is_preferable(8, 1.0, 16, 1.2, threshold * 1.1))

    def test_no_reversal_when_the_current_strategy_is_also_faster_per_sample(self):
        self.assertIsNone(reversal_gns_threshold(8, 1.0, 16, 1.0))
        self.assertFalse(previous_is_preferable(8, 1.0, 16, 1.0, 0.0))

    def test_no_reversal_when_the_current_strategy_has_lower_throughput(self):
        self.assertIsNone(reversal_gns_threshold(8, 1.0, 16, 4.0))

    def test_rejects_non_positive_times(self):
        with self.assertRaises(ValueError):
            reversal_gns_threshold(8, 0.0, 16, 1.2)
        with self.assertRaises(ValueError):
            previous_is_preferable(8, 1.0, 16, 1.2, -1.0)


class FallbackMonitorTests(unittest.TestCase):
    def test_a_sustained_decrease_fires_exactly_at_the_patience(self):
        monitor = GnsFallbackMonitor(ema_alpha=0.5, patience=3)
        self.assertFalse(monitor.observe(100.0))
        self.assertFalse(monitor.observe(90.0))
        self.assertFalse(monitor.observe(80.0))
        self.assertTrue(monitor.observe(70.0))
        self.assertEqual(monitor.consecutive_decreases, 3)

    def test_one_iteration_without_a_decrease_resets_the_streak(self):
        monitor = GnsFallbackMonitor(ema_alpha=0.5, patience=3)
        monitor.observe(100.0)
        monitor.observe(90.0)
        monitor.observe(80.0)
        self.assertFalse(monitor.observe(1000.0))
        self.assertEqual(monitor.consecutive_decreases, 0)

    def test_a_transient_dip_is_absorbed_by_the_ema(self):
        monitor = GnsFallbackMonitor(ema_alpha=0.5, patience=3)
        for gns in [100.0, 100.0, 10.0, 100.0, 100.0, 100.0, 100.0]:
            self.assertFalse(monitor.observe(gns))

    def test_missing_or_invalid_readings_are_skipped(self):
        monitor = GnsFallbackMonitor(ema_alpha=0.5, patience=2)
        monitor.observe(100.0)
        self.assertFalse(monitor.observe(None))
        self.assertFalse(monitor.observe(float("nan")))
        self.assertFalse(monitor.observe(-1.0))
        self.assertEqual(monitor.observations, 1)
        self.assertEqual(monitor.consecutive_decreases, 0)

    def test_reset_forgets_the_streak_and_the_smoothed_value(self):
        monitor = GnsFallbackMonitor(ema_alpha=0.5, patience=2)
        monitor.observe(100.0)
        monitor.observe(90.0)
        monitor.reset()
        self.assertIsNone(monitor.ema)
        self.assertEqual(monitor.consecutive_decreases, 0)

    def test_rejects_invalid_settings(self):
        with self.assertRaises(ValueError):
            GnsFallbackMonitor(ema_alpha=0.0)
        with self.assertRaises(ValueError):
            GnsFallbackMonitor(ema_alpha=1.5)
        with self.assertRaises(ValueError):
            GnsFallbackMonitor(patience=0)


class SearchFallbackTests(unittest.TestCase):
    def setUp(self):
        self.model = gpt2_medium()
        self.cluster = ClusterSpec.from_gib(num_gpus=4, gpu_memory_gib=80)
        self.time_model = CalculonTimeModel(
            self.model, system="a100_80g", vocab_size=VOCAB
        )

    def _search(self, **kwargs):
        kwargs.setdefault("use_botorch", False)
        kwargs.setdefault("initial_gbs", 8)
        kwargs.setdefault("max_gbs", 64)
        return OnlineStrategySearch(
            self.model, self.cluster, self.time_model, **kwargs
        )

    def test_no_fallback_by_default(self):
        search = self._search()
        self.assertIsNone(search.fallback)
        for gns in [1e6] + [1.0] * 500:
            decision = search.observe(gns=gns)
            self.assertFalse(decision.fell_back)
        self.assertEqual([s.gbs for s in search.chain], [8, 16])

    def test_sustained_decrease_reverts_to_the_previous_strategy(self):
        search = self._search(
            fallback=GnsFallbackMonitor(ema_alpha=0.5, patience=3)
        )
        search.observe(gns=1e6)
        self.assertEqual(search.current.gbs, 16)
        previous = search.previous
        self.assertEqual(previous.gbs, 8)

        gns = 1.0
        for _ in range(10):
            gns *= 0.5
            decision = search.observe(gns=gns)
            self.assertFalse(decision.switched)
            if decision.fell_back:
                break
        self.assertTrue(decision.fell_back)
        self.assertEqual(search.current, previous)
        self.assertEqual([s.gbs for s in search.chain], [8, 16, 8])
        self.assertEqual(search.candidate.gbs, 16)
        self.assertEqual(search.fallback.consecutive_decreases, 0)

    def test_fallback_before_any_switch_is_a_no_op(self):
        search = self._search(
            fallback=GnsFallbackMonitor(ema_alpha=0.5, patience=2)
        )
        self.assertIsNone(search.previous)
        self.assertIsNone(search.fall_back())
        gns = 1.0
        for _ in range(20):
            gns *= 0.5
            decision = search.observe(gns=gns)
            self.assertFalse(decision.fell_back)
        self.assertEqual([s.gbs for s in search.chain], [8])


if __name__ == "__main__":
    unittest.main()
