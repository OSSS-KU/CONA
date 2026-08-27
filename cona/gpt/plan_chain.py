#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import ConfigManager, apply_model_preset
from feasibility import ClusterSpec, Strategy, enumerate_strategies
from perf_model import CalculonTimeModel, ExecutionOptions, PerfModelError
from strategy_search import OnlineStrategySearch


def load_gns_trace(path: str) -> List[Tuple[int, float]]:
    """Read ``(iteration, gns)`` pairs from a CSV or JSON trace."""
    file_path = Path(path)
    if not file_path.is_file():
        raise SystemExit(f"[ERR] GNS trace not found: {file_path}")

    if file_path.suffix.lower() == ".json":
        rows = json.loads(file_path.read_text())
        if not isinstance(rows, list):
            raise SystemExit("[ERR] JSON trace must be a list of objects")
    else:
        with open(file_path, newline="") as handle:
            rows = list(csv.DictReader(handle))

    trace: List[Tuple[int, float]] = []
    for index, row in enumerate(rows):
        gns = _row_gns(row)
        if gns is None:
            continue
        iteration = row.get("iteration") or row.get("iteration #") or index + 1
        trace.append((int(float(iteration)), gns))
    if not trace:
        raise SystemExit(
            "[ERR] No usable rows: need a 'gns' column, or 'grad_var' and 'grad_sqr'"
        )
    trace.sort(key=lambda item: item[0])
    return trace


def _row_gns(row: Dict[str, object]) -> Optional[float]:
    def value(*names: str) -> Optional[float]:
        for name in names:
            raw = row.get(name)
            if raw not in (None, ""):
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    return None
        return None

    direct = value("gns", "GNS", "adaptdl/gns")
    if direct is not None and direct >= 0:
        return direct
    grad_var = value("grad_var", "adaptdl/grad_var")
    grad_sqr = value("grad_sqr", "adaptdl/grad_sqr")
    if grad_var is not None and grad_sqr is not None and grad_sqr > 0:
        return max(grad_var / grad_sqr, 0.0)
    return None


def build_chain(
    search: OnlineStrategySearch,
    trace: List[Tuple[int, float]],
    total_iters: int,
    verbose: bool = True,
) -> List[Dict[str, object]]:
    """Replay the trace and return chain steps with their iteration spans."""
    switch_points: List[Tuple[int, Strategy]] = [(0, search.current)]
    for iteration, gns in trace:
        decision = search.observe(gns=gns)
        if verbose and decision.switched:
            print(f"[INFO] train_iter={iteration} {decision.summary()}")
        if decision.switched:
            switch_points.append((iteration, search.current))

    steps: List[Dict[str, object]] = []
    for index, (start, strategy) in enumerate(switch_points):
        end = (
            switch_points[index + 1][0]
            if index + 1 < len(switch_points)
            else total_iters
        )
        if end <= start:
            continue
        step: Dict[str, object] = {
            "step_num": index + 1,
            "dp": strategy.dp,
            "tp": strategy.tp,
            "pp": strategy.pp,
            "gbs": strategy.gbs,
            "mbs": strategy.mbs,
        }
        if index == 0:
            step["train_iters"] = end
            step["is_initial"] = True
        else:
            previous = switch_points[index - 1][1]
            step["extra_iters"] = end - start
            step["load_dp"] = previous.dp
            step["load_tp"] = previous.tp
            step["load_pp"] = previous.pp
        step["convert"] = index + 1 < len(switch_points)
        steps.append(step)
    return steps


def report_space(
    model, cluster: ClusterSpec, time_model: CalculonTimeModel, gbs: int, vocab: int
) -> None:
    """Print the feasible set at one batch size with its analytical times."""
    candidates = enumerate_strategies(model, cluster, gbs, vocab)
    scored = []
    for candidate in candidates:
        seconds = time_model.try_iteration_time(candidate)
        if seconds is not None:
            scored.append((seconds, candidate))
    scored.sort()
    rejected = len(candidates) - len(scored)
    print(
        f"\n[INFO] S(gbs={gbs}): {len(candidates)} pass the feasibility constraints, "
        f"{len(scored)} scored by Calculon"
        + (f" ({rejected} rejected by its own model)" if rejected else "")
    )
    for seconds, candidate in scored:
        print(f"        {candidate}  T(s)={seconds * 1e3:9.2f} ms")
    if not scored:
        print("        (Calculon rejected every candidate)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan a CONA strategy chain with the online search"
    )
    parser.add_argument("--config", type=str, help="Path to .conaconfig")
    parser.add_argument("--model-preset", type=str, default=None)
    parser.add_argument(
        "--gpus", type=int, required=True, help="Total GPUs (dp * tp * pp)"
    )
    parser.add_argument(
        "--system",
        type=str,
        default="a6000_48g",
        help="Calculon system file name or path",
    )
    parser.add_argument("--initial-gbs", type=int, default=8)
    parser.add_argument("--max-gbs", type=int, default=256)
    parser.add_argument("--max-mbs", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument(
        "--activation-recompute",
        choices=["none", "attn_only", "full"],
        default="full",
    )
    parser.add_argument(
        "--gns",
        type=float,
        default=None,
        help="Constant GNS_t, for planning without a measured trace",
    )
    parser.add_argument(
        "--gns-trace",
        type=str,
        default=None,
        help="CSV/JSON trace of measured GNS (or grad_var and grad_sqr)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Iterations to simulate when --gns is used",
    )
    parser.add_argument("--total-iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-botorch",
        action="store_true",
        help="Skip the GP/EI refinement and score the frontier exhaustively",
    )
    parser.add_argument(
        "--show-space",
        action="store_true",
        help="Print the feasible set and analytical T(s) at every frontier",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Write the chain config here"
    )
    args = parser.parse_args()

    if (args.gns is None) == (args.gns_trace is None):
        parser.error("Specify exactly one of --gns or --gns-trace")

    config = ConfigManager(config_path=args.config)
    if args.model_preset:
        apply_model_preset(config.model, args.model_preset)

    try:
        time_model = CalculonTimeModel(
            config.model,
            system=args.system,
            vocab_size=args.vocab_size,
            options=ExecutionOptions(
                activation_recompute=args.activation_recompute
            ),
        )
    except PerfModelError as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1

    cluster = ClusterSpec(
        num_gpus=args.gpus, gpu_memory_bytes=time_model.device_memory_bytes
    )
    print(
        f"[INFO] model={config.model.name} layers={config.model.layers} "
        f"hidden={config.model.hidden} heads={config.model.heads} "
        f"seq={config.model.seq}"
    )
    print(
        f"[INFO] cluster={args.gpus} GPUs, system={time_model.system_path.name}, "
        f"{time_model.device_memory_bytes / 1024 ** 3:.0f} GiB/GPU"
    )

    if args.show_space:
        gbs = args.initial_gbs
        while gbs <= args.max_gbs:
            report_space(
                config.model, cluster, time_model, gbs, args.vocab_size
            )
            gbs *= 2
        print()

    try:
        search = OnlineStrategySearch(
            config.model,
            cluster,
            time_model,
            initial_gbs=args.initial_gbs,
            vocab_size=args.vocab_size,
            max_gbs=args.max_gbs,
            max_mbs=args.max_mbs,
            seed=args.seed,
            use_botorch=not args.no_botorch,
        )
    except (ValueError, PerfModelError) as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1

    if args.gns_trace:
        trace = load_gns_trace(args.gns_trace)
        total_iters = args.total_iters or trace[-1][0]
    else:
        trace = [(i + 1, args.gns) for i in range(args.iters)]
        total_iters = args.total_iters

    steps = build_chain(search, trace, total_iters)

    print("\n[INFO] Strategy chain:")
    for strategy in search.chain:
        print(f"        {strategy}")

    chain_config = {"steps": steps, "pruning": []}
    payload = json.dumps(chain_config, indent=2)
    if args.out:
        Path(args.out).write_text(payload + "\n")
        print(f"\n[OK] Wrote {args.out}")
    else:
        print("\n" + payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
