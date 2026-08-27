#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import ConfigManager, apply_model_preset
from gpt_backend import TrainingExecutor, TrainingStep
from conversion import ConversionExecutor
from pruning import prune_universal_checkpoints
from distributed_barrier import (
    clear_step_markers,
    post_train_exit_ok,
    wait_all_train_exit_ok,
)


def _is_leader(no_ssh: bool, node_rank: Optional[int]) -> bool:
    if not no_ssh:
        return True
    return node_rank in (None, 0)


def _wait_for_path(path: Path, timeout_sec: int, poll_sec: float = 5.0) -> bool:
    print(f"[INFO] Waiting for {path} (timeout={timeout_sec}s)")
    start = time.time()
    parent = path.parent
    while time.time() - start < timeout_sec:
        try:
            list(parent.iterdir())
        except OSError:
            pass
        try:
            with open(path, "r") as f:
                f.read()
            return True
        except OSError:
            pass
        time.sleep(poll_sec)
    print(f"[ERR] Timeout waiting for {path}")
    return False


def run_online_search(
    config: ConfigManager,
    trainer: TrainingExecutor,
    converter: ConversionExecutor,
    workdir: str,
    *,
    gpus: int,
    gpu_memory_gib: int,
    system: str,
    initial_gbs: int,
    max_gbs: Optional[int],
    max_mbs: int,
    total_iters: int,
    poll_seconds: float,
    vocab_size: int,
    use_botorch: bool,
    seed: int,
    out: Optional[str],
) -> bool:
    from fallback import GnsFallbackMonitor
    from feasibility import ClusterSpec
    from perf_model import CalculonTimeModel
    from strategy_search import OnlineStrategySearch

    cluster = ClusterSpec(
        num_gpus=gpus, gpu_memory_bytes=gpu_memory_gib * (1024 ** 3)
    )
    time_model = CalculonTimeModel(
        config.model, system=system, vocab_size=vocab_size
    )
    fallback_enabled = bool(getattr(config.training, "fallback", False))
    monitor = (
        GnsFallbackMonitor(
            ema_alpha=config.training.fallback_ema_alpha,
            patience=config.training.fallback_patience,
        )
        if fallback_enabled
        else None
    )
    search = OnlineStrategySearch(
        config.model,
        cluster,
        time_model,
        initial_gbs=initial_gbs,
        vocab_size=vocab_size,
        max_gbs=max_gbs,
        max_mbs=max_mbs,
        seed=seed,
        use_botorch=use_botorch,
        fallback=monitor,
    )
    trainer.watch_poll_seconds = poll_seconds

    print(f"[INFO] Online search on {gpus} GPUs, system={system}")
    print(f"[INFO] Initial strategy: {search.current}")
    if fallback_enabled:
        print(
            f"[INFO] Fallback enabled: reverting after "
            f"{config.training.fallback_patience} consecutive decreases of "
            f"GNS smoothed at alpha={config.training.fallback_ema_alpha}"
        )

    completed = 0
    step_num = 1
    previous = None
    measured_gns: Optional[float] = None
    stage_gns: List[Optional[float]] = []
    boundary_iters: List[int] = []
    switch_iters: List[int] = []
    fallback_iters: List[int] = []

    while completed < total_iters:
        strategy = search.current
        is_initial = previous is None
        ckpt_dir = Path(
            config.get_checkpoint_dir(
                strategy.dp, strategy.tp, strategy.pp, workdir
            )
        )

        print(
            f"\n[INFO] Training from iter {completed} to {total_iters} "
            f"under {strategy}"
        )
        step = TrainingStep(
            step_num=step_num,
            dp=strategy.dp,
            tp=strategy.tp,
            pp=strategy.pp,
            gbs=strategy.gbs,
            mbs=strategy.mbs,
            target_iters=total_iters,
            load_dp=None if is_initial else previous.dp,
            load_tp=None if is_initial else previous.tp,
            load_pp=None if is_initial else previous.pp,
            master_port=29700 + (step_num - 1) % 100,
            is_initial=is_initial,
            prev_gbs=None if search.previous is None else search.previous.gbs,
            prev_iteration_seconds=search.previous_time,
            gns=measured_gns,
        )

        stage_gns.append(measured_gns)
        publisher = _CandidatePublisher(search, trainer)
        if not trainer.execute(step, watcher=publisher):
            return False

        previous = strategy
        if trainer.last_gns is not None:
            measured_gns = trainer.last_gns
        reached = _checkpointed_iteration(ckpt_dir)
        if reached is None:
            print(f"[ERR] No checkpoint iteration recorded under {ckpt_dir}")
            return False
        completed = reached

        if not trainer.stopped_for_strategy_change:
            if completed < total_iters:
                print(
                    f"[ERR] Job ended at iter {completed} before {total_iters} "
                    "without a switch request"
                )
                return False
            break

        if trainer.stopped_for_fallback:
            if search.fall_back() is None:
                print(
                    "[ERR] The job asked to fall back, but the run has not "
                    "passed any earlier strategy to revert to"
                )
                return False
            print(f"[INFO] Fell back to {search.current} at iter {completed}")
            fallback_iters.append(completed)
        else:
            if publisher.candidate is None:
                print("[ERR] The job asked to switch but no candidate was published")
                return False
            search.adopt(publisher.candidate, publisher.candidate_time)
            print(f"[INFO] Switched to {search.current} at iter {completed}")
            switch_iters.append(completed)
        boundary_iters.append(completed)

        if not converter.convert_to_universal(
            step_num=step_num,
            dp=strategy.dp,
            tp=strategy.tp,
            pp=strategy.pp,
            use_relaxed=step_num > 1,
        ):
            return False
        step_num += 1

    print("\n[OK] Online search completed.")
    print(f"[OK] Chain: {' -> '.join(str(s) for s in search.chain)}")
    print(f"[OK] Switched at iterations: {switch_iters or 'never'}")
    if fallback_enabled:
        print(f"[OK] Fell back at iterations: {fallback_iters or 'never'}")
    print(f"[OK] Search observations: {search.iteration}")

    if out:
        boundaries = boundary_iters + [total_iters]
        steps = []
        for i, s in enumerate(search.chain):
            entry = {
                "step_num": i + 1,
                "dp": s.dp,
                "tp": s.tp,
                "pp": s.pp,
                "gbs": s.gbs,
                "mbs": s.mbs,
                "target_iters": boundaries[i],
                "is_initial": i == 0,
                "convert": i < len(boundary_iters),
            }
            if i > 0:
                previous_stage = search.chain[i - 1]
                entry["load_dp"] = previous_stage.dp
                entry["load_tp"] = previous_stage.tp
                entry["load_pp"] = previous_stage.pp
            if i < len(stage_gns) and stage_gns[i] is not None:
                entry["gns"] = stage_gns[i]
            steps.append(entry)
        chain = {"steps": steps}
        Path(out).write_text(json.dumps(chain, indent=2))
        print(f"[OK] Wrote the executed chain to {out}")

    return True


class _CandidatePublisher:
    def __init__(self, search, trainer: TrainingExecutor):
        self.search = search
        self.trainer = trainer
        self.candidate = None
        self.candidate_time = None
        self._last_refine = None

    def _refine_steps(self, now: float) -> int:
        iteration_seconds = self.trainer.last_iteration_seconds
        elapsed = None if self._last_refine is None else now - self._last_refine
        self._last_refine = now
        if not elapsed or not iteration_seconds or iteration_seconds <= 0:
            return 1
        return max(1, min(int(elapsed / iteration_seconds),
                          self.search.max_gp_updates))

    def __call__(self) -> bool:
        from switch_rule import write_candidate

        candidate = None
        for _ in range(self._refine_steps(time.time())):
            candidate = self.search.search_by_bo()
        candidate_time = self.search.candidate_time
        if candidate is None or candidate_time is None:
            print(f"[SEARCH] cur={self.search.current} candidate=<no larger feasible frontier>")
            return False

        path = self.trainer.candidate_file
        if path is None:
            return False
        write_candidate(
            str(path),
            {
                "dp": candidate.dp,
                "tp": candidate.tp,
                "pp": candidate.pp,
                "mbs": candidate.mbs,
                "gbs": candidate.gbs,
                "t_next_sec": candidate_time,
            },
        )
        self.candidate, self.candidate_time = candidate, candidate_time
        frontier = self.search._frontier
        print(
            f"[SEARCH] cur={self.search.current} next={candidate} "
            f"T_next={candidate_time * 1e3:.1f}ms "
            f"gp_updates={0 if frontier is None else frontier.updates} published"
        )
        return False


def _checkpointed_iteration(ckpt_dir: Path) -> Optional[int]:
    marker = ckpt_dir / "latest_checkpointed_iteration.txt"
    try:
        return int(marker.read_text().strip())
    except (OSError, ValueError):
        return None


def run_chain_from_json(
    config: ConfigManager,
    trainer: TrainingExecutor,
    converter: ConversionExecutor,
    json_file: str,
    repo_path: str,
    workdir: str,
    num_nodes: Optional[int] = None,
    gpus_per_node: Optional[int] = None,
    hostfile: Optional[str] = None,
    master_addr: Optional[str] = None,
    no_ssh: bool = False,
    node_rank: Optional[int] = None,
) -> bool:
    with open(json_file, "r") as f:
        chain_config: Dict[str, Any] = json.load(f)

    steps_config = chain_config.get("steps", [])
    pruning_config = chain_config.get("pruning", [])

    print(f"[INFO] Running chain from {json_file}")
    print(f"[INFO] Total steps: {len(steps_config)}")
    if pruning_config:
        print(f"[INFO] Pruning steps: {len(pruning_config)}")
    print()

    nn = num_nodes if num_nodes is not None else 1

    for i, step_cfg in enumerate(steps_config):
        step_num = step_cfg.get("step_num", i + 1)
        convert_after = step_cfg.get("convert", True)

        step = TrainingStep(
            step_num=step_num,
            dp=step_cfg["dp"],
            tp=step_cfg["tp"],
            pp=step_cfg["pp"],
            gbs=step_cfg["gbs"],
            mbs=step_cfg["mbs"],
            train_iters=step_cfg.get("train_iters"),
            extra_iters=step_cfg.get("extra_iters"),
            target_iters=step_cfg.get("target_iters"),
            load_dp=step_cfg.get("load_dp"),
            load_tp=step_cfg.get("load_tp"),
            load_pp=step_cfg.get("load_pp"),
            gns=step_cfg.get("gns"),
            master_port=step_cfg.get("master_port", 29700 + step_num - 1),
            master_addr=step_cfg.get("master_addr", master_addr),
            num_nodes=step_cfg.get("num_nodes", nn),
            gpus_per_node=step_cfg.get("gpus_per_node", gpus_per_node),
            hostfile=step_cfg.get("hostfile", hostfile),
            no_ssh=step_cfg.get("no_ssh", no_ssh),
            node_rank=step_cfg.get("node_rank", node_rank),
            is_initial=step_cfg.get("is_initial", False),
        )

        sn = step.num_nodes or 1
        if step.no_ssh and sn > 1:
            clear_step_markers(workdir, step_num, sn)

        if not trainer.execute(step):
            return False

        if step.no_ssh and sn > 1 and step.node_rank is not None:
            post_train_exit_ok(workdir, step_num, step.node_rank)
            if not wait_all_train_exit_ok(workdir, step_num, sn, timeout_sec=7200):
                return False

        if convert_after:
            use_relaxed = step_num >= 2
            ckpt_dir = config.get_checkpoint_dir(step.dp, step.tp, step.pp, workdir)
            latest_universal = Path(ckpt_dir) / "latest_universal"

            if _is_leader(step.no_ssh, step.node_rank):
                if not converter.convert_to_universal(
                    step_num=step_num,
                    dp=step.dp,
                    tp=step.tp,
                    pp=step.pp,
                    use_relaxed=use_relaxed,
                ):
                    return False
            else:
                if not _wait_for_path(latest_universal, timeout_sec=7200):
                    return False

        for prune_cfg in pruning_config:
            after_step = prune_cfg.get("after_step")
            if after_step and step_num == after_step:
                if not _is_leader(step.no_ssh, step.node_rank):
                    continue
                ckpt_dir = prune_cfg.get("ckpt_dir")
                if ckpt_dir.startswith("/"):
                    full_ckpt_dir = ckpt_dir
                else:
                    full_ckpt_dir = f"{workdir}/{ckpt_dir}"
                mode = prune_cfg.get("mode", "keep-latest")
                print(f"\n[INFO] Pruning after step {step_num}")
                prune_universal_checkpoints(full_ckpt_dir, mode)

    print("\n[OK] Chain completed successfully!")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Universal Checkpoint Training Runner")
    parser.add_argument("--config", type=str, help="Path to .conaconfig")
    parser.add_argument("--model-preset", type=str, default=None)
    parser.add_argument("--repo-path", type=str, default="/workspace/Megatron-DeepSpeed")
    parser.add_argument("--workdir", type=str, default="/workspace/cona")
    parser.add_argument("--chain", type=str, metavar="JSON_FILE", help="Chain config")

    search = parser.add_argument_group("strategy search")
    search.add_argument("--gpus", type=int, default=None,
                        help="Cluster size; defaults to num_nodes * gpus_per_node")
    search.add_argument("--gpu-memory-gib", type=int, default=48)
    search.add_argument("--system", type=str, default="a6000_48g",
                        help="Calculon system in cona/config/calculon/systems")
    search.add_argument("--initial-gbs", type=int, default=8)
    search.add_argument("--max-gbs", type=int, default=None)
    search.add_argument("--max-mbs", type=int, default=32)
    search.add_argument("--total-iters", type=int, default=10000)
    search.add_argument("--poll-seconds", type=float, default=20.0,
                        help="How often the search consumes new GNS rows while training runs")
    search.add_argument("--vocab-size", type=int, default=50257)
    search.add_argument("--no-botorch", action="store_true",
                        help="Score each frontier exhaustively instead of with BO")
    search.add_argument("--seed", type=int, default=0)
    search.add_argument("--out", type=str, default=None,
                        help="Write the executed chain as a replayable chain config")

    parser.add_argument("--dp", type=int)
    parser.add_argument("--tp", type=int)
    parser.add_argument("--pp", type=int)
    parser.add_argument("--gbs", type=int)
    parser.add_argument("--mbs", type=int)
    parser.add_argument("--gns", type=float, default=None,
                        help="Gradient noise scale for lr_scale_strategy 'adascale'")
    parser.add_argument("--iters", type=int)
    parser.add_argument("--extra-iters", type=int)
    parser.add_argument("--target-iters", type=int)
    parser.add_argument("--load-dp", type=int)
    parser.add_argument("--load-tp", type=int)
    parser.add_argument("--load-pp", type=int)
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--master-port", type=int, default=29700)
    parser.add_argument("--master-addr", type=str, default=None)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=None)
    parser.add_argument("--hostfile", type=str, default=None)
    parser.add_argument("--no-ssh", action="store_true")
    parser.add_argument("--node-rank", type=int, default=None)
    parser.add_argument("--step-num", type=int, default=1)

    args = parser.parse_args()
    workdir_abs = str(Path(args.workdir).resolve())

    config = ConfigManager(config_path=args.config)
    if args.model_preset:
        apply_model_preset(config.model, args.model_preset)
    trainer = TrainingExecutor(config, args.repo_path, workdir_abs)
    converter = ConversionExecutor(config, workdir_abs)

    fixed_strategy = any([args.dp, args.tp, args.pp, args.gbs, args.mbs])
    if not args.chain and not fixed_strategy:
        gpus = args.gpus or (args.num_nodes * (args.gpus_per_node or 0))
        if gpus <= 0:
            parser.error(
                "The strategy search needs the cluster size: pass --gpus, or "
                "--num-nodes with --gpus-per-node"
            )
        ok = run_online_search(
            config,
            trainer,
            converter,
            workdir_abs,
            gpus=gpus,
            gpu_memory_gib=args.gpu_memory_gib,
            system=args.system,
            initial_gbs=args.initial_gbs,
            max_gbs=args.max_gbs,
            max_mbs=args.max_mbs,
            total_iters=args.total_iters,
            poll_seconds=args.poll_seconds,
            vocab_size=args.vocab_size,
            use_botorch=not args.no_botorch,
            seed=args.seed,
            out=args.out,
        )
        sys.exit(0 if ok else 1)

    if args.chain:
        ok = run_chain_from_json(
            config,
            trainer,
            converter,
            args.chain,
            args.repo_path,
            workdir_abs,
            num_nodes=args.num_nodes,
            gpus_per_node=args.gpus_per_node,
            hostfile=args.hostfile,
            master_addr=args.master_addr,
            no_ssh=args.no_ssh,
            node_rank=args.node_rank,
        )
        sys.exit(0 if ok else 1)

    if not all([args.dp, args.tp, args.pp, args.gbs, args.mbs]):
        parser.error("A fixed strategy needs all of --dp, --tp, --pp, --gbs, --mbs")
    if not any([args.iters, args.extra_iters, args.target_iters]):
        parser.error("Must specify one of --iters, --extra-iters, or --target-iters")

    is_initial = not any([args.load_dp, args.load_tp, args.load_pp])
    load_dp = args.load_dp if args.load_dp is not None else (None if is_initial else args.dp)
    load_tp = args.load_tp if args.load_tp is not None else (None if is_initial else args.tp)
    load_pp = args.load_pp if args.load_pp is not None else (None if is_initial else args.pp)

    step = TrainingStep(
        step_num=args.step_num,
        dp=args.dp,
        tp=args.tp,
        pp=args.pp,
        gbs=args.gbs,
        mbs=args.mbs,
        train_iters=args.iters,
        extra_iters=args.extra_iters,
        target_iters=args.target_iters,
        load_dp=load_dp,
        load_tp=load_tp,
        load_pp=load_pp,
        gns=args.gns,
        master_port=args.master_port,
        master_addr=args.master_addr,
        num_nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
        hostfile=args.hostfile,
        no_ssh=args.no_ssh,
        node_rank=args.node_rank,
        is_initial=is_initial,
    )

    nn = args.num_nodes or 1
    if args.no_ssh and nn > 1:
        clear_step_markers(workdir_abs, args.step_num, nn)

    if not trainer.execute(step):
        sys.exit(1)

    if args.no_ssh and nn > 1 and args.node_rank is not None:
        post_train_exit_ok(workdir_abs, args.step_num, args.node_rank)
        if not wait_all_train_exit_ok(workdir_abs, args.step_num, nn, timeout_sec=7200):
            sys.exit(1)

    if args.convert:
        if _is_leader(args.no_ssh, args.node_rank):
            if not converter.convert_to_universal(
                step_num=args.step_num,
                dp=args.dp,
                tp=args.tp,
                pp=args.pp,
                use_relaxed=(not is_initial),
            ):
                sys.exit(1)
        else:
            ckpt_dir = config.get_checkpoint_dir(args.dp, args.tp, args.pp, workdir_abs)
            latest_universal = Path(ckpt_dir) / "latest_universal"
            if not _wait_for_path(latest_universal, timeout_sec=7200):
                sys.exit(1)

    print("\n[OK] Training completed successfully!")
    sys.exit(0)


if __name__ == "__main__":
    main()
