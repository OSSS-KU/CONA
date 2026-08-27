#!/usr/bin/env python3
"""Training step execution (DeepSpeed / Megatron). Multi-node aware."""

import os
import sys
import json
import time
import re
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional
from dataclasses import dataclass

from lr_scaling import scale_lr
from config import ConfigManager

_ITER_TIME_RE = re.compile(r"elapsed time per iteration \(ms\):\s*([0-9.]+)")


def _marker_gns(path: Path) -> Optional[float]:
    """``GNS_t`` recorded on the second line of a switch or fallback marker."""
    try:
        lines = path.read_text().split()
    except OSError:
        return None
    if len(lines) < 2:
        return None
    try:
        gns = float(lines[1])
    except ValueError:
        return None
    return gns if gns >= 0 else None


def _wait_for_universal_checkpoint(
    load_dir: Path, timeout_sec: int = 7200, poll_sec: float = 3.0
) -> bool:
    """NFS-safe wait: all nodes must see latest_universal before launching DeepSpeed."""
    marker = load_dir / "latest_universal"
    print(f"[INFO] Waiting for universal checkpoint marker: {marker} (timeout={timeout_sec}s)")
    start = time.time()
    last_log = 0.0
    while time.time() - start < timeout_sec:
        try:
            list(load_dir.iterdir())
        except OSError:
            pass
        try:
            if marker.exists():
                with open(marker, "r") as f:
                    f.read()
                print(f"[INFO] Found readable {marker}")
                return True
        except OSError:
            pass
        now = time.time()
        if now - last_log >= 30.0:
            print(f"[INFO] Still waiting for {marker} ...")
            last_log = now
        time.sleep(poll_sec)
    print(f"[ERR] Timeout waiting for {marker}")
    return False


@dataclass
class TrainingStep:
    step_num: int
    dp: int
    tp: int
    pp: int
    gbs: int
    mbs: int
    train_iters: Optional[int] = None
    extra_iters: Optional[int] = None
    load_dp: Optional[int] = None
    load_tp: Optional[int] = None
    load_pp: Optional[int] = None
    master_port: int = 29700
    master_addr: Optional[str] = None
    num_nodes: int = 1
    gpus_per_node: Optional[int] = None
    hostfile: Optional[str] = None
    no_ssh: bool = False
    node_rank: Optional[int] = None
    is_initial: bool = False
    target_iters: Optional[int] = None
    prev_gbs: Optional[int] = None
    prev_iteration_seconds: Optional[float] = None
    gns: Optional[float] = None


class TrainingExecutor:
    def __init__(
        self,
        config: ConfigManager,
        repo_path: str = "/workspace/Megatron-DeepSpeed",
        workdir: str = "/workspace/cona",
    ):
        self.config = config
        self.repo_path = repo_path
        self.workdir = workdir
        self.last_gns_csv: Optional[Path] = None
        self.last_gns: Optional[float] = None
        self.last_iteration_seconds: Optional[float] = None
        self.stopped_for_switch = False
        self.stopped_for_fallback = False
        self.watch_poll_seconds = 20.0
        self._watch_stop = threading.Event()
        self.candidate_file: Optional[Path] = None
        os.environ.setdefault("NCCL_DEBUG", "WARN")
        os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,ENV")
        os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")

    @property
    def stopped_for_strategy_change(self) -> bool:
        return self.stopped_for_switch or self.stopped_for_fallback

    def _watch_for_switch(self, process, watcher: Callable[[], bool]) -> None:
        """Run the CPU-side search beside the job for as long as it lasts.

        ``watcher`` publishes the search's current candidate to a file. The
        training loop reads it once per iteration and stops itself at an
        iteration boundary, where a checkpoint is safe to take.
        """
        warned = False
        while not self._watch_stop.wait(self.watch_poll_seconds):
            if process.poll() is not None:
                return
            try:
                watcher()
            except Exception as exc:  # a search failure must not kill training
                if not warned:
                    print(f"[WARN] Strategy search failed: {exc}")
                    warned = True

    def execute(
        self, step: TrainingStep, watcher: Optional[Callable[[], bool]] = None
    ) -> bool:
        """Run one training job.

        ``watcher`` is called on a background thread while the job runs; it
        publishes the search's candidate for the job to score. The search
        therefore runs beside training, and the job ends itself at an iteration
        boundary when the candidate wins.
        """
        self.last_gns_csv = None
        self.last_gns = None
        self.last_iteration_seconds = None
        self.stopped_for_switch = False
        self.stopped_for_fallback = False
        self._watch_stop = threading.Event()
        print(f"[INFO] Starting Step {step.step_num}")
        print(f"[INFO] Config: DP={step.dp}, TP={step.tp}, PP={step.pp}, GBS={step.gbs}, MBS={step.mbs}")

        os.makedirs(self.config.training.wandb_dir, exist_ok=True)
        os.makedirs(self.config.training.log_root, exist_ok=True)

        ts = datetime.now().strftime("%y%m%d_%H%M%S")
        log_dir = (
            f"{self.config.training.log_root}/{ts}_univ_ckpt_step{step.step_num}_dp{step.dp}"
            f"_tp{step.tp}_pp{step.pp}_gbs{step.gbs}_mbs{step.mbs}"
        )
        os.makedirs(log_dir, exist_ok=True)
        log_file = f"{log_dir}/train.log"

        results_root = Path(self.workdir).resolve() / "results"
        run_dir = results_root / f"gpt_dp{step.dp}_tp{step.tp}_pp{step.pp}_gbs{step.gbs}_mbs{step.mbs}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        efficiency_csv = run_dir / f"dp{step.dp}_tp{step.tp}_pp{step.pp}_efficiency.csv"

        self.candidate_file = run_dir / "search_candidate.json"
        switch_file = run_dir / "switch_taken"
        fallback_file = run_dir / "fallback_taken"
        for stale in (self.candidate_file, switch_file, fallback_file):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass

        same_layout = False
        if step.is_initial:
            ckpt_dir = self.config.get_checkpoint_dir(step.dp, step.tp, step.pp, self.workdir)
            load_dir = None
            start_iter = 0
        else:
            load_dp = step.load_dp or step.dp
            load_tp = step.load_tp or step.tp
            load_pp = step.load_pp or step.pp
            load_dir = self.config.get_checkpoint_dir(load_dp, load_tp, load_pp, self.workdir)
            load_dir_p = Path(load_dir)
            same_layout = (load_dp, load_tp, load_pp) == (step.dp, step.tp, step.pp)
            if not same_layout and not _wait_for_universal_checkpoint(load_dir_p):
                print(f"[ERR] latest_universal not available under: {load_dir}")
                return False
            start_iter = 0
            latest_iter_file = Path(load_dir) / "latest_checkpointed_iteration.txt"
            if latest_iter_file.exists():
                with open(latest_iter_file, "r") as f:
                    start_iter = int(f.read().strip())
            ckpt_dir = self.config.get_checkpoint_dir(step.dp, step.tp, step.pp, self.workdir)

        os.makedirs(ckpt_dir, exist_ok=True)

        if step.target_iters is not None:
            if start_iter >= step.target_iters:
                print(f"[ERR] START_ITER ({start_iter}) >= TARGET_ITERS ({step.target_iters}).")
                return False
            extra_iters = step.target_iters - start_iter
            train_iters = step.target_iters
        elif step.train_iters is not None:
            train_iters = step.train_iters
            extra_iters = step.extra_iters or (train_iters - start_iter)
        elif step.extra_iters is not None:
            train_iters = start_iter + step.extra_iters
            extra_iters = step.extra_iters
        else:
            print("[ERR] Must specify train_iters, extra_iters, or target_iters")
            return False

        lr = scale_lr(
            self.config.training.lr_scale_strategy,
            self.config.training.base_lr,
            self.config.training.base_gbs,
            step.gbs,
            gns=step.gns,
            exp=self.config.training.lr_scale_exp,
            pivot_gbs=self.config.training.lr_scale_pivot_gbs,
            low_exp=self.config.training.lr_scale_low_exp,
            high_exp=self.config.training.lr_scale_high_exp,
            lr_cap=self.config.training.lr_scale_lr_cap,
        )
        if self.config.training.scale_min_lr:
            min_lr = scale_lr(
                self.config.training.lr_scale_strategy,
                self.config.training.base_min_lr,
                self.config.training.base_gbs,
                step.gbs,
                gns=step.gns,
                exp=self.config.training.lr_scale_exp,
                pivot_gbs=self.config.training.lr_scale_pivot_gbs,
                low_exp=self.config.training.lr_scale_low_exp,
                high_exp=self.config.training.lr_scale_high_exp,
                lr_cap=self.config.training.lr_scale_lr_cap,
            )
        else:
            min_lr = self.config.training.base_min_lr

        world_size = step.dp * step.tp * step.pp
        gradient_accumulation_steps = step.gbs // (step.mbs * step.dp)

        num_nodes = step.num_nodes or 1
        if num_nodes < 1:
            print(f"[ERR] num_nodes must be >= 1 (got {num_nodes})")
            return False
        if num_nodes > 1 and not step.hostfile:
            print("[ERR] Multi-node run requires --hostfile")
            return False
        if step.no_ssh and step.node_rank is None:
            print("[ERR] --no-ssh requires --node-rank")
            return False
        if step.gpus_per_node is None:
            if world_size % num_nodes != 0:
                print(
                    f"[ERR] world_size {world_size} not divisible by num_nodes {num_nodes}. "
                    "Set --gpus-per-node or adjust dp/tp/pp/num-nodes."
                )
                return False
            gpus_per_node = world_size // num_nodes
        else:
            gpus_per_node = step.gpus_per_node
            if gpus_per_node <= 0 or gpus_per_node * num_nodes != world_size:
                print(
                    f"[ERR] gpus_per_node * num_nodes must equal world_size "
                    f"({gpus_per_node} * {num_nodes} != {world_size})"
                )
                return False

        workdir_abs = Path(self.workdir).resolve()
        ds_root = workdir_abs / "config" / "deepspeed_config" / "gpt"
        ds_root.mkdir(parents=True, exist_ok=True)
        ds_config_path = str(ds_root / f"ds_z{self.config.training.zero_stage}_step{step.step_num}.json")
        ds_config = {
            "train_batch_size": step.gbs,
            "train_micro_batch_size_per_gpu": step.mbs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "steps_per_print": 1,
            "zero_optimization": {"stage": self.config.training.zero_stage},
            "fp16": {"enabled": True, "loss_scale": 0, "initial_scale_power": 12},
            "wall_clock_breakdown": False,
        }
        with open(ds_config_path, "w") as f:
            json.dump(ds_config, f, indent=2)

        cmd = [
            "deepspeed",
            "--master_port",
            str(step.master_port),
        ]
        if step.master_addr:
            cmd.extend(["--master_addr", str(step.master_addr)])
        if step.hostfile:
            cmd.extend(["--hostfile", str(step.hostfile)])
        if step.no_ssh:
            cmd.append("--no_ssh")
            cmd.extend(["--node_rank", str(step.node_rank)])
        cmd.extend(
            [
                "--num_nodes",
                str(num_nodes),
                "--num_gpus",
                str(gpus_per_node),
                "pretrain_gpt.py",
                "--tensor-model-parallel-size",
                str(step.tp),
                "--pipeline-model-parallel-size",
                str(step.pp),
                "--ds-sequence-parallel-size",
                "1",
                "--num-layers",
                str(self.config.model.layers),
                "--hidden-size",
                str(self.config.model.hidden),
                "--ffn-hidden-size",
                str(getattr(self.config.model, "ffn_hidden", 4 * int(self.config.model.hidden))),
                "--num-attention-heads",
                str(self.config.model.heads),
                "--seq-length",
                str(self.config.model.seq),
                "--max-position-embeddings",
                str(self.config.model.seq),
                "--attention-dropout",
                str(getattr(self.config.model, "attention_dropout", 0.1)),
                "--hidden-dropout",
                str(getattr(self.config.model, "hidden_dropout", 0.1)),
                "--normalization",
                str(getattr(self.config.model, "normalization", "layernorm")),
                "--layernorm-epsilon",
                str(getattr(self.config.model, "layernorm_epsilon", 1e-5)),
                "--micro-batch-size",
                str(step.mbs),
                "--global-batch-size",
                str(step.gbs),
                "--train-iters",
                str(train_iters),
                "--lr",
                f"{lr:.10e}",
                "--min-lr",
                f"{min_lr:.10e}",
                "--lr-decay-style",
                "cosine",
                "--lr-decay-tokens",
                str(self.config.training.lr_decay_tokens),
                "--weight-decay",
                str(self.config.training.weight_decay),
                "--log-interval",
                str(self.config.training.log_interval),
                "--eval-interval",
                str(self.config.training.eval_interval),
                "--eval-iters",
                str(self.config.training.eval_iters),
                "--split",
                self.config.training.split,
                "--data-path",
                self.config.training.data_prefix,
            ]
        )
        warmup_tokens = getattr(self.config.training, "lr_warmup_tokens", None)
        if warmup_tokens is not None:
            cmd.extend(["--lr-warmup-tokens", str(warmup_tokens)])
        else:
            cmd.extend(
                ["--lr-warmup-fraction",
                 str(getattr(self.config.training, "lr_warmup_frac", 0.01))]
            )
        dcp = getattr(self.config.training, "data_cache_path", None)
        wandb_mode_cfg = getattr(self.config.training, "wandb_mode", None)
        if dcp:
            cmd.extend(["--data-cache-path", str(dcp)])
        cmd.extend(
            [
                "--vocab-file",
                self.config.training.vocab_file,
                "--merge-file",
                self.config.training.merge_file,
                "--tokenizer-type",
                self.config.training.tokenizer_type,
            ]
        )
        cmd.extend(
            [
                "--checkpoint-activations",
                "--save-interval",
                str(train_iters),
                "--exit-interval",
                str(train_iters),
                "--exit-signal-handler",
                "--save",
                ckpt_dir,
            ]
        )
        if watcher is not None:
            cmd.extend(
                [
                    "--cona-candidate-file",
                    str(self.candidate_file),
                    "--cona-switch-file",
                    str(switch_file),
                ]
            )
            if (
                getattr(self.config.training, "fallback", False)
                and step.prev_gbs
                and step.prev_iteration_seconds
            ):
                cmd.extend(
                    [
                        "--cona-fallback-file",
                        str(fallback_file),
                        "--cona-fallback-ema-alpha",
                        str(self.config.training.fallback_ema_alpha),
                        "--cona-fallback-patience",
                        str(self.config.training.fallback_patience),
                        "--cona-fallback-prev-gbs",
                        str(step.prev_gbs),
                        "--cona-fallback-prev-time-sec",
                        f"{step.prev_iteration_seconds:.9f}",
                    ]
                )
        if not step.is_initial:
            cmd.extend(["--load", load_dir])
        else:
            cmd.extend(["--load", ckpt_dir])
        cmd.extend(
            [
                "--make-vocab-size-divisible-by",
                "128",
                f"--{self.config.training.dtype}",
                "--distributed-timeout-minutes",
                str(self.config.training.distributed_timeout_minutes),
                "--deepspeed",
                "--deepspeed_config",
                ds_config_path,
                "--zero-stage",
                str(self.config.training.zero_stage),
            ]
        )
        if not step.is_initial:
            cmd.append("--override-opt_param-scheduler")
        if not step.is_initial and not same_layout:
            cmd.append("--universal-checkpoint")
        cmd.extend(
            [
                "--wandb-project",
                self.config.training.wandb_project,
                "--wandb-exp-name",
                f"step{step.step_num}-dp{step.dp}-tp{step.tp}-pp{step.pp}-gbs{step.gbs}-mbs{step.mbs}"
                f"-start{start_iter}-plus{extra_iters}-to{train_iters}-{ts}",
                "--wandb-save-dir",
                self.config.training.wandb_dir,
            ]
        )

        log_info = f"""[INFO] NCCL_DEBUG={os.environ.get('NCCL_DEBUG')}
[INFO] Logging to: {log_file}
[INFO] {'CKPT_DIR' if step.is_initial else 'LOAD_DIR'}: {load_dir or ckpt_dir}
[INFO] SAVE_DIR: {ckpt_dir}
[INFO] START_ITER={start_iter} EXTRA_ITERS={extra_iters} TRAIN_ITERS={train_iters}
[INFO] Gradient Accumulation Steps: {gradient_accumulation_steps}
[INFO] DIST: num_nodes={num_nodes} gpus_per_node={gpus_per_node} master_addr={step.master_addr or ''} """
        log_info += f"hostfile={step.hostfile or ''} no_ssh={step.no_ssh} node_rank={step.node_rank if step.no_ssh else ''}\n"
        log_info += (
            f"[INFO] LR: {lr:.6e} (min {min_lr:.6e}) from base_lr "
            f"{self.config.training.base_lr:.6e} at base_gbs "
            f"{self.config.training.base_gbs} by "
            f"{self.config.training.lr_scale_strategy}"
            + (f" at gns={step.gns:.4g}\n" if step.gns is not None else "\n")
        )
        if dcp:
            log_info += f"[INFO] data-cache-path: {dcp}\n"
        if wandb_mode_cfg:
            log_info += f"[INFO] WANDB_MODE (from config): {wandb_mode_cfg}\n"
        log_info += f"[INFO] Command: {' '.join(cmd)}\n\n"

        with open(log_file, "w") as f:
            f.write(log_info)
        print(log_info)

        try:
            os.chdir(self.repo_path)
            log_f = open(log_file, "a")

            iter_ms: list = []

            def write_to_both(data: str) -> None:
                log_f.write(data)
                log_f.flush()
                match = _ITER_TIME_RE.search(data)
                if match:
                    iter_ms.append(float(match.group(1)))
                    ordered = sorted(iter_ms)
                    self.last_iteration_seconds = ordered[len(ordered) // 2] / 1000.0
                try:
                    sys.stdout.write(data)
                    sys.stdout.flush()
                except BrokenPipeError:
                    pass

            env = os.environ.copy()
            if wandb_mode_cfg:
                env["WANDB_MODE"] = str(wandb_mode_cfg)
            env["MEGATRON_DISABLE_FUSED_KERNELS"] = "1"
            env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
            env.pop("TORCH_NCCL_BLOCKING_WAIT", None)
            env.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0")
            env.setdefault("NCCL_CUMEM_ENABLE", "0")
            env.setdefault("ADAPTDL_LOG_EFFICIENCY", "1")
            env.setdefault("ADAPTDL_INIT_BATCH_SIZE", "1")
            env["ADAPTDL_EFFICIENCY_CSV_FILE"] = str(efficiency_csv)
            rule_dir = Path(__file__).resolve().parent
            env["CONA_SWITCH_RULE"] = str(rule_dir / "switch_rule.py")
            env["CONA_FALLBACK_RULE"] = str(rule_dir / "fallback.py")
            adaptdl_root = Path("/workspace/utils/adaptdl/adaptdl")
            if adaptdl_root.exists():
                pp = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = f"{adaptdl_root}:{pp}" if pp else str(adaptdl_root)

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                start_new_session=(os.name != "nt"),
            )

            stop_thread = None
            if watcher is not None:
                stop_thread = threading.Thread(
                    target=self._watch_for_switch,
                    args=(process, watcher),
                    daemon=True,
                )
                stop_thread.start()

            assert process.stdout is not None
            for line in process.stdout:
                write_to_both(line)
            process.wait()
            if stop_thread is not None:
                self._watch_stop.set()
                stop_thread.join(timeout=5)
            log_f.close()
            self.stopped_for_fallback = fallback_file.exists()
            self.stopped_for_switch = switch_file.exists()
            if self.stopped_for_fallback:
                self.last_gns = _marker_gns(fallback_file)
            elif self.stopped_for_switch:
                self.last_gns = _marker_gns(switch_file)
            if self.stopped_for_fallback:
                print("[INFO] The job fell back to the previous strategy at an "
                      "iteration boundary")
            elif self.stopped_for_switch:
                print(f"[INFO] The job switched strategy at iteration boundary")
            if process.returncode != 0 and not self.stopped_for_strategy_change:
                raise subprocess.CalledProcessError(process.returncode, cmd)
            self.last_gns_csv = efficiency_csv if efficiency_csv.exists() else None
            print(f"[OK] Step {step.step_num} done. CKPT_DIR:")
            subprocess.run(["ls", "-la", ckpt_dir], check=False)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERR] Step {step.step_num} failed with exit code {e.returncode}")
            return False
