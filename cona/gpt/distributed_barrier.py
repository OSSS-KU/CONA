#!/usr/bin/env python3

import os
import time
from pathlib import Path


def barrier_dir(workdir: str) -> Path:
    d = Path(workdir).resolve() / ".cona_multi_barrier"
    d.mkdir(parents=True, exist_ok=True)
    return d


def clear_step_markers(workdir: str, step_num: int, num_nodes: int) -> None:
    d = barrier_dir(workdir)
    for i in range(num_nodes):
        p = d / f"step{step_num}_node{i}_train_exit_ok"
        try:
            p.unlink()
        except OSError:
            pass


def post_train_exit_ok(workdir: str, step_num: int, node_rank: int) -> None:
    d = barrier_dir(workdir)
    marker = d / f"step{step_num}_node{node_rank}_train_exit_ok"
    t = time.time()
    with open(marker, "w") as f:
        f.write(f"{t}\n")
        f.flush()
        os.fsync(f.fileno())
    try:
        fd = os.open(str(d), os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    except OSError:
        pass
    print(f"[INFO] Barrier: wrote {marker}")


def _marker_ready(marker: Path) -> bool:
    """True if marker is visible and readable (helps with NFS attribute cache)."""
    try:
        with open(marker, "r") as f:
            f.read()
        return True
    except OSError:
        return False


def wait_all_train_exit_ok(
    workdir: str, step_num: int, num_nodes: int, timeout_sec: int = 7200, poll_sec: float = 3.0
) -> bool:
    d = barrier_dir(workdir)
    print(
        f"[INFO] Barrier: waiting for all {num_nodes} nodes to finish training step {step_num} "
        f"(timeout={timeout_sec}s)"
    )
    start = time.time()
    last_log = 0.0
    while time.time() - start < timeout_sec:
        try:
            list(d.iterdir())
        except OSError:
            pass
        missing = []
        for i in range(num_nodes):
            marker = d / f"step{step_num}_node{i}_train_exit_ok"
            if not _marker_ready(marker):
                missing.append(i)
        if not missing:
            print(f"[INFO] Barrier: all nodes reported train exit for step {step_num}")
            return True
        now = time.time()
        if now - last_log >= 30.0:
            print(f"[INFO] Barrier: still waiting for node marker(s): {missing}")
            last_log = now
        time.sleep(poll_sec)
    print(f"[ERR] Barrier timeout: still missing node markers for ranks {missing}")
    return False
