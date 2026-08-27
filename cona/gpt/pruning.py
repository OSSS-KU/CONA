#!/usr/bin/env python3
"""
Universal checkpoint pruning module
"""

import os
import shutil
from pathlib import Path
from typing import List, Literal


def prune_universal_checkpoints(
    ckpt_dir: str,
    mode: Literal["keep-latest", "delete-all"]
) -> bool:
    """
    Prune universal checkpoint folders

    Args:
        ckpt_dir: Checkpoint directory path
        mode: "keep-latest" to keep only the latest, "delete-all" to delete all

    Returns:
        True if successful, False otherwise
    """
    ckpt_path = Path(ckpt_dir)

    if not ckpt_path.exists():
        print(f"[ERR] CKPT_DIR not found: {ckpt_dir}")
        return False

    latest_universal_file = ckpt_path / "latest_universal"
    latest_name = None

    if latest_universal_file.exists():
        with open(latest_universal_file, 'r') as f:
            latest_name = f.read().strip().rstrip('\r\n')

    print(f"[INFO] Pruning universal checkpoints under: {ckpt_dir}")
    print(f"[INFO] Mode={mode} latest_universal={latest_name or '<missing>'}")

    universal_dirs = list(ckpt_path.glob("global_step*_universal"))

    if not universal_dirs:
        print("[OK] No universal checkpoint dirs found.")
        return True

    if mode == "keep-latest":
        if not latest_name:
            print("[ERR] latest_universal missing or empty, refusing to keep-latest safely.")
            print("[HINT] Use delete-all or ensure latest_universal exists.")
            return False

        keep_path = ckpt_path / latest_name
        if not keep_path.exists():
            print(f"[ERR] latest_universal points to missing dir: {keep_path}")
            return False

        deleted = 0
        for d in universal_dirs:
            if d == keep_path:
                continue
            print(f"[INFO] rm -rf {d}")
            shutil.rmtree(d)
            deleted += 1

        print(f"[OK] Deleted {deleted} universal dirs; kept: {keep_path}")
        return True

    elif mode == "delete-all":
        deleted = 0
        for d in universal_dirs:
            print(f"[INFO] rm -rf {d}")
            shutil.rmtree(d)
            deleted += 1

        if latest_universal_file.exists():
            latest_universal_file.unlink()

        print(f"[OK] Deleted {deleted} universal dirs; removed latest_universal pointer.")
        return True

    else:
        print(f"[ERR] Unknown mode: {mode} (expected keep-latest|delete-all)")
        return False
