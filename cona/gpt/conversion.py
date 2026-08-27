#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

from config import ConfigManager


class ConversionExecutor:
    """Executes checkpoint conversion steps"""

    def __init__(
        self,
        config: ConfigManager,
        workdir: str = "/workspace/cona"
    ):
        self.config = config
        self.workdir = workdir

    def convert_to_universal(
        self,
        step_num: int,
        dp: int,
        tp: int,
        pp: int,
        use_relaxed: bool = False
    ) -> bool:

        print(f"[INFO] Converting the step {step_num} checkpoint to Universal")

        ckpt_dir = self.config.get_checkpoint_dir(dp, tp, pp, self.workdir)
        ckpt_path = Path(ckpt_dir)

        latest_iter_file = ckpt_path / "latest_checkpointed_iteration.txt"
        if not latest_iter_file.exists():
            print(f"[ERR] latest_checkpointed_iteration.txt not found in: {ckpt_dir}")
            print(f"[HINT] Run previous step first.")
            return False

        with open(latest_iter_file, 'r') as f:
            iter_num = int(f.read().strip())

        in_dir = ckpt_path / f"global_step{iter_num}"
        out_dir = ckpt_path / f"global_step{iter_num}_universal"

        if not in_dir.exists():
            print(f"[ERR] input checkpoint folder not found: {in_dir}")
            return False

        print(f"[INFO] Converting:")
        print(f"  IN : {in_dir}")
        print(f"  OUT: {out_dir}")

        cmd = [
            "python3", "-m", "deepspeed.checkpoint.ds_to_universal",
            "--input_folder", str(in_dir),
            "--output_folder", str(out_dir),
        ]

        if use_relaxed:
            cmd.extend([
                "--num_merge_workers", "1",
                "--no_strict",
                "--inject_missing_state",
                "--keep_temp_folder"
            ])

        start_time = datetime.now()
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            end_time = datetime.now()
            elapsed_sec = (end_time - start_time).total_seconds()

            print(f"[TIME] step{step_num} ds_to_universal elapsed_sec={elapsed_sec}")

            timings_file = Path(self.config.training.log_root) / "convert_timings.tsv"
            timings_file.parent.mkdir(parents=True, exist_ok=True)

            with open(timings_file, 'a') as f:
                f.write(
                    f"{datetime.now().isoformat()}\t"
                    f"step{step_num}\t"
                    f"iter={iter_num}\t"
                    f"elapsed_sec={elapsed_sec}\t"
                    f"in={in_dir}\t"
                    f"out={out_dir}\n"
                )

            latest_universal_file = ckpt_path / "latest_universal"
            with open(latest_universal_file, 'w') as f:
                f.write(f"global_step{iter_num}_universal\n")
                f.flush()
                os.fsync(f.fileno())
            try:
                fd = os.open(str(ckpt_path), os.O_RDONLY)
                os.fsync(fd)
                os.close(fd)
            except OSError:
                pass

            print(f"[OK] Step {step_num} conversion done. CKPT_DIR contents:")
            subprocess.run(["ls", "-la", ckpt_dir], check=False)
            return True

        except subprocess.CalledProcessError as e:
            print(f"[ERR] Conversion failed with exit code {e.returncode}")
            print(f"[ERR] stderr: {e.stderr}")
            return False
