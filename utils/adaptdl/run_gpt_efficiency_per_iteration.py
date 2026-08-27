#!/usr/bin/env python3
"""
GPT Efficiency Calculator - Per Iteration
Runs GPT training and calculates efficiency for all batch sizes at each iteration.
Baseline batch size is 1 (as per user requirement).
"""

import os
import sys
import subprocess
import re
import json
import time
import argparse
import csv
import math
from pathlib import Path
import numpy as np
from collections import deque

# Add adaptdl to path
script_dir = os.path.dirname(os.path.abspath(__file__))
adaptdl_paths = [
    os.path.join(script_dir, 'adaptdl'),
    os.path.join(script_dir, 'adaptdl', 'adaptdl'),
    script_dir,
]
for adaptdl_path in adaptdl_paths:
    if os.path.exists(adaptdl_path) and adaptdl_path not in sys.path:
        sys.path.insert(0, adaptdl_path)


def run_gpt_training(batch_size, micro_batch_size, megatron_dir, output_dir, 
                     train_iters=1000, seq_length=1024, init_batch_size=1, 
                     target_batch_sizes=[8,16,32,64,128,256],
                     dp_launch=4, tp=1, pp=1):
    """Run GPT training with specified batch size."""
    print(f"\n{'='*80}")
    print(f"Running GPT training: batch_size={batch_size}, micro_batch_size={micro_batch_size}")
    print(f"{'='*80}")
    
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f"gpt_bs{batch_size}.log")
    
    # Build DeepSpeed config on the fly (no external file dependency).
    import tempfile
    
    calculated_grad_acc = batch_size // max(1, micro_batch_size * dp_launch)
    if calculated_grad_acc == 0:
        calculated_grad_acc = 1
    
    ds_config = {
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": calculated_grad_acc,
        "zero_optimization": {"stage": 1},
        "fp16": {"enabled": True, "loss_scale": 0, "initial_scale_power": 12},
        "wall_clock_breakdown": False
    }

    temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(ds_config, temp_config, indent=2)
    temp_config.close()
    temp_config_path = temp_config.name
    
    # GPT data paths (updated to cona layout)
    data_path = "/workspace/cona/Megatron-DeepSpeed/wikitext_gpt_text_document_text_document"
    vocab_file = "/workspace/cona/Megatron-DeepSpeed/vocab.json"
    merge_file = "/workspace/cona/Megatron-DeepSpeed/merges.txt"
    
    # Build command
    cmd = [
        "deepspeed",
        f"--num_gpus={dp_launch}",
        "pretrain_gpt.py",
        "--tensor-model-parallel-size", str(tp),
        "--pipeline-model-parallel-size", str(pp),
        "--num-layers", "24",
        "--hidden-size", "1024",
        "--num-attention-heads", "16",
        "--seq-length", str(seq_length),
        "--max-position-embeddings", str(seq_length),
        "--micro-batch-size", str(micro_batch_size),
        "--global-batch-size", str(batch_size),
        "--train-iters", str(train_iters),
        "--data-path", data_path,
        "--vocab-file", vocab_file,
        "--merge-file", merge_file,
        "--tokenizer-type", "GPT2BPETokenizer",
        "--lr", "1.5e-4",
        "--min-lr", "1e-5",
        "--lr-decay-style", "cosine",
        "--lr-warmup-fraction", "0.01",
        "--weight-decay", "0.01",
        "--log-interval", "1",  # Log every iteration
        "--eval-iters", "10",
        "--deepspeed",
        "--deepspeed_config", temp_config_path,
        "--fp16",
        "--distributed-timeout-minutes", "30",
        "--no-pipeline-parallel",
    ]
    
    original_dir = os.getcwd()
    os.chdir(megatron_dir)
    
    try:
        # Clear GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                time.sleep(2)
        except:
            pass
        
        print(f"Running command: {' '.join(cmd)}")
        print(f"Logging to: {log_file}")
        
        log_file_abs = os.path.abspath(log_file)
        
        # Set CSV output file path
        csv_output_path = os.path.join(output_dir, f'dp{dp_launch}_tp{tp}_pp{pp}_efficiency.csv')
        
        with open(log_file_abs, 'w') as f:
            env = os.environ.copy()
            env["MEGATRON_DISABLE_FUSED_KERNELS"] = "1"
            env["NO_CKPT"] = "1"
            
            # Enable AdaptDL GNS logging and CSV generation (like BERT benchmark)
            env["ADAPTDL_LOG_EFFICIENCY"] = "1"
            env["ADAPTDL_GNS_DEBUG"] = os.getenv("ADAPTDL_GNS_DEBUG", "0")
            env.setdefault("ADAPTDL_GNS_STRICT_SYNC", os.getenv("ADAPTDL_GNS_STRICT_SYNC", "1"))
            env["ADAPTDL_INIT_BATCH_SIZE"] = str(int(init_batch_size))
            env["ADAPTDL_EFFICIENCY_BATCH_SIZES"] = ",".join(map(str, target_batch_sizes))
            env["ADAPTDL_EFFICIENCY_CSV_FILE"] = csv_output_path
            
            # Make sure AdaptDL can be imported
            existing_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"/workspace/cona/adaptdl/adaptdl:/workspace/cona/adaptdl:{existing_pp}"
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=megatron_dir,
                env=env,
            )
            
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()
            
            process.wait()
        
        if process.returncode != 0:
            print(f"Warning: Training exited with code {process.returncode}")
            return None
        
        return log_file_abs
        
    except Exception as e:
        print(f"Error running training: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if 'temp_config_path' in locals():
            try:
                os.unlink(temp_config_path)
            except:
                pass
        os.chdir(original_dir)


def main():
    parser = argparse.ArgumentParser(description='GPT Efficiency Calculator (Per Iteration)')
    parser.add_argument('--megatron-dir', type=str, default='/workspace/cona/Megatron-DeepSpeed',
                       help='Path to Megatron-DeepSpeed directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save logs and results (default: auto under /workspace/cona/adaptdl/results)')
    parser.add_argument('--train-iters', type=int, default=1000,
                       help='Number of training iterations')
    parser.add_argument('--run-batch-size', type=int, default=8,
                       help='Batch size to run training with (default: 8)')
    parser.add_argument('--target-batch-sizes', type=str, default='8,16,32,64,128,256',
                       help='Comma-separated batch sizes to calculate efficiency for')
    parser.add_argument('--init-batch-size', type=int, default=4,
                       help='Baseline batch size for efficiency calculation (default: 4)')
    parser.add_argument('--seq-length', type=int, default=1024,
                       help='Sequence length')
    args = parser.parse_args()
    
    init_batch_size = args.init_batch_size  # Baseline = 1
    run_batch_size = args.run_batch_size
    target_batch_sizes = [int(x.strip()) for x in args.target_batch_sizes.split(",")]
    dp = 4 if run_batch_size >= 4 else 1  # DP auto-adjust for tiny batches
    tp = 1
    pp = 1
    micro_batch_size = max(1, run_batch_size // dp)

    base_results = Path("/workspace/cona/adaptdl/results")
    base_results.mkdir(parents=True, exist_ok=True)

    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir_name = f"gpt_dp{dp}_tp{tp}_pp{pp}_gbs{run_batch_size}_mbs{micro_batch_size}"
        run_dir = base_results / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("GPT Efficiency Calculator (Per Iteration, Pollux/AdaptDL Method)")
    print("="*80)
    print(f"Running training with batch_size={run_batch_size}")
    print(f"Baseline batch size (init_batch_size)={init_batch_size}")
    print(f"Will calculate efficiency for batch sizes: {target_batch_sizes}")
    print(f"Output dir: {run_dir}")
    print("="*80)
    
    # Run training
    log_file = run_gpt_training(
        run_batch_size, micro_batch_size,
        args.megatron_dir, str(run_dir),
        args.train_iters, args.seq_length,
        init_batch_size=init_batch_size,
        target_batch_sizes=target_batch_sizes,
        dp_launch=dp,
        tp=tp,
        pp=pp
    )
    
    # Check if CSV was generated by Megatron-DeepSpeed (AdaptDL GNS)
    csv_file = run_dir / f'dp{dp}_tp{tp}_pp{pp}_efficiency.csv'
    
    if csv_file.exists():
        print(f"\n✓ CSV file generated by AdaptDL GNS: {csv_file}")
        print(f"File size: {csv_file.stat().st_size} bytes")
        
        # Print first few rows as preview
        print("\nFirst 10 rows preview:")
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:11]):  # Header + 10 rows
                print(line.strip())
        
        print(f"\nTotal rows: {len(lines)}")
        print("="*80)
    else:
        print(f"\n✗ ERROR: CSV file not found at {csv_file}")
        print("AdaptDL GNS must generate the CSV directly from actual gradient statistics.")
        print("Please check:")
        print("  1. ADAPTDL_LOG_EFFICIENCY=1 is set")
        print("  2. ADAPTDL_INIT_BATCH_SIZE is set correctly")
        print("  3. ADAPTDL_EFFICIENCY_CSV_FILE is set correctly")
        print("  4. AdaptDL package is importable (PYTHONPATH includes /workspace/cona/adaptdl/adaptdl)")
        print("  5. Training completed successfully")
        sys.exit(1)


if __name__ == '__main__':
    main()

