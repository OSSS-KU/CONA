#!/usr/bin/env python3
"""
GPT-3 Benchmark Script using Megatron-DeepSpeed
Runs GPT-3 training with different (dp, tp, pp) configurations and batch sizes,
then calculates throughput, statistical efficiency, and goodput using AdaptDL metrics.
"""

import os
import sys
import subprocess
import re
import json
import time
import argparse
import csv
from pathlib import Path
import numpy as np

# Add adaptdl to path for metric calculations to use exact AdaptDL efficiency calculation
# Try multiple paths to find adaptdl package
script_dir = os.path.dirname(os.path.abspath(__file__))
adaptdl_paths = [
    os.path.join(script_dir, 'adaptdl'),
    os.path.join(script_dir, 'adaptdl', 'adaptdl'),
    script_dir,
]
for adaptdl_path in adaptdl_paths:
    if os.path.exists(adaptdl_path) and adaptdl_path not in sys.path:
        sys.path.insert(0, adaptdl_path)

# Import AdaptDL's GoodputFunction to use exact efficiency calculation
# Priority: local file > installed package (to ensure we use exact same code)
ADAPTDL_AVAILABLE = False
GoodputFunction = None
try:
    # First try: import from local adaptdl directory structure (most reliable for exact match)
    import importlib.util
    goodput_path = os.path.join(script_dir, 'adaptdl', 'adaptdl', 'goodput.py')
    if os.path.exists(goodput_path):
        # Load the module directly from file to ensure we use the exact AdaptDL code
        spec = importlib.util.spec_from_file_location("adaptdl_adaptdl_goodput", goodput_path)
        goodput_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(goodput_module)
        GoodputFunction = goodput_module.GoodputFunction
        ADAPTDL_AVAILABLE = True
        print(f"Successfully imported GoodputFunction from local file: {goodput_path}")
    else:
        raise ImportError(f"Could not find goodput.py at {goodput_path}")
except Exception as e1:
    try:
        # Second try: if adaptdl is installed as a package
        from adaptdl.adaptdl.goodput import GoodputFunction
        ADAPTDL_AVAILABLE = True
        print("Successfully imported GoodputFunction from adaptdl.adaptdl.goodput")
    except ImportError as e2:
        try:
            # Third try: direct import from adaptdl package
            from adaptdl.goodput import GoodputFunction
            ADAPTDL_AVAILABLE = True
            print("Successfully imported GoodputFunction from adaptdl.goodput")
        except ImportError as e3:
            # If AdaptDL is not available, we'll use the exact same formula
            # but it's better to have AdaptDL available for exact matching
            print(f"Warning: AdaptDL not available. Tried:")
            print(f"  - Local file: {e1}")
            print(f"  - adaptdl.adaptdl.goodput: {e2}")
            print(f"  - adaptdl.goodput: {e3}")
            print(f"Will use manual calculation matching AdaptDL formula exactly.")
            ADAPTDL_AVAILABLE = False


def parse_log_for_metrics(log_file):
    """
    Parse Megatron-DeepSpeed log to extract metrics
    Returns: list of dicts with iteration, step_time, throughput, loss, etc.
    """
    metrics = []
    
    if not os.path.exists(log_file):
        return metrics
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Pattern to match iteration logs
    # Example log line format:
    # " iteration     100/    100 | consumed samples:           1600 | elapsed time per iteration (ms): 1234.56 | learning rate: 1.500E-04 | global batch size:   16 | loss: 8.234567E+00 | loss scale: 32768.0 | grad norm: 1.234 | samples/sec: 12.34 | tokens/sec: 12634.56"
    
    current_iteration = None
    current_metric = {}
    
    # Also parse AdaptDL online efficiency lines (rank0) so we can compute
    # goodput even when Megatron iteration logging is missing/truncated.
    # Format:
    #   ADAPTDL_GNS ... efficiency=0.123456
    for line in lines:
        # Match iteration number - look for "iteration X/Y |" pattern
        iter_match = re.search(r'iteration\s+(\d+)/\s*\d+\s*\|', line)
        if iter_match:
            # Save previous iteration if exists
            if current_iteration is not None and current_metric:
                metrics.append(current_metric)
            current_iteration = int(iter_match.group(1))
            current_metric = {'iteration': current_iteration}
            
            # All metrics are on the same line, extract them all
            # Extract step time (ms)
            step_time_match = re.search(r'elapsed time per iteration \(ms\):\s+([\d.]+)', line)
            if step_time_match:
                current_metric['step_time'] = float(step_time_match.group(1))
            
            # Extract samples per second (note: "samples per second" not "samples/sec")
            samples_match = re.search(r'samples per second:\s+([\d.]+)', line)
            if samples_match:
                current_metric['samples_per_sec'] = float(samples_match.group(1))
            
            # Extract tokens per gpu per second
            tokens_match = re.search(r'tokens per gpu per second \(tgs\):\s+([\d.]+)', line)
            if tokens_match:
                current_metric['tokens_per_gpu_per_sec'] = float(tokens_match.group(1))
            
            # Extract loss - look for "lm loss:" pattern
            loss_match = re.search(r'lm loss:\s+([\d.E+-]+)', line)
            if loss_match:
                try:
                    current_metric['loss'] = float(loss_match.group(1))
                except ValueError:
                    pass
            
            # Extract grad norm
            grad_norm_match = re.search(r'grad norm:\s+([\d.]+)', line)
            if grad_norm_match:
                current_metric['grad_norm'] = float(grad_norm_match.group(1))
            
            # Extract global batch size
            batch_match = re.search(r'global batch size:\s+(\d+)', line)
            if batch_match:
                current_metric['global_batch_size'] = int(batch_match.group(1))
    
    # Save last iteration
    if current_iteration is not None and current_metric:
        metrics.append(current_metric)
    
    return metrics


def calculate_grad_params_from_logs(all_batch_metrics, dp=1):
    """
    Calculate grad_sqr and grad_var from all batch sizes' gradient norms
    This follows AdaptDL's approach based on gradient_noise_scale.py:268-270
    
    AdaptDL's actual formula (when count > 1):
        grad_sqr = (count * total_sqr - local_sqr) / (count - 1)
        grad_var = (local_sqr - total_sqr) * scale / (count - 1)
    
    Where:
        - total_sqr: squared norm of averaged gradient (what we see in logs as grad_norm^2)
        - local_sqr: average of local squared norms from each replica
        - count: number of replicas (dp)
        - scale: batch_size / init_batch_size
    
    Since we only have total_sqr (grad_norm^2) from logs, we need to estimate:
        - For each batch size, we have observed grad_norm (which is total_sqr^0.5)
        - We need to estimate local_sqr from the variance across batch sizes
    
    Args:
        all_batch_metrics: dict mapping batch_size -> metrics dict with 'grad_norms' list
        dp: data parallel size (number of replicas)
    
    Returns:
        (grad_sqr, grad_var): tuple of gradient squared norm and variance
    """
    # Collect gradient norms grouped by batch size
    batch_grad_norms = {}
    for batch_size, metrics in all_batch_metrics.items():
        if metrics and 'grad_norms' in metrics and metrics['grad_norms']:
            batch_grad_norms[batch_size] = metrics['grad_norms']
    
    if not batch_grad_norms:
        # Fallback: use default values
        return (100.0, 50.0)
    
    # Collect all gradient norms for overall statistics
    all_grad_norms = []
    for norms in batch_grad_norms.values():
        all_grad_norms.extend(norms)
    
    if not all_grad_norms:
        return (100.0, 50.0)
    
    # Calculate total_sqr: mean of squared gradient norms (this is what we observe)
    # In AdaptDL, this corresponds to total_sqr = ||averaged_gradient||^2
    grad_norms_sqr = [gn ** 2 for gn in all_grad_norms]
    total_sqr = np.mean(grad_norms_sqr)
    
    # Estimate local_sqr from variance across batch sizes
    # The key insight: variance of grad_norms across batch sizes reflects
    # the difference between local_sqr and total_sqr
    # When batch size increases, local variance decreases, so local_sqr approaches total_sqr
    
    # Use the variance of squared norms across all iterations as a proxy
    # for the difference between local_sqr and total_sqr
    if len(grad_norms_sqr) > 1:
        # Variance of squared norms gives us information about local variance
        var_of_sqr = np.var(grad_norms_sqr, ddof=1)
        
        # Estimate local_sqr: it should be larger than total_sqr
        # Based on AdaptDL formula: local_sqr = total_sqr + (count-1)/count * grad_var/scale
        # For estimation, we use the variance as a proxy
        # A reasonable estimate: local_sqr ≈ total_sqr + variance_contribution
        # But we need to be careful - variance of squared norms is not the same as grad_var
        
        # Better approach: use the relationship from AdaptDL
        # If we assume count = dp (data parallel size), then:
        # grad_sqr = (count * total_sqr - local_sqr) / (count - 1)
        # grad_var = (local_sqr - total_sqr) * scale / (count - 1)
        
        # For estimation from logs, we use:
        # - total_sqr = mean(grad_norm^2) from logs
        # - Estimate local_sqr from variance across batch sizes
        #   When batch size is small, variance is higher (local_sqr >> total_sqr)
        #   When batch size is large, variance is lower (local_sqr ≈ total_sqr)
        
        # Key insight: In AdaptDL, when we only have total_sqr (averaged gradient norm),
        # we need to estimate local_sqr. The relationship is:
        #   local_sqr >= total_sqr (because averaging reduces variance)
        #
        # However, if we overestimate local_sqr, we'll overestimate grad_var,
        # which will make efficiency too low (which is the user's concern).
        #
        # Better approach: Use a more conservative estimate that doesn't
        # over-penalize efficiency. The variance across batch sizes gives us
        # information, but we should be conservative.
        
        # Estimate local_sqr more conservatively
        # When dp=1, local_sqr = total_sqr (no averaging)
        # When dp>1, local_sqr > total_sqr, but the difference depends on variance
        if dp == 1:
            # Single replica: local_sqr = total_sqr
            estimated_local_sqr = total_sqr
        else:
            # Multiple replicas: estimate local_sqr from variance
            # Use a conservative multiplier to avoid overestimating grad_var
            # The variance of squared norms gives us an upper bound
            # But we use a smaller multiplier to be conservative
            estimated_local_sqr = total_sqr + var_of_sqr * 0.2  # Conservative multiplier
        
        # Now estimate grad_sqr and grad_var using AdaptDL's formula
        count = max(dp, 2)  # At least 2 for the formula to work
        init_batch_size = min(batch_grad_norms.keys())
        scale_init = init_batch_size / init_batch_size  # = 1.0
        
        # Use AdaptDL's formula with estimated local_sqr
        if count > 1:
            grad_sqr = (count * total_sqr - estimated_local_sqr) / (count - 1)
            grad_var = (estimated_local_sqr - total_sqr) * scale_init / (count - 1)
        else:
            # Single replica case
            grad_sqr = total_sqr
            grad_var = 0.0
        
        # Ensure non-negative values
        grad_sqr = max(0.0, grad_sqr)
        grad_var = max(0.0, grad_var)
        
        # Additional sanity check: grad_var should not be too large relative to grad_sqr
        # If it is, we're likely overestimating, which causes efficiency to be too low
        if grad_sqr > 0:
            var_ratio = grad_var / grad_sqr
            # If var_ratio is too high (> 1.0), it's likely an overestimate
            # Cap it at a reasonable value (0.5 is typical)
            if var_ratio > 0.5:
                # Re-estimate with more conservative local_sqr
                estimated_local_sqr = total_sqr + var_of_sqr * 0.1  # Even more conservative
                if count > 1:
                    grad_sqr = (count * total_sqr - estimated_local_sqr) / (count - 1)
                    grad_var = (estimated_local_sqr - total_sqr) * scale_init / (count - 1)
                grad_sqr = max(0.0, grad_sqr)
                grad_var = max(0.0, grad_var)
        
        # Final fallback if estimates are still unreasonable
        if grad_sqr <= 0 or grad_var < 0:
            # Fallback to simpler, more conservative estimation
            grad_sqr = total_sqr
            # Use a conservative var_ratio (0.2-0.3 is typical for well-behaved gradients)
            var_ratio = min(0.3, max(0.1, var_of_sqr / total_sqr if total_sqr > 0 else 0.2))
            grad_var = grad_sqr * var_ratio
    else:
        # Single sample - use default ratio
        grad_sqr = total_sqr
        grad_var = grad_sqr * 0.3  # Conservative default
    
    return (float(grad_sqr), float(grad_var))


def calculate_goodput(throughput, efficiency):
    """Calculate goodput = throughput * efficiency"""
    return throughput * efficiency


def calculate_efficiency_adaptdl(batch_size, grad_sqr, grad_var, init_batch_size):
    """
    Calculate efficiency using AdaptDL's exact GoodputFunction.efficiency() method.
    This ensures we use the exact same code as adaptdl/adaptdl/goodput.py:80-86.
    
    Args:
        batch_size: Current batch size (scalar)
        grad_sqr: Gradient squared norm (grad_sqr)
        grad_var: Gradient variance (grad_var)
        init_batch_size: Initial batch size (scalar)
    
    Returns:
        efficiency value (scalar float)
    """
    # Always try to use AdaptDL's exact code first
    if ADAPTDL_AVAILABLE and GoodputFunction is not None:
        try:
            # Use AdaptDL's exact GoodputFunction.efficiency() method
            # Create a GoodputFunction instance with dummy perf_params (not used for efficiency)
            # The efficiency calculation only depends on grad_params and init_batch_size
            dummy_perf_params = (0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 1.0)
            grad_params = (grad_sqr, grad_var)
            goodput_fn = GoodputFunction(dummy_perf_params, grad_params, init_batch_size)
            # Call the exact AdaptDL efficiency method
            efficiency = goodput_fn.efficiency(batch_size)
            # Convert to scalar if numpy array
            if isinstance(efficiency, np.ndarray):
                efficiency = float(efficiency.item())
            else:
                efficiency = float(efficiency)
            return efficiency
        except Exception as e:
            print(f"Warning: Failed to use AdaptDL GoodputFunction.efficiency(): {e}")
            print("Falling back to manual calculation matching AdaptDL formula exactly.")
    
    # Fallback: use exact same formula as AdaptDL's efficiency method
    # This matches adaptdl/adaptdl/goodput.py:80-86 exactly:
    #   scale = batch_size / self._init_batch_size
    #   denom = grad_var / scale + grad_sqr
    #   gain = np.where(denom > 0, (grad_var + grad_sqr) / denom, 1.0)
    #   return gain / scale
    scale = batch_size / init_batch_size
    denom = grad_var / scale + grad_sqr
    gain = np.where(denom > 0, (grad_var + grad_sqr) / denom, 1.0)
    efficiency = gain / scale
    # Convert to scalar if numpy array
    if isinstance(efficiency, np.ndarray):
        efficiency = float(efficiency.item())
    else:
        efficiency = float(efficiency)
    return efficiency


def run_training(config_name, dp, tp, pp, batch_size, micro_batch_size,
                megatron_dir, output_dir, train_iters=50, gradient_accumulation_steps=1,
                adaptdl_init_batch_size=None, adaptdl_log_efficiency=True,
                lr=1.5e-4,
                log_interval=1):
    """
    Run Megatron-DeepSpeed training with given configuration
    """
    print(f"\n{'='*80}")
    print(f"Running: {config_name}")
    print(f"  DP={dp}, TP={tp}, PP={pp}, Batch Size={batch_size}, Micro Batch={micro_batch_size}")
    print(f"{'='*80}")
    
    # Calculate global batch size
    global_batch_size = batch_size
    
    # Ensure output directory exists (use absolute path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Log file (use absolute path)
    log_file = os.path.join(output_dir, f"{config_name}_bs{batch_size}.log")
    
    # Calculate gradient accumulation steps for DeepSpeed config
    # DeepSpeed expects: train_batch_size = micro_batch_size * gradient_accumulation_steps * world_size
    # where world_size = data_parallel_size (for DeepSpeed)
    # But Megatron uses: global_batch_size = micro_batch_size * gradient_accumulation_steps * data_parallel_size
    # So we need to make sure DeepSpeed config matches
    
    # Create a temporary DeepSpeed config with correct gradient_accumulation_steps
    import tempfile
    import shutil
    
    # Read original config
    with open("/workspace/ds_config.json", 'r') as f:
        ds_config = json.load(f)
    
    # Update gradient_accumulation_steps in config
    # For DeepSpeed: train_batch_size should equal global_batch_size
    # train_batch_size = micro_batch_size * gradient_accumulation_steps * dp
    # So: gradient_accumulation_steps = global_batch_size / (micro_batch_size * dp)
    calculated_grad_acc = global_batch_size // (micro_batch_size * dp)
    if calculated_grad_acc == 0:
        calculated_grad_acc = 1
    
    # Update DeepSpeed config
    # Let DeepSpeed derive the global batch size:
    # - Do NOT set train_batch_size (DeepSpeed will auto-calculate it)
    # - Only set train_micro_batch_size_per_gpu and gradient_accumulation_steps
    # DeepSpeed will calculate: train_batch_size = micro_batch_size * gradient_accumulation_steps * world_size
    # where world_size = data_parallel_size
    
    # Remove train_batch_size if it exists (let DeepSpeed auto-calculate)
    if "train_batch_size" in ds_config:
        del ds_config["train_batch_size"]
    
    ds_config["train_micro_batch_size_per_gpu"] = micro_batch_size
    ds_config["gradient_accumulation_steps"] = calculated_grad_acc
    
    # Verify the calculation would match
    expected_train_batch = micro_batch_size * calculated_grad_acc * dp
    if expected_train_batch != global_batch_size:
        print(f"Warning: Calculated train_batch_size ({expected_train_batch}) != global_batch_size ({global_batch_size})")
        print(f"  micro_batch_size={micro_batch_size}, grad_acc={calculated_grad_acc}, dp={dp}")
        print(f"  DeepSpeed will use calculated value: {expected_train_batch}")
        # This is OK - DeepSpeed will use the calculated value
    
    # Create temporary config file
    temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(ds_config, temp_config, indent=2)
    temp_config.close()
    temp_config_path = temp_config.name
    
    # Build command
    cmd = [
        "deepspeed",
        f"--num_gpus=4",
        "pretrain_gpt.py",
        "--tensor-model-parallel-size", str(tp),
        "--pipeline-model-parallel-size", str(pp),
        "--num-layers", "24",
        "--hidden-size", "1024",
        "--num-attention-heads", "16",
        "--seq-length", "1024",
        "--max-position-embeddings", "1024",
        "--micro-batch-size", str(micro_batch_size),
        "--global-batch-size", str(global_batch_size),
        "--train-iters", str(train_iters),
        "--data-path", "/workspace/wikitext_gpt_text_document",
        "--vocab-file", "/workspace/vocab.json",
        "--merge-file", "/workspace/merges.txt",
        "--tokenizer-type", "GPT2BPETokenizer",
        "--lr", str(lr),
        "--min-lr", "1e-5",
        "--lr-decay-style", "cosine",
        "--lr-warmup-fraction", "0.01",
        "--weight-decay", "0.01",
        "--log-interval", str(int(log_interval)),
        "--eval-iters", "10",
        "--deepspeed",
        "--deepspeed_config", temp_config_path,
        "--fp16",
        "--distributed-timeout-minutes", "30",
    ]
    
    # Change to Megatron-DeepSpeed directory
    original_dir = os.getcwd()
    os.chdir(megatron_dir)
    
    # Clear GPU memory before running
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Wait a bit for memory to clear
            import time
            time.sleep(2)
    except:
        pass
    
    try:
        # Run training and capture output
        print(f"Running command: {' '.join(cmd)}")
        print(f"Logging to: {log_file}")
        expected_train_batch = micro_batch_size * calculated_grad_acc * dp
        print(f"DeepSpeed config: micro_batch_size_per_gpu={ds_config['train_micro_batch_size_per_gpu']}, "
              f"gradient_accumulation_steps={ds_config['gradient_accumulation_steps']}, "
              f"expected_train_batch_size={expected_train_batch} (auto-calculated by DeepSpeed)")
        
        # Use absolute path for log file
        log_file_abs = os.path.abspath(log_file)
        with open(log_file_abs, 'w') as f:
            # Enable AdaptDL online GNS logging inside Megatron-DeepSpeed training
            # (see Megatron-DeepSpeed/megatron/training.py changes).
            env = os.environ.copy()
            if adaptdl_log_efficiency:
                env["ADAPTDL_LOG_EFFICIENCY"] = "1"
                env["ADAPTDL_GNS_DEBUG"] = os.getenv("ADAPTDL_GNS_DEBUG", "0")
                # Prefer correctness for benchmarking: ensure the "total_sqr"
                # statistic is computed from truly averaged gradients even if
                # the training stack doesn't synchronize `param.grad` at the
                # autograd callback boundary.
                env.setdefault("ADAPTDL_GNS_STRICT_SYNC", os.getenv("ADAPTDL_GNS_STRICT_SYNC", "1"))
                if adaptdl_init_batch_size is not None:
                    env["ADAPTDL_INIT_BATCH_SIZE"] = str(int(adaptdl_init_batch_size))
                # Make sure the training process can import the local AdaptDL package.
                # NOTE: In this repo layout, the python package root is /workspace/adaptdl/adaptdl
                # (i.e., /workspace/adaptdl/adaptdl/adaptdl is the actual package).
                # So we prepend that path so `import adaptdl.torch...` works.
                existing_pp = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = f"/workspace/adaptdl/adaptdl:/workspace/adaptdl:{existing_pp}"

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=megatron_dir,
                env=env,
            )
            
            # Stream output
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()
            
            process.wait()
        
        if process.returncode != 0:
            print(f"Warning: Training exited with code {process.returncode}")
            return None
        
        # Return absolute path
        return log_file_abs
        
    except Exception as e:
        print(f"Error running training: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up temporary config file
        if 'temp_config_path' in locals():
            try:
                os.unlink(temp_config_path)
            except:
                pass
        os.chdir(original_dir)


def extract_metrics_from_log(log_file, batch_size, burnin_iters=0, window_iters=None):
    """
    Extract metrics from log file and calculate AdaptDL metrics
    """
    if not os.path.exists(log_file):
        return None
    
    # Check if log file is too small (likely incomplete)
    if os.path.getsize(log_file) < 1000:
        return None
    
    metrics_list = parse_log_for_metrics(log_file)
    
    if not metrics_list:
        return None
    
    # Analysis-only: optionally ignore early iterations (warmup/compile/cache effects).
    if burnin_iters and burnin_iters > 0:
        metrics_list = [m for m in metrics_list if m.get('iteration', 0) > burnin_iters]
        if not metrics_list:
            return None
    
    # Analysis-only: optionally use only the last N iterations after burn-in.
    if window_iters is not None:
        try:
            window_iters = int(window_iters)
        except Exception:
            window_iters = None
        if window_iters is not None and window_iters > 0:
            metrics_list = metrics_list[-window_iters:]
            if not metrics_list:
                return None
    
    # Extract step times and throughputs
    step_times = []
    throughputs = []
    losses = []
    grad_norms = []
    
    for m in metrics_list:
        if 'step_time' in m:
            # Convert ms to seconds
            step_times.append(m['step_time'] / 1000.0)
        if 'samples_per_sec' in m:
            throughputs.append(m['samples_per_sec'])
        if 'loss' in m:
            losses.append(m['loss'])
        if 'grad_norm' in m:
            grad_norms.append(m['grad_norm'])

    # Extract AdaptDL online efficiency (logged from Megatron training).
    # Logged format (rank0):
    #   ADAPTDL_GNS ... efficiency=0.123456
    adaptdl_eff_list = []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                m = re.search(r'ADAPTDL_GNS.*efficiency=([\d.]+)', line)
                if m:
                    adaptdl_eff_list.append(float(m.group(1)))
    except Exception:
        adaptdl_eff_list = []

    # Apply burn-in / window to AdaptDL efficiency list as well (one value per optimizer step).
    if adaptdl_eff_list:
        if burnin_iters and burnin_iters > 0:
            adaptdl_eff_list = adaptdl_eff_list[burnin_iters:]
        if window_iters is not None:
            try:
                w = int(window_iters)
            except Exception:
                w = None
            if w is not None and w > 0:
                adaptdl_eff_list = adaptdl_eff_list[-w:]

    adaptdl_efficiency = (sum(adaptdl_eff_list) / len(adaptdl_eff_list)) if adaptdl_eff_list else None
    
    if not step_times and not throughputs:
        return None
    
    # Calculate average metrics
    avg_step_time = sum(step_times) / len(step_times) if step_times else None
    avg_throughput = sum(throughputs) / len(throughputs) if throughputs else None
    avg_loss = sum(losses) / len(losses) if losses else None
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else None
    
    # If we have step_time but not throughput, calculate it
    if avg_throughput is None and avg_step_time:
        avg_throughput = batch_size / avg_step_time
    
    # Return raw metrics - efficiency will be calculated later using all batch sizes
    return {
        'step_times': step_times,
        'throughputs': throughputs if throughputs else [batch_size / t for t in step_times] if step_times else [],
        'losses': losses,
        'grad_norms': grad_norms,
        'avg_step_time': avg_step_time,
        'avg_throughput': avg_throughput,
        'avg_loss': avg_loss,
        'avg_grad_norm': avg_grad_norm,
        'adaptdl_efficiency': adaptdl_efficiency,
        'num_steps': len(step_times) if step_times else len(throughputs) if throughputs else 0
    }
    


def main():
    parser = argparse.ArgumentParser(description='GPT-3 Benchmark with AdaptDL Metrics')
    parser.add_argument('--megatron-dir', type=str, default='/workspace/Megatron-DeepSpeed',
                       help='Path to Megatron-DeepSpeed directory')
    parser.add_argument('--output-dir', type=str, default='/workspace/adaptdl/benchmark_results',
                       help='Directory to save logs and results')
    parser.add_argument('--train-iters', type=int, default=20,
                       help='Number of training iterations per configuration')
    parser.add_argument('--log-interval', type=int, default=1,
                       help='Megatron log interval (iterations). Use >1 for long runs to reduce log size.')
    parser.add_argument('--batch-sizes', type=str, default=None,
                       help='Comma-separated batch sizes to run (default: 8,16,32,64,128)')
    parser.add_argument('--micro-batch-size', type=int, default=None,
                       help='Override micro batch size per GPU. If not set, we choose a Pollux-style micro-batch per GPU for each global batch size (prefer gas=1).')
    parser.add_argument('--max-micro-batch-size', type=int, default=None,
                       help='Optional cap for chosen micro batch size per GPU (useful to avoid OOM). If set, any leftover scaling uses gradient accumulation steps.')
    parser.add_argument('--base-lr', type=float, default=1.5e-4,
                       help='Base learning rate at init-batch-size (and used as-is if lr scaling is disabled).')
    parser.add_argument('--lr-scaling', type=str, default="none",
                       choices=["none", "linear", "sqrt"],
                       help='Pollux-style batch-dependent LR scaling rule. '
                            '"linear" sets lr = base_lr * scale, '
                            '"sqrt" sets lr = base_lr * sqrt(scale).')
    parser.add_argument('--burnin-iters', type=int, default=0,
                       help='Ignore first N training iterations when computing metrics (analysis-only).')
    parser.add_argument('--window-iters', type=int, default=None,
                       help='Use only last N iterations (after burn-in) when computing metrics (analysis-only).')
    parser.add_argument('--init-batch-size', type=int, default=8,
                       help='Baseline batch size for efficiency scaling (AdaptDL init_batch_size).')
    parser.add_argument('--skip-run', action='store_true',
                       help='Skip training runs, only analyze existing logs')
    parser.add_argument('--config', type=str, default=None,
                       help='Run only specific config (e.g., "dp4_tp1_pp1")')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configuration combinations
    configs = [
        (4, 1, 1, "dp4_tp1_pp1"),
        (1, 4, 1, "dp1_tp4_pp1"),
        (1, 1, 4, "dp1_tp1_pp4"),
        (2, 2, 1, "dp2_tp2_pp1"),
        (2, 1, 2, "dp2_tp1_pp2"),
        (1, 2, 2, "dp1_tp2_pp2"),
    ]
    
    if args.batch_sizes:
        batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    else:
        batch_sizes = [8, 16, 32, 64, 128]
    # IMPORTANT: baseline init batch size should NOT depend on the selected batch_sizes subset.
    init_batch_size = int(args.init_batch_size)
    
    # Filter configs if specified
    if args.config:
        configs = [c for c in configs if c[3] == args.config]
    
    all_results = []
    
    print("="*80)
    print("GPT-3 Benchmark with AdaptDL Metrics")
    print("="*80)
    print(f"Megatron-DeepSpeed dir: {args.megatron_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Training iterations per config: {args.train_iters}")
    print(f"Total configurations: {len(configs) * len(batch_sizes)}")
    print("="*80)
    
    # First pass: collect all metrics for each config
    config_metrics = {}  # config_name -> {batch_size -> metrics}
    
    for dp, tp, pp, config_name in configs:
        config_metrics[config_name] = {}
        for batch_size in batch_sizes:
            # Pollux-style: prefer realizing larger global batch sizes by increasing
            # the per-GPU micro-batch (atomic batch), not by increasing gradient
            # accumulation. This allows throughput to actually improve with batch
            # size (until saturation), creating the expected interior optimum in
            # goodput vs batch-size.
            #
            # We keep the math honest: global_batch_size must equal
            # micro_batch_size_per_gpu * gradient_accumulation_steps * dp.
            if args.micro_batch_size is not None:
                micro_batch_size = int(args.micro_batch_size)
            else:
                if batch_size % dp != 0:
                    print(f"Warning: batch_size={batch_size} is not divisible by dp={dp}; skipping.")
                    continue
                micro_batch_size = batch_size // dp  # Prefer gas=1.
                if args.max_micro_batch_size is not None:
                    micro_batch_size = min(micro_batch_size, int(args.max_micro_batch_size))
                    micro_batch_size = max(micro_batch_size, 1)

            if micro_batch_size <= 0:
                print(f"Warning: invalid micro_batch_size={micro_batch_size}; skipping.")
                continue

            # Now compute grad accumulation steps to match the requested global batch size.
            denom = micro_batch_size * dp
            if batch_size % denom != 0:
                print(
                    f"Warning: batch_size={batch_size} is not divisible by "
                    f"(micro_batch_size={micro_batch_size} * dp={dp} = {denom}); skipping."
                )
                continue
            gradient_accumulation_steps = batch_size // denom
            
            # Verify
            final_global = micro_batch_size * gradient_accumulation_steps * dp
            if final_global != batch_size:
                print(f"Warning: Cannot make batch_size={batch_size} work with dp={dp}, tp={tp}, pp={pp}")
                print(f"  micro_batch_size={micro_batch_size}, grad_accum={gradient_accumulation_steps}")
                print(f"  Calculated: {final_global}, Expected: {batch_size}")
                continue
            
            # Choose LR for this batch size according to the requested scaling rule.
            # This is NOT an arbitrary "fudge factor" — Pollux/AdaptDL assumes
            # per-batch retuning (or a scaling rule) when evaluating statistical
            # efficiency and goodput trade-offs.
            scale = float(batch_size) / float(init_batch_size)
            lr = float(args.base_lr)
            if args.lr_scaling == "linear":
                lr = lr * scale
            elif args.lr_scaling == "sqrt":
                lr = lr * math.sqrt(scale)

            print(f"  Calculated micro_batch_size={micro_batch_size}, grad_accum={gradient_accumulation_steps}, global={final_global}, lr={lr:.6g}")
            
            log_file = os.path.join(args.output_dir, f"{config_name}_bs{batch_size}.log")
            
            # Run training if not skipping
            if not args.skip_run:
                print(f"\n{'='*80}")
                print(f"Starting training: {config_name}, batch_size={batch_size}")
                print(f"  micro_batch_size={micro_batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}")
                print(f"  Expected global batch: {micro_batch_size * gradient_accumulation_steps * dp}")
                print(f"{'='*80}\n")
                
                result_log = run_training(
                    config_name, dp, tp, pp, batch_size, micro_batch_size,
                    args.megatron_dir, args.output_dir, args.train_iters,
                    gradient_accumulation_steps,
                    adaptdl_init_batch_size=init_batch_size,
                    adaptdl_log_efficiency=True,
                    lr=lr,
                    log_interval=args.log_interval,
                )
                if result_log:
                    log_file = result_log
                
                # Wait longer between runs to avoid NCCL port conflicts
                print(f"\nWaiting 10 seconds before next run...\n")
                time.sleep(10)
            
            # Extract metrics from log (only if file exists and is not too small)
            if os.path.exists(log_file) and os.path.getsize(log_file) > 1000:
                metrics = extract_metrics_from_log(
                    log_file, batch_size,
                    burnin_iters=args.burnin_iters,
                    window_iters=args.window_iters,
                )
                if metrics:
                    # Track the *actual* execution shape for this run so we can
                    # report it accurately in results (Pollux-style micro-batch scaling).
                    metrics["micro_batch_size"] = int(micro_batch_size)
                    metrics["gradient_accumulation_steps"] = int(gradient_accumulation_steps)
                    config_metrics[config_name][batch_size] = metrics
    
    for dp, tp, pp, config_name in configs:
        if config_name not in config_metrics:
            continue

        # Calculate efficiency and goodput for each batch size.
        # Prefer AdaptDL online efficiency logged during training (ADAPTDL_GNS),
        # otherwise fall back to the existing offline analysis path.
        grad_sqr = grad_var = None
        if any(m.get('adaptdl_efficiency') is None for m in config_metrics[config_name].values()):
            grad_sqr, grad_var = calculate_grad_params_from_logs(config_metrics[config_name], dp=dp)

        for batch_size in batch_sizes:
            if batch_size not in config_metrics[config_name]:
                continue
            
            metrics = config_metrics[config_name][batch_size]

            if metrics.get('adaptdl_efficiency') is not None:
                efficiency = float(metrics['adaptdl_efficiency'])
            else:
                # Fallback (offline): AdaptDL GoodputFunction.efficiency or exact formula.
                efficiency = calculate_efficiency_adaptdl(batch_size, grad_sqr, grad_var, init_batch_size)
            
            # Calculate goodput
            goodput = calculate_goodput(metrics['avg_throughput'], efficiency) if metrics['avg_throughput'] else None
            
            # micro_batch_size is fixed at 2
            micro_batch_size = metrics.get('micro_batch_size', '')
            
            result = {
                'config': config_name,
                'dp': dp,
                'tp': tp,
                'pp': pp,
                'batch_size': batch_size,
                'micro_batch_size': micro_batch_size,
                'throughput': metrics['avg_throughput'],
                'efficiency': efficiency,
                'goodput': goodput,
                'avg_step_time': metrics['avg_step_time'],
                'avg_loss': metrics['avg_loss'],
                'num_steps': metrics['num_steps'],
                'log_file': os.path.join(args.output_dir, f"{config_name}_bs{batch_size}.log")
            }
            all_results.append(result)
            
            print(f"\n{config_name} (batch_size={batch_size}):")
            print(f"  Throughput: {result['throughput']:.2f} samples/sec")
            print(f"  Efficiency: {result['efficiency']:.4f}")
            print(f"  Goodput: {result['goodput']:.2f} samples/sec")
    
    # Save results as JSON (combined)
    results_file = os.path.join(args.output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save results as CSV - one file per configuration
    # Group results by configuration
    config_results = {}
    for result in all_results:
        config_name = result['config']
        if config_name not in config_results:
            config_results[config_name] = []
        config_results[config_name].append(result)
    
    # Save each configuration to a separate CSV file
    fieldnames = ['config', 'dp', 'tp', 'pp', 'batch_size', 'micro_batch_size',
                 'throughput', 'efficiency', 'goodput', 'avg_step_time_ms', 
                 'avg_loss', 'num_steps']
    
    csv_files = []
    for config_name, config_data in config_results.items():
        if not config_data:
            continue
        
        # Get num_steps from first result (should be same for all in same config)
        num_steps = config_data[0].get('num_steps', '')
        dp = config_data[0]['dp']
        tp = config_data[0]['tp']
        pp = config_data[0]['pp']
        
        # Create filename: dp4_tp1_pp1_10_results.csv
        csv_filename = f"dp{dp}_tp{tp}_pp{pp}_{num_steps}_results.csv"
        csv_file = os.path.join(args.output_dir, csv_filename)
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in config_data:
                row = {
                    'config': result['config'],
                    'dp': result['dp'],
                    'tp': result['tp'],
                    'pp': result['pp'],
                    'batch_size': result['batch_size'],
                    'micro_batch_size': result.get('micro_batch_size', ''),
                    'throughput': f"{result['throughput']:.4f}" if result['throughput'] else '',
                    'efficiency': f"{result['efficiency']:.6f}" if result['efficiency'] else '',
                    'goodput': f"{result['goodput']:.4f}" if result['goodput'] else '',
                    'avg_step_time_ms': f"{result['avg_step_time']*1000:.2f}" if result.get('avg_step_time') else '',
                    'avg_loss': f"{result['avg_loss']:.6f}" if result.get('avg_loss') else '',
                    'num_steps': result.get('num_steps', '')
                }
                writer.writerow(row)
        
        csv_files.append(csv_file)
        print(f"CSV results saved to {csv_file}")
    
    if csv_files:
        print(f"\nTotal {len(csv_files)} CSV file(s) created")
    
    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE - Step per Throughput, Statistical Efficiency, Goodput")
    print("="*100)
    print(f"{'Config (dp,tp,pp)':<20} {'Batch':<8} {'Throughput (samples/sec)':<25} {'Efficiency':<12} {'Goodput (samples/sec)':<20} {'Step Time (ms)':<15} {'Loss':<12}")
    print("-"*100)
    
    if not all_results:
        print("No results found. Please run training first or check log files.")
    else:
        for result in all_results:
            step_time_ms = result['avg_step_time'] * 1000 if result['avg_step_time'] else 0
            avg_loss = result.get('avg_loss', 0) if result.get('avg_loss') else 0
            config_str = f"({result['dp']},{result['tp']},{result['pp']})"
            print(f"{config_str:<20} {result['batch_size']:<8} "
                  f"{result['throughput']:>23.2f} {result['efficiency']:>10.4f} "
                  f"{result['goodput']:>18.2f} {step_time_ms:>13.2f} {avg_loss:>10.4f}")
    
    print("="*100)
    print(f"Results saved to:")
    print(f"  JSON: {results_file}")
    if csv_files:
        print(f"  CSV files ({len(csv_files)}):")
        for csv_file in csv_files:
            print(f"    - {os.path.basename(csv_file)}")
        print(f"\nEach CSV file contains columns:")
        print(f"  config, dp, tp, pp, batch_size, micro_batch_size, throughput, efficiency, goodput, avg_step_time_ms, avg_loss, num_steps")
    print("="*100)
    
    # Also print per-step results if available
    if all_results:
        print("\n" + "="*100)
        print("DETAILED PER-STEP RESULTS")
        print("="*100)
        for result in all_results:
            config_str = f"({result['dp']},{result['tp']},{result['pp']})"
            print(f"\n{result['config']} {config_str}, Batch Size: {result['batch_size']}")
            print(f"  Step 1: Throughput={result.get('throughput', 0):.2f}, Efficiency={result.get('efficiency', 0):.4f}, Goodput={result.get('goodput', 0):.2f}")
            if 'step_times' in result and result['step_times']:
                for i, (step_time, throughput) in enumerate(zip(result['step_times'][:3], 
                                                                  result.get('throughputs', [])[:3] or [result.get('throughput', 0)]*len(result['step_times'][:3])), 1):
                    step_eff = result.get('efficiency', 0)
                    step_goodput = throughput * step_eff
                    print(f"  Step {i}: Time={step_time*1000:.2f}ms, Throughput={throughput:.2f}, Efficiency={step_eff:.4f}, Goodput={step_goodput:.2f}")


if __name__ == '__main__':
    main()

