#!/usr/bin/env bash

set -euo pipefail

REPO_PATH="${REPO_PATH:-/workspace/Megatron-DeepSpeed}"
DATA_PATH="${DATA_PATH:-/workspace/datasets/gpt/wikitext_gpt_text_document}"
VOCAB="${VOCAB:-/workspace/datasets/gpt/vocab.json}"
MERGES="${MERGES:-/workspace/datasets/gpt/merges.txt}"
TOKENIZER="${TOKENIZER:-GPT2BPETokenizer}"
WANDB_DIR="${WANDB_DIR:-/workspace/wandb}"
LOG_ROOT="${LOG_ROOT:-/workspace/logs}"
PROJECT="${PROJECT:-cona-fixed-strategy}"

# Model shape (defaults are a GPT-2 medium proxy).
NUM_LAYERS="${NUM_LAYERS:-24}"
HIDDEN="${HIDDEN:-1024}"
HEADS="${HEADS:-16}"
SEQ="${SEQ:-1024}"

MBS="${MBS:-2}"
ITERS="${ITERS:-1000}"
LR="${LR:-1.5e-4}"
MIN_LR="${MIN_LR:-1e-5}"
ZERO_STAGE="${ZERO_STAGE:-1}"
COOLDOWN_SEC="${COOLDOWN_SEC:-60}"

# "name:dp:tp:pp" — every entry must satisfy dp * tp * pp = available GPUs.
CONFIGS="${CONFIGS:-DP4_TP1_PP1:4:1:1 DP2_TP2_PP1:2:2:1 DP2_TP1_PP2:2:1:2 DP1_TP4_PP1:1:4:1 DP1_TP2_PP2:1:2:2 DP1_TP1_PP4:1:1:4}"
GBS_LIST="${GBS_LIST:-8 16 32 64 128}"

mkdir -p "${WANDB_DIR}" "${LOG_ROOT}"

run_one () {
  local name="$1" dp="$2" tp="$3" pp="$4" gbs="$5"
  local ngpu=$(( dp * tp * pp ))
  local tag="${name}_g${gbs}_mb${MBS}_it${ITERS}"
  local log_dir="${LOG_ROOT}/fixed_${tag}"
  local ds_config="${log_dir}/ds_config.json"
  mkdir -p "${log_dir}"

  # DeepSpeed config is derived from the strategy, matching run_training.py.
  python3 - "$ds_config" "$gbs" "$MBS" "$(( gbs / (MBS * dp) ))" "$ZERO_STAGE" <<'PY'
import json, sys
path, gbs, mbs, accum, zero = sys.argv[1], *map(int, sys.argv[2:6])
json.dump({
    "train_batch_size": gbs,
    "train_micro_batch_size_per_gpu": mbs,
    "gradient_accumulation_steps": accum,
    "steps_per_print": 1,
    "zero_optimization": {"stage": zero},
    "fp16": {"enabled": True, "loss_scale": 0, "initial_scale_power": 12},
    "wall_clock_breakdown": False,
}, open(path, "w"), indent=2)
PY

  echo "==== [${name}] gbs=${gbs} mbs=${MBS} iters=${ITERS} (ngpu=${ngpu}) ===="
  ( cd "${REPO_PATH}" && deepspeed --num_gpus="${ngpu}" pretrain_gpt.py \
      --tensor-model-parallel-size "${tp}" \
      --pipeline-model-parallel-size "${pp}" \
      --num-layers "${NUM_LAYERS}" \
      --hidden-size "${HIDDEN}" \
      --num-attention-heads "${HEADS}" \
      --seq-length "${SEQ}" \
      --max-position-embeddings "${SEQ}" \
      --micro-batch-size "${MBS}" \
      --global-batch-size "${gbs}" \
      --train-iters "${ITERS}" \
      --data-path "${DATA_PATH}" \
      --vocab-file "${VOCAB}" \
      --merge-file "${MERGES}" \
      --tokenizer-type "${TOKENIZER}" \
      --lr "${LR}" \
      --min-lr "${MIN_LR}" \
      --lr-decay-style cosine \
      --lr-warmup-fraction 0.01 \
      --weight-decay 0.01 \
      --log-interval 1 \
      --eval-interval 100 \
      --eval-iters 100 \
      --deepspeed \
      --deepspeed_config "${ds_config}" \
      --fp16 \
      --wandb-project "${PROJECT}" \
      --wandb-exp-name "${tag}" \
      --wandb-save-dir "${WANDB_DIR}" ) 2>&1 | tee "${log_dir}/train.log"

  echo ">>> ${tag} done; cooling down ${COOLDOWN_SEC}s"
  sleep "${COOLDOWN_SEC}"
}

for gbs in ${GBS_LIST}; do
  for cfg in ${CONFIGS}; do
    IFS=':' read -r name dp tp pp <<< "${cfg}"
    if (( gbs % dp != 0 || (gbs / dp) % MBS != 0 )); then
      echo "[SKIP] ${name} gbs=${gbs}: gbs must be divisible by dp and mbs*dp"
      continue
    fi
    run_one "${name}" "${dp}" "${tp}" "${pp}" "${gbs}"
  done
done

echo "==== All runs complete ===="
