#!/usr/bin/env bash

# CANN loading
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# export CPU_AFFINITY_CONF=2
# export HCCL_BUFFSIZE=400
# export HCCL_CONNECT_TIMEOUT=1600

SEED_ALL=False
FSDP_SIZE=4
DP_SIZE=2
DATASET_ID=ms://swift/self-cognition
MODEL_ID=ms://Qwen/Qwen3-30B-A3B-Instruct-2507
NPU_GEMM=True
# prefetch layers set only support native_fsdp
PREFETCH_FORWARD_LAYERS=1
PREFETCH_BACKWARD_LAYERS=1
PREFETCH=True
PREFETCH_BLOCK=model.model.model.layers
LOG_DIR=/path/to/logs

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
AI_CORE_RAW_FILE="${LOG_DIR}/ai_core_${TIMESTAMP}_raw.csv"
AI_CORE_FILE="${LOG_DIR}/ai_core_${TIMESTAMP}.csv"

sample_ai_core() {
  local train_pid="$1"
  local raw_file="$2"
  local elapsed=0

  printf 'elapsed_seconds,rank_id,ai_core_percent\n' > "${raw_file}"

  while kill -0 "${train_pid}" 2>/dev/null; do
    npu-smi info | awk -F '|' -v elapsed="${elapsed}" '
      /Process id/ {t=0}
      BEGIN {t=1}
      t && $2 ~ /^[ \t]*[0-9]+[ \t]+[A-Za-z0-9]+[ \t]*$/ {
        split($2, a)
        rank_id=a[1]
        getline
        split($4, b)
        ai_core=b[1]
        gsub(/^[ \t]+|[ \t]+$/, "", rank_id)
        gsub(/^[ \t]+|[ \t]+$/, "", ai_core)
        gsub(/%/, "", ai_core)
        print elapsed "," rank_id "," ai_core
      }
    ' >> "${raw_file}"
    sleep 1
    elapsed=$((elapsed + 1))
  done
}

trim_ai_core_csv() {
  local raw_file="$1"
  local final_file="$2"

  awk -F ',' '
    NR == 1 {
      header=$0
      next
    }
    {
      rows[NR]=$0
      row_sec[NR]=$1 + 0
      if (!seen[$1]++) {
        secs[++sec_count]=$1 + 0
      }
      if (($3 + 0) != 0) {
        active[$1 + 0]=1
      }
    }
    END {
      print header > out
      for (i = 1; i <= sec_count; i++) {
        sec=secs[i]
        if (active[sec]) {
          if (first == "") {
            first=sec
          }
          last=sec
        }
      }
      if (first == "") {
        exit
      }
      for (i = 2; i <= NR; i++) {
        sec=row_sec[i]
        if (sec >= first && sec <= last) {
          print rows[i] > out
        }
      }
    }
  ' out="${final_file}" "${raw_file}"
}

(
  echo "[launch] timestamp=${TIMESTAMP}"
  echo "[launch] log_file=${LOG_FILE}"
  echo "[launch] ai_core_file=${AI_CORE_FILE}"

  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 fsdp2_moe.py \
    --dataset-id "${DATASET_ID}" \
    --model-id "${MODEL_ID}" \
    --fsdp-size "${FSDP_SIZE}" \
    --dp-size "${DP_SIZE}" \
    --seed-all "${SEED_ALL}" \
    --npu-gemm "${NPU_GEMM}" \
    --prefetch-forward-layers "${PREFETCH_FORWARD_LAYERS}" \
    --prefetch-backward-layers "${PREFETCH_BACKWARD_LAYERS}" \
    --prefetch "${PREFETCH}" \
    --prefetch-block "${PREFETCH_BLOCK}" &
  TRAIN_PID=$!

  sample_ai_core "${TRAIN_PID}" "${AI_CORE_RAW_FILE}" &
  AI_CORE_PID=$!

  wait "${TRAIN_PID}"
  TRAIN_EXIT_CODE=$?

  wait "${AI_CORE_PID}" || true
  trim_ai_core_csv "${AI_CORE_RAW_FILE}" "${AI_CORE_FILE}"
  rm -f "${AI_CORE_RAW_FILE}"

  echo "[done] train_exit_code=${TRAIN_EXIT_CODE}"
  echo "[done] log_file=${LOG_FILE}"
  echo "[done] ai_core_file=${AI_CORE_FILE}"

  exit "${TRAIN_EXIT_CODE}"
) > "${LOG_FILE}" 2>&1 < /dev/null &

BACKEND_PID=$!

echo "backend_pid=${BACKEND_PID}"
echo "log_file=${LOG_FILE}"
echo "ai_core_file=${AI_CORE_FILE}"
