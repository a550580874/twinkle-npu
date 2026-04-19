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
#prefetch layers set only support native_fsdp
PREFETCH_FORWARD_LAYERS=1
PREFETCH_BACKWARD_LAYERS=1
PREFETCH=True
PREFETCH_BLOCK=model.model.model.layers

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
  --prefetch-block "${PREFETCH_BLOCK}" \
