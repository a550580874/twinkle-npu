MODEL_REF="${1:-/data/weight/Qwen3-Coder-Next/}"
TEMPLATE_MODEL_ID="${2:-${MODEL_REF}}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 fsdp2_moe.py --model-ref "${MODEL_REF}" --template-model-id "${TEMPLATE_MODEL_ID}"
