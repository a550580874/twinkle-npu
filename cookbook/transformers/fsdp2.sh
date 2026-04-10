MODEL_PATH="${1:-ms://Qwen/Qwen3.5-4B}"

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 torchrun --nproc_per_node=8 fsdp2.py --model-path "${MODEL_PATH}"
