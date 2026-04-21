#!/usr/bin/env bash

# CANN loading
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# export CPU_AFFINITY_CONF=2
# export HCCL_BUFFSIZE=400
# export HCCL_CONNECT_TIMEOUT=1600

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 fsdp2_moe.py
