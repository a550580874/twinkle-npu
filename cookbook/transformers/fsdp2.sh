MODEL_REF="/data/weight/Qwen3-Coder-Next/"
TEMPLATE_MODEL_ID="/data/weight/Qwen3-Coder-Next/"
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
IP=
export http_proxy="http://p_atlas:proxy%40123@$IP:8080"
export https_proxy="http://p_atlas:proxy%40123@$IP:8080"
export no_proxy=127.0.0.1,.huawei.com,localhost,local,.local

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 torchrun --nproc_per_node=8 fsdp2.py --model-ref "${MODEL_REF}" --template-model-id "${TEMPLATE_MODEL_ID}"
