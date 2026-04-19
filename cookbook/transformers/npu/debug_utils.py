#固定随机
import os
import random
import time
import numpy as np
import torch
import torch.distributed as dist


def seed_all(seed=1234, mode=True, is_gpu=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['GLOBAL_SEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(mode)
    if is_gpu:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enable = False
        torch.backends.cudnn.benchmark = False
    else:
        import torch_npu
        os.environ['HCCL_DETERMINISTIC'] = 'true'
        os.environ['CLOSE_MATMUL_K_SHIFT'] = '1'
        torch_npu.npu.manual_seed_all(seed)
        torch_npu.npu.manual_seed(seed)
    print("====== seed all ========")


def get_time():
    torch.npu.synchronize()
    return time.time()

def get_rank():
    return dist.get_rank()