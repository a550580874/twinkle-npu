import argparse
import os
from peft import LoraConfig
from tqdm import tqdm

import twinkle
from twinkle import DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor
from debug_utils import seed_all, get_time, get_rank
from monkey_patch import apply_hf_moe_grouped_mm_patch
from loader import _install_prefetch_patch



logger = get_logger()



def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {'true', '1', 'yes', 'y', 'on'}:
        return True
    if value in {'false', '0', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-id',
        default='ms://swift/self-cognition',
        help='Dataset id used by DatasetMeta.',
    )
    parser.add_argument(
        '--model-id',
        default='ms://Qwen/Qwen3-30B-A3B-Instruct-2507',
        help='Model id used by template initialization and model loading.',
    )
    parser.add_argument(
        '--seed-all',
        type=str2bool,
        default=False,
        help='Whether to call debug_utils.seed_all before training starts.',
    )
    parser.add_argument('--fsdp-size', type=int, default=4, help='FSDP mesh size.')
    parser.add_argument('--dp-size', type=int, default=2, help='DP mesh size.')
    parser.add_argument('--npu-gemm', type=str2bool, default=True, help='gemm_npu')
    parser.add_argument('--prefetch', type=str2bool, default=True, help='Whether to enable prefetch patch.')
    parser.add_argument('--prefetch-forward-layers', type=int, default=1, help='Forward prefetch layers.')
    parser.add_argument('--prefetch-backward-layers', type=int, default=1, help='Backward prefetch layers.')
    parser.add_argument(
        '--prefetch-block',
        default='model.model.model.layers',
        help='Attribute path used to locate the block passed into _install_prefetch_patch.',
    )
    return parser.parse_args()


args = parse_args()
world_size = int(os.environ.get('WORLD_SIZE', '1'))
assert world_size == args.fsdp_size * args.dp_size, (
    f'WORLD_SIZE({world_size}) must equal fsdp_size({args.fsdp_size}) * dp_size({args.dp_size})'
)
# Construct a device_mesh, fsdp/dp from args
device_mesh = DeviceMesh.from_sizes(fsdp_size=args.fsdp_size, dp_size=args.dp_size)
# use torchrun mode
twinkle.initialize(mode='local', global_device_mesh=device_mesh)

if args.npu_gemm:
    apply_hf_moe_grouped_mm_patch()

def eval(model):
    # 100 Samples
    dataset = Dataset(dataset_meta=DatasetMeta(args.dataset_id, data_slice=range(100)))
    dataset.set_template('Template', model_id=args.model_id)
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    dataset.encode()
    dataloader = DataLoader(dataset=dataset, batch_size=4)
    for step, batch in tqdm(enumerate(dataloader)):
        model.forward_only(inputs=batch)
        model.calculate_loss()
    metrics = model.calculate_metric(is_training=False)
    return metrics


def train():
    if args.seed_all:
        seed_all(is_gpu=False)
    ##patch预取
    _install_prefetch_patch(block=args.prefetch_block, args=args)
    # 1000 samples
    dataset = Dataset(dataset_meta=DatasetMeta(args.dataset_id, data_slice=range(1000)))
    # Set template to prepare encoding
    dataset.set_template('Template', model_id=args.model_id)
    # Preprocess the dataset to standard format
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    # Encode dataset
    dataset.encode()
    # Global batch size = 4, for GPUs, so 1 sample per GPU
    dataloader = DataLoader(dataset=dataset, batch_size=8)
    if args.prefetch:
        model = TransformersModel(model_id=args.model_id, fsdp_config={'transformer_cls_names_to_wrap':['Qwen3MoeSparseMoeBlock'], 'forward_prefetch':True})
    else:        
        # Use a TransformersModel, transformer_cls_names_to_wrap=Qwen3MoeSparseMoeBlock to avoid hang of fsdp2
        model = TransformersModel(model_id=args.model_id, fsdp_config={'transformer_cls_names_to_wrap':['Qwen3MoeSparseMoeBlock']})

    # Patch MoE model to fix the hang bug, support transformers==4.*
    # model.apply_patch('ms://twinkle-kit/qwen3_moe_transformers4_patch')
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules='all-linear'
    )

    # Add a lora to model, with name `default`
    # Comment this to use full-parameter training
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)
    # Add Optimizer for lora `default`
    model.set_optimizer(optimizer_cls='AdamW', lr=1e-4)
    # Add LRScheduler for lora `default`
    model.set_lr_scheduler(scheduler_cls='CosineWarmupScheduler', num_warmup_steps=5, num_training_steps=len(dataloader))
    logger.info(get_device_placement())
    # Print the training config
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')
    loss_metric = 99.0
    # lora: 34G * 8
    for step, batch in enumerate(dataloader):
        start_time = get_time()
        # Do forward and backward
        model.forward_backward(inputs=batch)
        # Step
        model.clip_grad_and_step()
        # loss&grad metric
        metric = model.calculate_metric(is_training=True)
        if step > 0 and step % 40 == 0:
           metrics = eval(model)
           logger.info(f'Eval metric: {metrics}')
           metrics['step'] = step
           if loss_metric > float(metrics['loss']):
               model.save(f'checkpoint-{step}')
               loss_metric = float(metrics['loss'])
        end_time = get_time()
        if get_rank() == 0:
            logger.info(f'Current is step {step} of {len(dataloader)}, metric: {metric}')
            print(f"{get_rank()}-rank-step-time: {end_time - start_time}")
    model.save(f'last-checkpoint')


if __name__ == '__main__':
    train()
