import argparse

from peft import LoraConfig
from tqdm import tqdm

import twinkle
from twinkle import DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

# Construct a device_mesh, fsdp=4, dp=2
device_mesh = DeviceMesh.from_sizes(fsdp_size=4, dp_size=2)
# use torchrun mode
twinkle.initialize(mode='local', global_device_mesh=device_mesh)

logger = get_logger()
DEFAULT_TEMPLATE_MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
DEFAULT_MODEL_REF = DEFAULT_TEMPLATE_MODEL_ID


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--template-model-id',
        default=DEFAULT_TEMPLATE_MODEL_ID,
        help='Model id or local path used by dataset/template initialization.',
    )
    parser.add_argument(
        '--model-ref',
        default=DEFAULT_MODEL_REF,
        help='Model reference used by TransformersModel. Supports a local directory or a remote model id.',
    )
    return parser.parse_args()


def eval(model, template_model_id):
    # 100 Samples
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(100)))
    dataset.set_template('Template', model_id=template_model_id)
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    dataset.encode()
    dataloader = DataLoader(dataset=dataset, batch_size=4)
    for step, batch in tqdm(enumerate(dataloader)):
        model.forward_only(inputs=batch)
        model.calculate_loss()
    metrics = model.calculate_metric(is_training=False)
    return metrics


def train(template_model_id, model_ref):
    # 1000 samples
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(1000)))
    # Set template to prepare encoding
    dataset.set_template('Template', model_id=template_model_id)
    # Preprocess the dataset to standard format
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    # Encode dataset
    dataset.encode()
    # Global batch size = 4, for GPUs, so 1 sample per GPU
    dataloader = DataLoader(dataset=dataset, batch_size=8)
    # Qwen3-Next uses Qwen3NextSparseMoeBlock for MoE layers.
    model = TransformersModel(
        model_id=model_ref,
        fsdp_config={'transformer_cls_names_to_wrap': ['Qwen3NextSparseMoeBlock']},
    )
    # Patch MoE model to fix the hang bug, support transformers==4.*
    # transformer5.x执行会报错，4.x才需要这个补丁
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
        # Do forward and backward
        model.forward_backward(inputs=batch)
        # Step
        model.clip_grad_and_step()
        if step % 20 == 0:
            # Print metric
            metric = model.calculate_metric(is_training=True)
            logger.info(f'Current is step {step} of {len(dataloader)}, metric: {metric}')
        if step > 0 and step % 40 == 0:
           metrics = eval(model, template_model_id)
           logger.info(f'Eval metric: {metrics}')
           metrics['step'] = step
           if loss_metric > float(metrics['loss']):
               model.save(f'checkpoint-{step}')
               loss_metric = float(metrics['loss'])
    model.save(f'last-checkpoint')


if __name__ == '__main__':
    args = parse_args()
    train(args.template_model_id, args.model_ref)
