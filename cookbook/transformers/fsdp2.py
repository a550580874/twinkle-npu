import argparse

from peft import LoraConfig
from tqdm import tqdm

import twinkle
from twinkle import DeviceMesh, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.preprocessor import SelfCognitionProcessor

# Construct a device_mesh, fsdp_size=2, dp=4
device_mesh = DeviceMesh.from_sizes(fsdp_size=2, dp_size=4)
# use torchrun mode
twinkle.initialize(mode='local', global_device_mesh=device_mesh)

logger = get_logger()
DEFAULT_TEMPLATE_MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
DEFAULT_MODEL_REF = DEFAULT_TEMPLATE_MODEL_ID

import warnings
import urllib3
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# 1) patch HubApi session
from modelscope.hub.api import HubApi
_old_hubapi_init = HubApi.__init__

def _patched_hubapi_init(self, *args, **kwargs):
    _old_hubapi_init(self, *args, **kwargs)
    self.session.verify = False

HubApi.__init__ = _patched_hubapi_init

# 2) patch msdatasets hf_file_utils requests path
import modelscope.msdatasets.utils.hf_file_utils as hf_file_utils

_old_request_with_retry_ms = hf_file_utils._request_with_retry_ms

def _patched_request_with_retry_ms(method, url, max_retries=2,
                                   base_wait_time=0.5, max_wait_time=2,
                                   timeout=10.0, **params):
    params.setdefault("verify", False)
    return _old_request_with_retry_ms(
        method=method,
        url=url,
        max_retries=max_retries,
        base_wait_time=base_wait_time,
        max_wait_time=max_wait_time,
        timeout=timeout,
        **params
    )

hf_file_utils._request_with_retry_ms = _patched_request_with_retry_ms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--template-model-id',
        default=DEFAULT_TEMPLATE_MODEL_ID,
        help='Model id used by dataset/template initialization. Must stay a valid hub model id.',
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
    dataset.set_template('Qwen3_5Template', model_id=template_model_id)
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    dataset.encode()
    dataloader = DataLoader(dataset=dataset, batch_size=8)
    for step, batch in tqdm(enumerate(dataloader)):
        model.forward_only(inputs=batch)
        model.calculate_loss()
    metrics = model.calculate_metric(is_training=False)
    return metrics


def train(template_model_id, model_ref):
    # 1000 samples
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(1000)))
    # Set template to prepare encoding
    dataset.set_template('Qwen3_5Template', model_id=template_model_id)
    # Preprocess the dataset to standard format
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    # Encode dataset
    dataset.encode()
    # Global batch size = 8, for GPUs, so 1 sample per GPU
    dataloader = DataLoader(dataset=dataset, batch_size=8)
    # Use a TransformersModel
    model = TransformersModel(model_id=model_ref)
    model.model._no_split_modules = {'Qwen3_5DecoderLayer'}

    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules='all-linear')

    # Add a lora to model, with name `default`
    # Comment this to use full-parameter training
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)
    # Add Optimizer for lora `default`
    model.set_optimizer(optimizer_cls='AdamW', lr=1e-4)
    # Add LRScheduler for lora `default`
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler', num_warmup_steps=5, num_training_steps=len(dataloader))
    logger.info(get_device_placement())
    # Print the training config
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')
    loss_metric = 99.0
    # lora: 8G * 8
    # full: 18G * 8
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
