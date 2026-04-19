from twinkle.model import TransformersModel
import functools
import warnings
from types import SimpleNamespace


def set_modules_to_forward_prefetch(block, num_to_forward_prefetch):
    layers = block.layers
    for i, layer in enumerate(layers):
        if i < num_to_forward_prefetch:
            continue
        layers_to_prefetch = [layers[i + j] for j in range(1, num_to_forward_prefetch + 1)]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def set_modules_to_backward_prefetch(block, num_to_backward_prefetch):
    layers = block.layers
    for i, layer in enumerate(layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [layers[i - j] for j in range(1, num_to_backward_prefetch + 1)]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)

#是否支持prefetch的 model
def _is_prefetch_target_model(model_id):
    model_id = str(model_id).lower()
    return (
        "qwen3-30b-a3b" in model_id
        or "qwen3-coder-next" in model_id
    )


def _to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def patch_native_fsdp_prefetch(
    block,
    prefetch_forward_layers,
    prefetch_backward_layers,
):
    from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy

    original_wrap_model = NativeFSDPStrategy.wrap_model

    @functools.wraps(original_wrap_model)
    def patched_wrap_model(self, model, optimizer=None):
        result = original_wrap_model(self, model, optimizer)

        if isinstance(result, tuple):
            wrapped_model, wrapped_optimizer = result
        else:
            wrapped_model, wrapped_optimizer = result, optimizer

        resolved_block = _resolve_prefetch_block(wrapped_model, block)

        set_modules_to_forward_prefetch(resolved_block, prefetch_forward_layers)
        set_modules_to_backward_prefetch(resolved_block, prefetch_backward_layers)

        print(
            "[prefetch][native] set done:",
            f"forward={prefetch_forward_layers}, "
            f"backward={prefetch_backward_layers}"
        )

        if isinstance(result, tuple):
            return wrapped_model, wrapped_optimizer
        return wrapped_model

    NativeFSDPStrategy.wrap_model = patched_wrap_model


def _resolve_prefetch_block(model, block):
    if not isinstance(block, str):
        return block
    obj = model
    for attr in block.split('.'):
        obj = getattr(obj, attr)
    if hasattr(obj, 'layers'):
        return obj
    return SimpleNamespace(layers=obj)


def _install_prefetch_patch(block, args):
    prefetch = _to_bool(args.prefetch)
    prefetch_forward_layers = int(args.prefetch_forward_layers)
    prefetch_backward_layers = int(args.prefetch_backward_layers)

    if not prefetch:
        print(
            "[prefetch] skip patch:",
            f"enabled={prefetch}"
        )
        return

    if not _is_prefetch_target_model(args.model_id):
        print(
            "[prefetch] skip patch:",
            f"enabled={prefetch}, model_id={args.model_id}"
        )
        return

    print(
        "[prefetch] install patch:",
        f"model_id={args.model_id}, "
        f"forward={prefetch_forward_layers}, "
        f"backward={prefetch_backward_layers}"
    )

    patch_native_fsdp_prefetch(
        block=block,
        prefetch_forward_layers=prefetch_forward_layers,
        prefetch_backward_layers=prefetch_backward_layers,
    )
