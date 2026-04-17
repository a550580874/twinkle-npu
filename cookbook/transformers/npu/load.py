def set_modules_to_forward_prefetch(block, num_to_forward_prefetch):
    for i, layer in enumerate(block.layers):
        if i < num_to_forward_prefetch:
            continue
        layers_to_prefetch = [layers[i + j] for j in range(1, num_to_forward_prefetch + 1)]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def set_modules_to_backward_prefetch(block, num_to_backward_prefetch):
    for i, layer in enumerate(block.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [layers[i - j] for j in range(1, num_to_backward_prefetch + 1)]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)
