import torch
import matplotlib.pyplot as plt
from torchview import draw_graph


def draw_forward_kernel_graph(
        model,
        input_shape=(3, 28, 28),
        graph_name='SimpleFCNN',
        expand_nested=True,
        depth=5):
    class _ForwardKernelAdapter(torch.nn.Module):
        def __init__(self, wrapped_model):
            super().__init__()
            self.wrapped_model = wrapped_model
            kernel_name = f'_{wrapped_model.__class__.__name__}__forward_kernel'
            self.forward_kernel = getattr(wrapped_model, kernel_name)

        def forward(self, signal):
            return self.forward_kernel(signal)

    adapter = _ForwardKernelAdapter(model)
    return draw_graph(
        adapter,
        input_data=torch.randn(*input_shape),
        expand_nested=expand_nested,
        graph_name=graph_name,
        depth=depth)


def plot_signal_stats(
        model,
        input_batch,
        log_scale=True,
        epsilon=1e-12):
    module_names = []

    max_grads = []
    mean_grads = []

    max_weights = []
    mean_weights = []

    max_signals = []
    mean_signals = []

    signal_stats = {}
    hooks = []

    def safe_value(value):
        if value is None:
            return epsilon
        return max(value, epsilon)

    def make_hook(module_name):
        def hook(module, inputs, output):
            if torch.is_tensor(output):
                signal = output.detach().abs().flatten()

                signal_stats[module_name] = {
                    'max': safe_value(signal.max().item()),
                    'mean': safe_value(signal.mean().item()),
                }

            elif isinstance(output, (tuple, list)):
                tensors = [
                    item.detach().abs().flatten()
                    for item in output
                    if torch.is_tensor(item)
                ]

                if tensors:
                    signal = torch.cat(tensors)

                    signal_stats[module_name] = {
                        'max': safe_value(signal.max().item()),
                        'mean': safe_value(signal.mean().item()),
                    }

        return hook

    for name, module in model.named_modules():
        if name == '':
            continue

        has_own_params = any(
            True for _ in module.parameters(recurse=False)
        )

        if has_own_params:
            hooks.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            _ = model(input_batch)
    finally:
        for hook in hooks:
            hook.remove()

        if was_training:
            model.train()

    for name, module in model.named_modules():
        if name == '':
            continue

        module_grad_values = []
        module_weight_values = []

        for param in module.parameters(recurse=False):
            module_weight_values.append(param.detach().abs().flatten())

            if param.grad is not None:
                module_grad_values.append(param.grad.detach().abs().flatten())

        if not module_weight_values:
            continue

        module_names.append(name)

        weights = torch.cat(module_weight_values)

        max_weights.append(safe_value(weights.max().item()))
        mean_weights.append(safe_value(weights.mean().item()))

        if module_grad_values:
            grads = torch.cat(module_grad_values)

            max_grads.append(safe_value(grads.max().item()))
            mean_grads.append(safe_value(grads.mean().item()))
        else:
            max_grads.append(epsilon)
            mean_grads.append(epsilon)

        if name in signal_stats:
            max_signals.append(signal_stats[name]['max'])
            mean_signals.append(signal_stats[name]['mean'])
        else:
            max_signals.append(epsilon)
            mean_signals.append(epsilon)

    plt.figure(figsize=(14, 6))

    plt.plot(
        module_names,
        max_grads,
        marker='o',
        linestyle='-',
        color='tab:blue',
        label='max |grad|')
    plt.plot(
        module_names,
        mean_grads,
        marker='o',
        linestyle=':',
        color='tab:blue',
        label='mean |grad|')

    plt.plot(
        module_names,
        max_weights,
        marker='o',
        linestyle='-',
        color='tab:orange',
        label='max |weight|')
    plt.plot(
        module_names,
        mean_weights,
        marker='o',
        linestyle=':',
        color='tab:orange',
        label='mean |weight|')

    plt.plot(
        module_names,
        max_signals,
        marker='o',
        linestyle='-',
        color='tab:green',
        label='max |signal|')
    plt.plot(
        module_names,
        mean_signals,
        marker='o',
        linestyle=':',
        color='tab:green',
        label='mean |signal|')

    if log_scale:
        plt.yscale('log')

    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Module')
    plt.ylabel('Amplitude')
    plt.title('Gradient, Weight, and Signal Amplitudes per Module')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
