import torch


class BatchNorm(torch.nn.Module):
    def __init__(self, channels=None, beta=0.90, eps=1.0e-4):
        super().__init__()

        self.beta = beta
        self.eps = eps
        self.register_buffer('running_mean', None)
        self.register_buffer('running_var', None)

    def _init_stats(self, signal):
        channels = signal.shape[1]
        shape = [1, channels]
        self.running_mean = torch.zeros(shape, dtype=signal.dtype, device=signal.device)
        self.running_var = torch.ones(shape, dtype=signal.dtype, device=signal.device)

    def _check_stats(self, signal):
        return (
            self.running_mean is not None
            and self.running_mean.device == signal.device
            and self.running_mean.dtype == signal.dtype
            and self.running_mean.ndim == signal.ndim
            and self.running_mean.shape[1] == signal.shape[1]
        )

    def forward(self, signal):
        if not self._check_stats(signal):
            self._init_stats(signal)

        if self.training:
            batch_mean = signal.mean(dim=0, keepdim=True)
            batch_var = ((signal - batch_mean) ** 2).mean(dim=0, keepdim=True)

            self.running_mean = self.beta * self.running_mean + (1 - self.beta) * batch_mean.detach()
            self.running_var = self.beta * self.running_var + (1 - self.beta) * batch_var.detach()

            signal = (signal - batch_mean) / (batch_var.sqrt() + self.eps)
        else:
            signal = (signal - self.running_mean) / (self.running_var.sqrt() + self.eps)

        return signal


class Residual(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, signal):
        return signal + self.module(signal)


class Bottleneck(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            prenormalization=torch.nn.Identity,
            postnormalization=torch.nn.Identity,
            activation=torch.nn.ReLU,
            compression=1,
            **kwargs):

        super().__init__()
        mid_channels = in_channels // compression
        self.block = torch.nn.Sequential(
            prenormalization(in_channels),
            torch.nn.Linear(in_channels, mid_channels),
            postnormalization(mid_channels),
            activation(),
            prenormalization(mid_channels),
            torch.nn.Linear(mid_channels, in_channels),
            postnormalization(in_channels),
        )

    def forward(self, signal):
        return self.block(signal)


class DeepFullyConnectedNet(torch.nn.Module):
    def __init__(
            self,
            block=lambda n_channels: torch.nn.Linear(n_channels, n_channels),
            dim_input=28 * 28,
            dim_embed=128,
            dim_output=10,
            n_blocks=3):
        super().__init__()
        self.encoder = torch.nn.Linear(dim_input, dim_embed)
        self.backbone = torch.nn.Sequential(*[block(dim_embed) for _ in range(n_blocks)])
        self.decoder = torch.nn.Linear(dim_embed, dim_output)

    def __forward_kernel(self, signal):
        signal = signal.reshape([signal.shape[0], -1])
        signal = self.encoder(signal)
        signal = self.backbone(signal)
        signal = self.decoder(signal)
        return signal

    def forward(self, batch):
        signal = batch['data']['image']
        signal = self.__forward_kernel(signal)

        batch['signals'] = {'output': signal}

        self.postprocessing(batch)

        return batch

    def postprocessing(self, batch):
        signal = batch['signals']['output']
        signal = signal.argmax(dim=1)
        batch['postprocessed'] = {'class': signal}
