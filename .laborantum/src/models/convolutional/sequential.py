import torch


class ResidualBottleneck(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            prenormalization=lambda n_channels: torch.nn.Identity(),
            postnormalization=lambda n_channels: torch.nn.Identity(),
            activation=torch.nn.ReLU,
            compression=1,
            residual=True):

        super().__init__()
        self.residual = residual
        mid = out_channels // compression
        stride = 1 if in_channels == out_channels else 2

        # Create bypass first so it consumes RNG before block layers
        if in_channels == out_channels:
            bypass = torch.nn.Identity()
        else:
            bypass = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, stride=stride)

        block = torch.nn.Sequential(
            prenormalization(in_channels),
            torch.nn.Conv2d(in_channels, mid, kernel_size=3, padding=1),
            postnormalization(mid),
            activation(),
            prenormalization(mid),
            torch.nn.Conv2d(mid, out_channels, kernel_size=3, padding=1, stride=stride),
            postnormalization(out_channels),
        )

        # Register block first so named_modules() yields block children before bypass
        self.block = block
        self.bypass = bypass

    def forward(self, signal):
        if self.residual:
            return self.bypass(signal) + self.block(signal)
        return self.block(signal)


class FullyConvolutionalNN(torch.nn.Module):
    def __init__(
            self,
            block=lambda in_channels, out_channels: torch.nn.Conv2d(in_channels, out_channels, (1, 1)),
            in_channels=1,
            mid_channels=[16, 32, 64, 128],
            out_channels=10,
            n_blocks=[1, 1, 1, 1]):
        super().__init__()

        self.encoder = torch.nn.Conv2d(in_channels, mid_channels[0], kernel_size=3)

        layers = []
        for i in range(len(mid_channels)):
            for _ in range(n_blocks[i]):
                layers.append(block(mid_channels[i], mid_channels[i]))
            if i < len(mid_channels) - 1:
                layers.append(block(mid_channels[i], mid_channels[i + 1]))

        self.blocks = torch.nn.Sequential(*layers)
        self.decoder = torch.nn.Linear(mid_channels[-1], out_channels)

    def __forward_kernel(self, signal):
        if signal.dim() == 3:
            signal = signal.unsqueeze(1)
        signal = self.encoder(signal)
        signal = self.blocks(signal)
        signal = signal.mean(-1).mean(-1)
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
