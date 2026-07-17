import torch

class Autoencoder(torch.nn.Module):
    def __init__(
            self,
            channels,
            activation=torch.nn.ReLU):
        ...
        super().__init__()

        in_features = channels[0]
        encoder_sizes = channels[1:]
        decoder_sizes = channels[-2::-1]

        encoder_layers = []
        current = in_features
        for i, size in enumerate(encoder_sizes):
            encoder_layers.append(torch.nn.Linear(current, size))
            if i < len(encoder_sizes) - 1:
                encoder_layers.append(activation())
            current = size
        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_layers = []
        current = encoder_sizes[-1]
        for i, size in enumerate(decoder_sizes):
            decoder_layers.append(torch.nn.Linear(current, size))
            if i < len(decoder_sizes) - 1:
                decoder_layers.append(activation())
            current = size
        self.decoder = torch.nn.Sequential(*decoder_layers)

        if not hasattr(self, 'encoder'):
            self.encoder = torch.nn.Identity()
        if not hasattr(self, 'decoder'):
            self.decoder = torch.nn.Identity()

    def __forward_kernel(self, signal):
        input_shape = signal.shape
        res = signal
        res = res.reshape(input_shape[0], -1)
        res = self.encoder(res)
        res = self.decoder(res)
        res = res.reshape(input_shape)
        return res

    def forward(self, batch):
        signal = batch['data']['image']
        signal = self.__forward_kernel(signal)
        batch['signals'] = {'reconstruction': signal}
        if 'signals' not in batch:
            batch['signals'] = {'reconstruction': batch['data']['image']}
        return batch
