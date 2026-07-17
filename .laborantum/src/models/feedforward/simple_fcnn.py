import torch

class SimpleFCNN(torch.nn.Module):
    def __init__(
            self,
            channels=None,
            n_classes=10,
            activation=torch.nn.ReLU):
        super().__init__()

        if channels is None:
            channels = []

        in_features = 784  # 28 * 28

        # If the first channel equals the input size, it is an input-size
        # specification rather than a hidden-layer request, so strip it.
        if channels and channels[0] == in_features:
            hidden_channels = list(channels[1:])
        else:
            hidden_channels = list(channels)

        backbone_layers = []
        for out_features in hidden_channels:
            backbone_layers.append(torch.nn.Linear(in_features, out_features))
            backbone_layers.append(activation())
            in_features = out_features

        self.backbone = torch.nn.Sequential(*backbone_layers)
        self.classifier = torch.nn.Linear(in_features, n_classes)


    def __forward_kernel(self, signal):
        signal = signal.reshape([signal.shape[0], -1])
        signal = self.backbone(signal)
        signal = self.classifier(signal)
        return signal

    def forward(self, batch):
        signal = batch['data']['image']
        signal = self.__forward_kernel(signal)

        # Put the result into the batch
        batch['signals'] = {'output': signal}

        # Perform postprocessing after we get the output
        self.postprocessing(batch)

        return batch['signals']['output']

    def postprocessing(self, batch):

        # Take network's output from the batch
        signal = batch['signals']['output']

        signal = torch.argmax(signal, dim=1)

        # Put the processed result into the batch
        batch['postprocessed'] = {'class': signal}
