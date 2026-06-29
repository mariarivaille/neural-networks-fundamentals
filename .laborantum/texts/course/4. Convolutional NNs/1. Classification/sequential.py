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
        self.block = torch.nn.Identity()
        self.bypass = torch.nn.Identity()
        self.residual = residual

        ## YOUR CODE HERE

    def forward(self, signal):
        ## YOUR CODE HERE

        return signal


class FullyConvolutionalNN(torch.nn.Module):
    def __init__(
            self,
            block=lambda in_channels, out_channels: torch.nn.Conv2d(in_channels, out_channels, (1, 1)),
            in_channels=1,
            mid_channels=[16, 32, 64, 128],
            out_channels=10,
            n_blocks=[1, 1, 1, 1]):
        ...
        ## YOUR CODE HERE
        # Define network modules in the constructor


    def __forward_kernel(self, signal):
        ## YOUR CODE HERE
        # Pass the signal through the modules in forward


        return signal

    def forward(self, batch):
        signal = batch['data']['image']
        signal = self.__forward_kernel(signal)

        # Put the result into the batch
        batch['signals'] = {'output': signal}

        # Perform postprocessing after we get the output
        self.postprocessing(batch)

        return batch

    def postprocessing(self, batch):

        # Take network's output from the batch
        signal = batch['signals']['output']

        ## YOUR CODE HERE

        # Put the processed result into the batch
        batch['postprocessed'] = {'class': signal}
