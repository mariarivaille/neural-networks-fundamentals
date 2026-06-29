import torch

class BatchNorm(torch.nn.Module):
    def __init__(self, beta=0.90, eps=1.0e-4):
        super().__init__()

        self.beta = beta
        self.eps = eps
        self.register_buffer('running_mean', None)
        
        ## YOUR CODE HERE


    def _init_stats(self, signal):
        channels = signal.shape[1]
        shape = [1, channels]

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
            ## YOUR CODE HERE

            
        else:
            ## YOUR CODE HERE


        return signal


class Residual(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        ## YOUR CODE HERE

    def forward(self, signal):
        ## YOUR CODE HERE

        return signal


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
        self.block = torch.nn.Identity()

        ## YOUR CODE HERE

    def forward(self, signal):
        ## YOUR CODE HERE

        return signal



class DeepFullyConnectedNet(torch.nn.Module):
    def __init__(
            self,
            block=lambda n_channels: torch.nn.Linear(n_channels, n_channels),
            dim_input=28 * 28,
            dim_embed=128,
            dim_output=10,
            n_blocks=3):
        ...
        ## YOUR CODE HERE
        # Define network modules in the constructor


    def __forward_kernel(self, signal):
        signal = signal.reshape([signal.shape[0], -1])
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
