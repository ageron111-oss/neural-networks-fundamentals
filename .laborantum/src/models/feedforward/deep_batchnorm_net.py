import torch

class BatchNorm(torch.nn.Module):
    def __init__(self, beta=0.90, eps=1.0e-4):
        super().__init__()

        self.beta = beta
        self.eps = eps
        self.register_buffer('running_mean', None)
        self.register_buffer('running_var', None)

        ## YOUR CODE HERE


    def _init_stats(self, signal):
        channels = signal.shape[1]
        shape = [1, channels]
        self.running_mean = torch.zeros(shape, device=signal.device, dtype=signal.dtype)
        self.running_var = torch.ones(shape, device=signal.device, dtype=signal.dtype)

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
            mean = signal.mean(dim=0, keepdim=True)
            var = signal.var(dim=0, keepdim=True, unbiased=False)
            self.running_mean = self.beta * self.running_mean + (1 - self.beta) * mean
            self.running_var = self.beta * self.running_var + (1 - self.beta) * var
            
        else:
            ## YOUR CODE HERE
            mean = self.running_mean
            var = self.running_var

        signal = (signal - mean) / torch.sqrt(var + self.eps)

        return signal


class Residual(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        ## YOUR CODE HERE
        self.module = module

    def forward(self, signal):
        ## YOUR CODE HERE

        return signal + self.module(signal)


class Bottleneck(torch.nn.Module):
    def __init__(
            self, 
            in_channels,
            prenormalization=torch.nn.Identity,
            postnormalization=torch.nn.Identity,
            activation=torch.nn.LeakyReLU,
            compression=1,
            **kwargs):

        super().__init__()
        hidden_channels = in_channels // compression

        self.block = torch.nn.Sequential(
            prenormalization(),
            torch.nn.Linear(in_channels, hidden_channels),
            postnormalization(),
            activation(),
            prenormalization(),
            torch.nn.Linear(hidden_channels, in_channels),
            postnormalization()
        )
        self.activation = activation()

        ## YOUR CODE HERE

    def forward(self, signal):
        ## YOUR CODE HERE

        return signal + self.block(signal)



class DeepFullyConnectedNet(torch.nn.Module):
    def __init__(
            self,
            block=lambda n_channels: torch.nn.Linear(n_channels, n_channels),
            dim_input=28 * 28,
            dim_embed=128,
            dim_output=10,
            n_blocks=3):
        super().__init__()
        self.input_dim = dim_input
        self.emb_dim = dim_embed
        self.out_dim = dim_output
        self.blocks = torch.nn.ModuleList([
            block(self.emb_dim) for _ in range(n_blocks)
        ])
        self.encoder = torch.nn.Linear(self.input_dim, self.emb_dim)
        self.proj = torch.nn.Linear(self.emb_dim, self.out_dim)
        self.activation = torch.nn.LeakyReLU()
        ## YOUR CODE HERE
        # Define network modules in the constructor


    def __forward_kernel(self, signal):
        signal = signal.reshape([signal.shape[0], -1])
        signal = self.encoder(signal)
        for block in self.blocks:
            signal = block(signal)
            signal = self.activation(signal)
        signal = self.proj(signal)
        return signal
        ## YOUR CODE HERE
        # Pass the signal through the modules in forward



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
        signal = torch.argmax(signal, dim=-1)
        # Put the processed result into the batch
        batch['postprocessed'] = {'class': signal}
