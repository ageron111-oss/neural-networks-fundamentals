import torch

class Autoencoder(torch.nn.Module):
    def __init__(
            self,
            channels,
            activation=torch.nn.LeakyReLU):
        ...
        super().__init__()
        ## YOUR CODE HERE
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            activation(),
            torch.nn.Linear(256, 128))
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            activation(),
            torch.nn.Linear(256, 784))

    def __forward_kernel(self, signal):
        input_shape = signal.shape
        signal = signal.reshape([input_shape[0], -1])
        signal = self.encoder(signal)
        signal = self.decoder(signal)
        ## YOUR CODE HERE
        signal = signal.reshape(input_shape)
        return signal

    def forward(self, batch):
        ## YOUR CODE HERE
        signal = batch['data']['image']
        signal = self.__forward_kernel(signal)
        
        # Put the result into the batch
        batch['signals'] = {'reconstruction': signal}
        return batch
