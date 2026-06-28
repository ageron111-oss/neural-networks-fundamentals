import torch


class ResidualBottleneck(torch.nn.Module):
    def __init__(
            self, 
            in_channels,
            out_channels,
            prenormalization=lambda n_channels: torch.nn.Identity(),
            postnormalization=lambda n_channels: torch.nn.Identity(),
            activation=torch.nn.LeakyReLU,
            compression=1,
            residual=True):

        super().__init__()

        self.residual = residual
        hidden_channels = out_channels // compression

        self.block = torch.nn.Sequential(
            prenormalization(in_channels),
            torch.nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            postnormalization(hidden_channels),
            activation(),
            prenormalization(hidden_channels),
            torch.nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            postnormalization(out_channels)
        )
        
        if in_channels != out_channels:
            self.bypass = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.bypass = torch.nn.Identity()
        
        self.activation = activation()


        ## YOUR CODE HERE

    def forward(self, signal):
        ## YOUR CODE HERE

        out = self.block(signal)

        if self.residual:
            out = out + self.bypass(signal)
            out = self.activation(out)
        else:
            out = self.activation(out)
        
        return out



class FullyConvolutionalNN(torch.nn.Module):
    def __init__(
            self,
            block=lambda in_channels, out_channels: torch.nn.Conv2d(in_channels, out_channels, (1, 1)),
            in_channels=1,
            mid_channels=[16, 32, 64, 128],
            out_channels=10,
            n_blocks=[1, 1, 1, 1]):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks

        self.levels = torch.nn.ModuleList()

        current_channels = in_channels
        
        for level_idx, (channels, n_blocks_level) in enumerate(zip(mid_channels, n_blocks)):
            level = torch.nn.ModuleList()
            
            if level_idx == 0:
                level.append(block(current_channels, channels))
            else:
                level.append(
                    torch.nn.Sequential(
                        block(current_channels, channels),
                        torch.nn.Conv2d(channels, channels, kernel_size=2, stride=2)
                    )
                )
            
            current_channels = channels

            for _ in range(n_blocks_level - 1):
                level.append(block(current_channels, current_channels))
            
            self.levels.append(level)

        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = torch.nn.Linear(current_channels, out_channels)
        ## YOUR CODE HERE
        # Define network modules in the constructor


    def __forward_kernel(self, signal):
        ## YOUR CODE HERE
        # Pass the signal through the modules in forward
        if signal.dim() == 3:
            signal = signal.unsqueeze(1)
    
        for level in self.levels:
            for block in level:
                signal = block(signal)
    
        signal = self.global_avg_pool(signal)
        signal = signal.reshape([signal.shape[0], -1])
        signal = self.classifier(signal)
        
        return signal


    def forward(self, batch):
        signal = batch['data']['image']
        if signal.dim() == 3:  
            signal = signal.unsqueeze(1) 
        signal = self.__forward_kernel(signal)

        # Put the result into the batch
        batch['signals'] = {'output': signal}

        # Perform postprocessing after we get the output
        self.postprocessing(batch)

        return batch

    def postprocessing(self, batch):

        # Take network's output from the batch
        signal = batch['signals']['output']
        signal = torch.argmax(signal, dim=-1)
        ## YOUR CODE HERE

        # Put the processed result into the batch
        batch['postprocessed'] = {'class': signal}
