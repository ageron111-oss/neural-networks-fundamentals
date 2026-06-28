import torch

class SimpleFCNN(torch.nn.Module):
    def __init__(
            self, 
            channels=None,
            n_classes=10,
            activation=torch.nn.LeakyReLU):
        super().__init__()
        self.n_classes = n_classes
        self.activation = activation()
        self.lin1 = torch.nn.Linear(784, 128)
        #self.lin2 = torch.nn.Linear(256, 128)
        #self.lin3 = torch.nn.Linear(128, 64)
        self.proj = torch.nn.Linear(128, self.n_classes)
        ## YOUR CODE HERE
        # Define network modules in the constructor
        
        
    def __forward_kernel(self, signal):
        signal = signal.reshape([signal.shape[0], -1])
        signal = self.lin1(signal)
        signal = self.activation(signal)
        #signal = self.lin2(signal)
        #signal = self.activation(signal)
        #signal = self.lin3(signal)
        #signal = self.activation(signal)
        logits = self.proj(signal)
        return logits
        ## YOUR CODE HERE
        # Pass the signal through the modules in forward
        

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
        signal = torch.argmax(signal, dim = 1)
        ## YOUR CODE HERE
        
        # Put the processed result into the batch
        batch['postprocessed'] = {'class': signal}
