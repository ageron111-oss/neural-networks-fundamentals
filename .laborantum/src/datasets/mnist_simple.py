import torchvision.datasets

class MNISTSimpleDataset:
    def __init__(self, train=True):
        ...
        ## Load MNIST dataset here
        ## YOUR CODE HERE
        import os
        self.mnist = torchvision.datasets.MNIST(
            root=os.path.expanduser('~/mnist_data'),
            train=train,
            download=True,
            transform=None  
        )
        self.X = self.mnist.data  
        self.y = self.mnist.targets  

    def __len__(self):
        ## Return number of items that is there in the dataset
        ## YOUR CODE HERE
        return len(self.X)


    def __getitem__(self, index):
        ## Return a sample of the dataset that corresponds to the input index
        ## YOUR CODE HERE
        image = self.X[index]  
        label = self.y[index]  
        
        image = image.float()  
        image = (image / 127.5) - 1.0         
        label = label.long()
        
        return {
            'image': image,
            'label': label
        }
