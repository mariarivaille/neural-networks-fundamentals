import os
import torch
import torchvision.datasets

class MNISTSimpleDataset:
    def __init__(self, train=True):
        root = os.path.expanduser('~')
        dataset = torchvision.datasets.MNIST(root=root, train=train, download=True)
        self.X = dataset.data
        self.y = dataset.targets

    def __len__(self):
        res = len(self.X)
        return res


    def __getitem__(self, index):
        image = self.X[index].float()
        image = (image / 255.0) * 2.0 - 1.0
        label = torch.as_tensor(self.y[index], dtype=torch.long)
        sample = {
            "image": image,
            "label": label
        }
        return sample 
