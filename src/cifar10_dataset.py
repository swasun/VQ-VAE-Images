import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os


class Cifar10Dataset(object):

    def __init__(self, batch_size, path):
        if not os.path.isdir(path):
            os.mkdir(path)

        self._training_data = datasets.CIFAR10(
            root=path,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        )

        self._validation_data = datasets.CIFAR10(
            root=path,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        )

        self._training_loader = DataLoader(
            self._training_data, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True
        )

        self._validation_loader = DataLoader(
            self._validation_data,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )

        self._train_data_variance = np.var(self._training_data.train_data / 255.0)

    @property
    def training_data(self):
        return self._training_data

    @property
    def validation_data(self):
        return self._validation_data

    @property
    def training_loader(self):
        return self._training_loader

    @property
    def validation_loader(self):
        return self._validation_loader

    @property
    def train_data_variance(self):
        return self._train_data_variance
