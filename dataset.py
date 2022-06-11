import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from data.mnist.load_mnist import mnist

class CustomDataset(Dataset):
    def __init__(self, directory='data/mnist/', train=True):
        self.train = train

        x_train, y_train, x_test, y_test = mnist(directory)

        if self.train:
            self.x_train = x_train
            self.y_train = y_train
        else:
            self.x_test = x_test
            self.y_test = y_test

    def __len__(self):
        return self.x_train.shape[0] if self.train else self.x_test.shape[0]

    def __getitem__(self, idx):
        if self.train:
            return self.x_train[idx][np.newaxis,...], self.y_train[idx]
        else:
            return self.x_test[idx][np.newaxis,...], self.y_test[idx]


train_dataset = CustomDataset(directory='data/mnist', train=True)
test_dataset = CustomDataset(directory='data/mnist', train=False)
if __name__ == '__main__':
    train_dataset = CustomDataset(directory='data/mnist', train=True)
    test_dataset = CustomDataset(directory='data/mnist', train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)
