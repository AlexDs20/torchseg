import os
import gzip
from urllib.request import urlretrieve

import numpy as np
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    """
    https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """
    def __init__(self, path='.', batch_size=32):
        super(DataModule, self).__init__()
        self.path = path
        self.batch_size = batch_size

    def prepare_data(self):
        """
        Run single process on CPU
        """
        # Download, tokenize, ...
        url = 'http://yann.lecun.com/exdb/mnist/'
        self.files = ['train-images-idx3-ubyte.gz',
                 'train-labels-idx1-ubyte.gz',
                 't10k-images-idx3-ubyte.gz',
                 't10k-labels-idx1-ubyte.gz']

        os.makedirs(self.path, exist_ok=True)
        for file in self.files:
            if file not in os.listdir(self.path):
                urlretrieve(url + file, os.path.join(self.path, file))
                print(f'Downloaded {file} to {self.path}')

    def setup(self):
        """
        Operations on every GPU
        """
        # split, transform, ...
        train_images = self._images(os.path.join(self.path, self.files[0]))
        train_labels = self._labels(os.path.join(self.path, self.files[1]))

        test_images = self._images(os.path.join(self.path, self.files[2]))
        test_labels = self._labels(os.path.join(self.path, self.files[3]))

        self.train = CustomDataset(train_images, train_labels)
        self.val   = CustomDataset(test_images, test_labels)
        #self.test =

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def _images(self, path):
        with gzip.open(path) as f:
            pixels = np.frombuffer(f.read(), 'B', offset=16)
        return pixels.reshape(-1, 1, 28, 28).astype('float32') / 255

    def _labels(self, path):
        with gzip.open(path) as f:
            integer_labels = np.frombuffer(f.read(), 'B', offset=8)

        def _onehot(integer_labels):
            n_rows = len(integer_labels)
            n_cols = integer_labels.max() + 1
            onehot = np.zeros((n_rows, n_cols), dtype='uint8')
            onehot[np.arange(n_rows), integer_labels] = 1
            return onehot

        return _onehot(integer_labels)


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


if __name__=='__main__':
    dm = DataModule('.')

    dm.prepare_data()
    dm.setup()
    t_dl = dm.train_dataloader()

    for batch in t_dl:
        x, y = batch
        print(x.shape, y.shape)
        quit()
