import os
import gzip
from urllib.request import urlretrieve
from typing import Optional, Tuple

import numpy as np


def data(path):
    url = 'http://yann.lecun.com/exdb/mnist/'
    files = ['train-images-idx3-ubyte.gz',
             'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz']

    os.makedirs(path, exist_ok=True)

    for file in files:
        if file not in os.listdir(path):
            urlretrieve(url + file, os.path.join(path, file))
            print(f'Downloaded {file} to {path}')


    def _images(path):
        with gzip.open(path) as f:
            pixels = np.frombuffer(f.read(), 'B', offset=16)
        return pixels.reshape(-1, 28, 28).astype('float32') / 255

    def _labels(path):
        with gzip.open(path) as f:
            integer_labels = np.frombuffer(f.read(), 'B', offset=8)


        def _onehot(integer_labels):
            n_rows = len(integer_labels)
            n_cols = integer_labels.max() + 1
            onehot = np.zeros((n_rows, n_cols), dtype='uint8')
            onehot[np.arange(n_rows), integer_labels] = 1
            return onehot

        return _onehot(integer_labels)

    train_images = _images(os.path.join(path, files[0]))
    train_labels = _labels(os.path.join(path, files[1]))

    test_images = _images(os.path.join(path, files[2]))
    test_labels = _labels(os.path.join(path, files[3]))

    return train_images, train_labels, test_images, test_labels

if __name__ == '__main__':
    x_train,y_train,x_test,y_test = data('.')
