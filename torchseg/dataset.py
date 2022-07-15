import os
from torch.utils.data import Dataset

import numpy as np
from PIL import Image


def _load_file(path):
    ext = os.path.splitext(path)[1]
    if ext == '.npy':
        data = np.load(path)
    elif ext == '.png':
        data = np.array(Image.open(path))
    else:
        raise ValueError(f'Extension {ext} not implemented!')
    return data


class FolderDataSet(Dataset):
    def __init__(self, path: str):
        super(FolderDataSet, self).__init__()
        image_folder = 'images'
        label_folder = 'labels'

        self.images = [os.path.join(path, image_folder, f) for f in os.listdir(os.path.join(path, image_folder))]
        self.labels = [os.path.join(path, label_folder, os.path.basename(f)) for f in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file_path = self.images[idx]
        label_path = self.labels[idx]

        # Read the file
        image = np.moveaxis(_load_file(file_path), -1, 0).astype(np.float32)
        label = _load_file(label_path)[None].astype(int)
        return image, label


class StackedDataSet(Dataset):
    def __init__(self, path: str):
        super(StackedDataSet, self).__init__()
        self.images = [os.path.join(path, f) for f in os.listdir(path)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file_path = self.images[idx]

        # Read the file
        data = _load_file(file_path)
        image = data[:-1]
        label = data[-1].astype(int)

        return image, label
