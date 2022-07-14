import os
import glob
from torch.utils.data import Dataset
import numpy as np


class FolderDataSet(Dataset):
    def __init__(self, path: str):
        super(FolderDataSet, self).__init__()
        image_folder = 'images'
        label_folder = 'labels'

        self.images = glob.glob(os.path.join(path, image_folder, '*.npy'))
        self.labels = [os.path.join(path, label_folder, os.path.basename(f)) for f in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file_path = self.images[idx]
        label_path = self.labels[idx]

        # Read the file
        image = np.load(file_path)
        label = np.load(label_path).astype(int)

        return image, label


class StackedDataSet(Dataset):
    def __init__(self, path: str):
        super(StackedDataSet, self).__init__()
        self.images = glob.glob(os.path.join(path, '*.npy'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file_path = self.images[idx]

        # Read the file
        data = np.load(file_path)
        image = data[:-1]
        label = data[-1]

        return image, label
