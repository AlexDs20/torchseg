import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def _load_file(path):
    ext = os.path.splitext(path)[1]
    if ext == '.png':
        data = Image.open(path)
    else:
        raise ValueError(f'Extension {ext} not implemented!')
    return data


class FolderDataSet(Dataset):
    def __init__(self, path: str,
                 image_transforms=None,
                 target_transforms=None):
        super(FolderDataSet, self).__init__()
        image_folder = 'images'
        target_folder = 'targets'

        self.images = [os.path.join(path, image_folder, f) for f in os.listdir(os.path.join(path, image_folder))]
        self.targets = [os.path.join(path, target_folder, os.path.basename(f)) for f in self.images]

        self.image_transforms = image_transforms
        self.target_transforms = target_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file_path = self.images[idx]
        target_path = self.targets[idx]

        # Read the file
        image = _load_file(file_path)
        target = _load_file(target_path)

        # Apply transforms
        if self.image_transforms is not None:
            image = self.image_transforms(image)
        if self.target_transforms is not None:
            target = self.target_transforms(target)

        # TODO: Fix how target dtype is handled
        return image, target.to(torch.long)
