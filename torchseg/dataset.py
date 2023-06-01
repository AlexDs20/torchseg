import os
import torch
import torchvision.transforms as transforms
import torchseg.transforms
from torch.utils.data import Dataset

import numpy as np
from PIL import Image


def _load_file(path):
    ext = os.path.splitext(path)[1]
    if ext == '.png':
        data = Image.open(path)
    else:
        raise ValueError(f'Extension {ext} not implemented!')
    return data


def get_transforms(config):
    transform_list = []

    for key, val in config.items():
        if hasattr(torchseg.transforms, key):
            cls = getattr(torchseg.transforms, key)
        elif hasattr(transforms, key):
            cls = getattr(transforms, key)

        if isinstance(val, dict):
            tf = cls(**val)
        else:
            tf = cls(val) if val is not None else cls()
        transform_list.append(tf)

    # TODO: Add support to run the transforms in random order
    return transforms.Compose(transform_list)


class FolderDataSet(Dataset):
    def __init__(self, path: str,
                 image_transforms=None,
                 target_transforms=None,
                 *args, **kwargs):
        super(FolderDataSet, self).__init__()
        image_folder = 'images'
        target_folder = 'targets'

        self.images = [os.path.join(path, image_folder, f) for f in os.listdir(os.path.join(path, image_folder))]
        self.targets = [os.path.join(path, target_folder, os.path.basename(f)) for f in self.images]

        if image_transforms is not None:
            self.image_transforms = get_transforms(image_transforms)
        else:
            self.image_transforms = lambda x: x

        if target_transforms is not None:
            self.target_transforms = get_transforms(target_transforms)
        else:
            self.target_transforms = lambda x: x

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

        return image, target


class InferenceDataSet(Dataset):
    def __init__(self, path: str,
                 image_transforms=None,
                 *args, **kwargs):
        super(InferenceDataSet, self).__init__()

        if os.path.isdir(path):
            self.images = [os.path.join(path, f) for f in os.listdir(os.path.join(path))]
        elif os.path.isfile(path):
            self.images = [path]
        else:
            raise ValueError(f'Wrong input path to file/folder: {path}.')

        if image_transforms is not None:
            self.image_transforms = get_transforms(image_transforms)
        else:
            self.image_transforms = lambda x: x

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file_path = self.images[idx]

        # Read the file
        image = _load_file(file_path)

        # Apply transforms
        if self.image_transforms is not None:
            image = self.image_transforms(image)

        return image


class AutoEncoderDataSet(Dataset):
    def __init__(self, path: str,
                 image_transforms=None,
                 *args,
                 **kwargs):
        super(AutoEncoderDataSet, self).__init__()
        image_folder = 'images'

        self.images = [os.path.join(path, image_folder, f) for f in os.listdir(os.path.join(path, image_folder))]

        if image_transforms is not None:
            self.image_transforms = get_transforms(image_transforms)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file_path = self.images[idx]

        # Read the file
        image = _load_file(file_path)

        # Apply transforms
        if self.image_transforms is not None:
            image = self.image_transforms(image)

        return image, image
