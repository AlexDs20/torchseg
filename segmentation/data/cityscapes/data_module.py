import numpy as np

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, directory='data/VOC/', train=True):
        self.train = train

        x_train, y_train, x_test, y_test = data(directory)

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


if __name__=='__main__':
    from torchvision.datasets import VOCSegmentation
    dataset = VOCSegmentation('.', download=False)

    img, lab = dataset[0]
    img, lab = np.moveaxis(np.array(img),2,0), np.array(lab)
    print(img.shape, lab.shape)
