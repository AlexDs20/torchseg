

class DataModule(pl.LightningDataModule):
    """
    https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """
    def __init__(self, batch_size=32):
        self.batch_size = batch_size

    def prepare_data(self):
        """
        Run single process on CPU
        """
        # Download, tokenize, ...
        pass

    def setup(self):
        """
        Operations on every GPU
        """
        # split, transform, ...

        #self.train =
        #self.val =
        #self.test =
        pass

    def train_dataloader(self):
        train = DataLoader(self.train, batch_size=self.batch_size)
        return train

    def val_dataloader(self):
        val = DataLoader(self.val, batch_size=self.batch_size)
        return val

    def test_dataloader(self):
        test = DataLoader(self.test, batch_size=self.batch_size)
        return test
