import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from simple_rnn import Classifier as Model
from dataset import train_dataset
from dataset import test_dataset as val_dataset


class plModel(pl.LightningModule):
    def __init__(self):
        super(plModel, self).__init__()

        self.model = Model()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.model(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y.to(torch.float))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y.to(torch.float))
        self.log('val_loss', loss)
        return loss


if __name__=='__main__':
    # https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction.html
    # https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09
    # https://pytorch-lightning.readthedocs.io/en/stable/common/early_stopping.html
    # https://pytorch-lightning.readthedocs.io/en/stable/common/loggers.html
    # https://pytorch-lightning.readthedocs.io/en/stable/advanced/training_tricks.html#batch-size-finder
    # https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#lightning-hooks
    # https://pytorch-lightning.readthedocs.io/en/latest/common/optimization.html#id5
    # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.callbacks.LearningRateMonitor.html#pytorch_lightning.callbacks.LearningRateMonitor
    # https://neptune.ai/blog/pytorch-loss-functions
    callbacks = []
    model = plModel()
    #dm    = DataModule()
    #trainer.fit(model, dm, callbacks=callbacks)
    trainer_params = {
            'gpus': 1,
            'max_epochs':2
            }

    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_dataloader   = DataLoader(val_dataset,   batch_size=1024, shuffle=False)

    trainer = pl.Trainer(**trainer_params)
    trainer.fit(model, train_dataloader, val_dataloader)
