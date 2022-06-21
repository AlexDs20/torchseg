import yaml
import argparse

import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl

import torchmetrics as tm

from models.simple_rnn import Classifier as Model
from dataset import train_dataset
from dataset import test_dataset as val_dataset

from pprint import pprint as pp

"""
Some documentation to look at:

Intro:
-----
https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction.html

https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09

Early Stopping:
--------------
https://pytorch-lightning.readthedocs.io/en/stable/common/early_stopping.html

Loggers:
-------
https://pytorch-lightning.readthedocs.io/en/stable/common/loggers.html

Lightning Module + hooks:
----------------------------
https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#lightning-hooks

Optimization:
------------
https://pytorch-lightning.readthedocs.io/en/latest/common/optimization.html#id5

Callbacks:
---------
https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.callbacks.LearningRateMonitor.html#pytorch_lightning.callbacks.LearningRateMonitor

Loss functions:
--------------
https://neptune.ai/blog/pytorch-loss-functions

Extra:
-----
https://pytorch-lightning.readthedocs.io/en/stable/advanced/training_tricks.html#batch-size-finder

"""

class plModel(pl.LightningModule):
    def __init__(self):
        super(plModel, self).__init__()

        self.save_hyperparameters()

        self.model = Model()
        self.loss = nn.CrossEntropyLoss()
        self.metrics = tm.Accuracy().to(self.device)

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
        self.log(f'train_{self.metrics}', self.metrics(logits, y))
        return loss

    def validation_step(self, batch, idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y.to(torch.float))
        self.log('val_loss', loss)
        self.log(f'val_{self.metrics}', self.metrics(logits, y))
        self.log('hp_metric', loss)
        return loss


if __name__=='__main__':
    # Inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='config.yaml', type=str,
                        help='path to configuration yaml file')
    args = parser.parse_args()

    with open(args.config) as cfg:
        config = yaml.load(cfg, Loader=yaml.Loader)

    callbacks = []
    model = plModel()
    #dm    = DataModule()
    #trainer.fit(model, dm, callbacks=callbacks)

    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_dataloader   = DataLoader(val_dataset,   batch_size=1024, shuffle=False)

    trainer = pl.Trainer(**config['trainer'])
    trainer.fit(model, train_dataloader, val_dataloader)
