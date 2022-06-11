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
    callbacks = []
    model = plModel()
    trainer_params = {
            'gpus': 1
            }

    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_dataloader   = DataLoader(val_dataset,   batch_size=512, shuffle=False)

    trainer = pl.Trainer(**trainer_params)
    trainer.fit(model, train_dataloader, val_dataloader)
