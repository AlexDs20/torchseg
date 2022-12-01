import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchseg.losses import MULTICLASS_MODE, BINARY_MODE, MULTILABEL_MODE
from torchseg.model import Model
from torchseg.dataset import FolderDataSet
from torchseg.transfer_learning import transfer_learning
import torchseg.metrics.functional as MF

from torchseg.cfgparser import get_metrics, get_callbacks, get_loss, get_optimizer, get_lr_scheduler, get_loggers

REGRESSION_MODE = 'regression'


# Defining LightningModule
class plModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.model = Model(config['model'])
        self.loss = {"train": get_loss(config['loss']['train']),
                     "valid": get_loss(config['loss']['valid']),
                     "test": get_loss(config['loss']['valid'])
                     }

        self.metrics = {"train": get_metrics(config['metrics']['train']),
                        "valid": get_metrics(config['metrics']['valid']),
                        "test": get_metrics(config['metrics']['valid'])
                        }

        self.optim = get_optimizer(config['optimizer'])
        self.lr_scheduler = get_lr_scheduler(config['lr_scheduler'])

        self.log_data = {"train": None,
                         "valid": None,
                         "test": None}

        self.mode = config['data']['mode']

    def forward(self, x):
        out = self.model(x)
        return out

    def configure_optimizers(self):
        optimizer = self.optim(self.model.parameters())
        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(optimizer)
            return [{"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": 'valid/loss'}}]
        else:
            return optimizer

    def _process_step(self, batch, batch_idx, stage, log_batch_idx=None):
        # Forward pass
        x, y = batch
        logits = self.model(x)

        # Get loss
        loss = 0
        for weight, loss_func in self.loss[stage]:
            loss += weight * loss_func(logits, y)

        # Logging to TensorBoard by default
        self.log(f"{stage}/loss", loss)

        # Log metrics
        if self.mode in [BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE]:
            predicted_value = self.logits_to_prob(logits)
        elif self.mode == REGRESSION_MODE:
            predicted_value = logits

        for name, metric in self.metrics[stage].items():
            self.log(f"{stage}/{name}_step", metric(predicted_value, y))

        # Save images for logging
        if batch_idx == log_batch_idx:
            self.log_data[stage] = [x, y, predicted_value]

        if stage == 'valid':
            # Log hp metric
            self.log("hp_metric", loss)
        return {'loss': loss}

    def training_step(self, batch, batch_idx):
        log_batch_idx = self.trainer.num_training_batches - 2
        return self._process_step(batch, batch_idx, 'train', log_batch_idx)

    def validation_step(self, batch, batch_idx):
        log_batch_idx = self.trainer.num_val_batches[0] - 2
        return self._process_step(batch, batch_idx, 'valid', log_batch_idx)

    def test_step(self, batch, batch_idx):
        log_batch_idx = self.trainer.num_val_batches[0] - 2
        return self._process_step(batch, batch_idx, 'test', log_batch_idx)

    def training_epoch_end(self, outputs):
        for name, metric in self.metrics["train"].items():
            self.log(f'train/{name}_epoch', metric.compute())

    def validation_epoch_end(self, outputs):
        for name, metric in self.metrics["valid"].items():
            self.log(f'valid/{name}_epoch', metric.compute())

    def test_epoch_end(self, outputs):
        for name, metric in self.metrics["test"].items():
            self.log(f'test/{name}_epoch', metric.compute())

    @torch.no_grad()
    def logits_to_prob(self, logits):
        if self.mode == MULTICLASS_MODE:
            prob = F.log_softmax(logits.detach(), dim=1).exp()
        elif self.mode in [BINARY_MODE, MULTILABEL_MODE]:
            prob = F.logsigmoid(logits.detach()).exp()
        return prob

    @torch.no_grad()
    def probs_to_classes(self, prob, threshold=0.5):
        if self.mode == MULTICLASS_MODE:
            classes = prob.argmax(dim=1)
        elif self.mode in [BINARY_MODE, MULTILABEL_MODE]:
            classes = torch.where(prob > threshold, 1, 0)
        return classes
