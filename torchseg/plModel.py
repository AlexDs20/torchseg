import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchseg.losses import MULTICLASS_MODE
from torchseg.model import Model
from torchseg.dataset import FolderDataSet
from torchseg.transfer_learning import transfer_learning
import torchseg.metrics.functional as MF

from torchseg.cfgparser import get_metrics, get_callbacks, get_loss, get_optimizer, get_lr_scheduler, get_loggers


# Defining LightningModule
class plModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.model = Model(config['model'])
        self.loss = {"train": get_loss(config['loss']['train']),
                     "valid": get_loss(config['loss']['valid']),
                     "test": get_loss(config['loss']['valid'])}
        self.metrics = {"train": get_metrics(config['metrics']['train']),
                        "valid": get_metrics(config['metrics']['valid']),
                        "test": get_metrics(config['metrics']['valid'])}

        self.optim = get_optimizer(config['optimizer'])
        self.lr_scheduler = get_lr_scheduler(config['lr_scheduler'])

        self.log_images = {"train": None,
                           "valid": None,
                           "test": None}

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

        # Compute statistics for metrics
        mode = self.config['data']['processing']['mode']
        ignore_index = self.config['data']['processing']['ignore_index']

        # TODO  implement this as a torchmetrics metrics class with this link:
        # https://torchmetrics.readthedocs.io/en/stable/pages/implement.html

        prob = self.logits_to_prob(logits)
        classes = self.probs_to_classes(prob)
        tp, fp, fn, tn = MF.get_stats(classes, y, mode=mode, ignore_index=ignore_index)

        # Save images for logging
        if batch_idx == log_batch_idx:
            self.log_images[f'{stage}'] = [x, y, prob]

        if stage == 'valid':
            # Log hp metric
            self.log("hp_metric", loss)
            loss_name = 'valid/loss'
        elif stage == 'train':
            loss_name = 'loss'
        elif stage == 'test':
            loss_name = 'test/loss'

        return {loss_name: loss, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

    def training_step(self, batch, batch_idx):
        log_batch_idx = self.trainer.num_training_batches - 2
        return self._process_step(batch, batch_idx, 'train', log_batch_idx)

    def validation_step(self, batch, batch_idx):
        log_batch_idx = self.trainer.num_val_batches[0] - 2
        return self._process_step(batch, batch_idx, 'valid', log_batch_idx)

    def test_step(self, batch, batch_idx):
        log_batch_idx = self.trainer.num_val_batches[0] - 2
        return self._process_step(batch, batch_idx, 'test', log_batch_idx)

    def _log_metrics(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        for name, metric in self.metrics[stage].items():
            value = metric(tp, fp, fn, tn)
            self.log(f'{stage}/{name}', value)

    def training_epoch_end(self, outputs):
        return self._log_metrics(outputs, 'train')

    def validation_epoch_end(self, outputs):
        return self._log_metrics(outputs, 'valid')

    def test_epoch_end(self, outputs):
        return self._log_metrics(outputs, 'test')

    @torch.no_grad()
    def logits_to_prob(self, logits):
        if self.config['data']['processing']['mode'] == MULTICLASS_MODE:
            prob = F.log_softmax(logits.detach(), dim=1).exp()
        else:
            prob = F.logsigmoid(logits.detach()).exp()
        return prob

    @torch.no_grad()
    def probs_to_classes(self, prob, threshold=0.5):
        if self.config['data']['processing']['mode'] == MULTICLASS_MODE:
            classes = prob.argmax(dim=1)
        else:
            classes = torch.where(prob > threshold, 1, 0)
        return classes
