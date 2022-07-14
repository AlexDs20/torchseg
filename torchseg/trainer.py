import sys
sys.path.append('.')

import yaml
import torch
import argparse
import pytorch_lightning as pl

from torch.utils.data import DataLoader
import torch.nn.functional as F

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
        self.train_loss = get_loss(config['loss']['train'])
        self.valid_loss = get_loss(config['loss']['valid'])
        self.metrics = {"train": get_metrics(config['metrics']['train']),
                        "valid": get_metrics(config['metrics']['valid']),
                        "test": None}

        self.optim = get_optimizer(config['optimizer'])
        self.lr_scheduler = get_lr_scheduler(config['lr_scheduler'])

        self.log_images = {'train': None,
                           'valid': None,
                           'test': None}

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


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        prob = self.logits_to_prob(logits)

        loss = 0
        for weight, loss_func in self.train_loss:
            loss += weight * loss_func(logits, y)

        # Logging to TensorBoard by default
        self.log("train/loss", loss)

        # Cache data for the log images callback
        if batch_idx==self.trainer.num_training_batches-2:
            self.log_images['train'] = [x.detach(), y.detach(), prob.detach()]

        #TODO  implement this as a torchmetrics metrics class with this link:
        # https://torchmetrics.readthedocs.io/en/stable/pages/implement.html

        mode = self.config['data']['processing']['mode']
        ignore_index = self.config['data']['processing']['ignore_index']
        classes = self.probs_to_classes(prob)
        tp, fp, fn, tn = MF.get_stats(classes, y, mode=mode, ignore_index=ignore_index)
        self._log_metrics([{'tp':tp, 'fp':fp, 'fn':fn, 'tn':tn}], 'train')

        return {'loss': loss, 'tp':tp, 'fp':fp, 'fn':fn, 'tn':tn}


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        prob = self.logits_to_prob(logits)

        loss = 0
        for weight, loss_func in self.valid_loss:
            loss += weight * loss_func(logits, y)

        # Logging to TensorBoard by default
        self.log("valid/loss", loss)

        # Cache data for the log images callback
        if batch_idx == self.trainer.num_val_batches[0]-2:
            self.log_images['valid'] = [x.detach(), y.detach(), prob.detach()]

        # Log metrics
        self.log("hp_metric", loss)

        mode = self.config['data']['processing']['mode']
        ignore_index = self.config['data']['processing']['ignore_index']
        classes = self.probs_to_classes(prob)
        tp, fp, fn, tn = MF.get_stats(classes, y, mode=mode, ignore_index=ignore_index)

        return {'valid/loss': loss, 'tp':tp, 'fp':fp, 'fn':fn, 'tn':tn}


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


    def logits_to_prob(self, logits):
        if self.config['data']['processing']['mode'] == 'multiclass':
            prob = F.log_softmax(logits.detach(), dim=1).exp()
        else:
            prob = F.logsigmoid(logits.detach()).exp()
        return prob

    def probs_to_classes(self, prob, threshold=0.5):
        if self.config['data']['processing']['mode'] == 'multiclass':
            classes = prob.argmax(dim=1)
        else:
            classes = torch.where(prob>threshold,1,0)
        return classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='path to the config.yaml file to be used', default='config.yaml')
    args = parser.parse_args()

    with open(args.config) as cfg:
        config = yaml.load(cfg, Loader=yaml.Loader)

    train_dataloader = DataLoader(FolderDataSet(config['data']['train_folder'], config['data']['processing']), **config['dataloader']['train'])
    valid_dataloader = DataLoader(FolderDataSet(config['data']['valid_folder'], config['data']['processing']), **config['dataloader']['valid'])

    callbacks = get_callbacks(config['callbacks'])
    loggers = get_loggers(config['loggers'])

    model = plModel(config)

    trainer = pl.Trainer(callbacks=callbacks, logger=loggers, **config['trainer'])

    if config['transfer_learning'] is not None:
        model = transfer_learning(model, config['transfer_learning'], callbacks, loggers, train_dataloader, valid_dataloader)

    trainer.fit(model, train_dataloader, valid_dataloader, ckpt_path=config['resume_from_ckpt'])
