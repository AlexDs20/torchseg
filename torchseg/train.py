import yaml

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from torchseg.dataset import FolderDataSet
from torchseg.cfgparser import get_callbacks, get_loggers
from torchseg.transfer_learning import transfer_learning
from .plModel import plModel


def train(cfg_file):
    with open(cfg_file) as cfg:
        config = yaml.load(cfg, Loader=yaml.Loader)

    train_dataloader = DataLoader(FolderDataSet(config['data']['train_folder']), **config['dataloader']['train'])
    valid_dataloader = DataLoader(FolderDataSet(config['data']['valid_folder']), **config['dataloader']['valid'])

    callbacks = get_callbacks(config['callbacks'])
    loggers = get_loggers(config['loggers'])

    model = plModel(config)

    trainer = pl.Trainer(callbacks=callbacks, logger=loggers, **config['trainer'])

    if config['transfer_learning'] is not None:
        model = transfer_learning(model, config['transfer_learning'], callbacks, loggers, train_dataloader, valid_dataloader)

    trainer.fit(model, train_dataloader, valid_dataloader, ckpt_path=config['resume_from_ckpt'])
