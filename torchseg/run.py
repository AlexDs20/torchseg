import yaml
import argparse

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from torchseg.dataset import FolderDataSet
from torchseg.cfgparser import get_callbacks, get_loggers, get_dataloader
from torchseg.transfer_learning import transfer_learning
from torchseg.utils import parse_kwargs
from torchseg.plModel import plModel


def run(cfg_file):
    with open(cfg_file) as cfg:
        config = yaml.load(cfg, Loader=yaml.Loader)

    model = plModel(config)

    # parse the config to load from whatever library/module what is needed
    config = parse_kwargs(config)

    train_dataloader = get_dataloader(config, 'train')
    valid_dataloader = get_dataloader(config, 'valid')
    test_dataloader = get_dataloader(config, 'test')

    callbacks = get_callbacks(config['callbacks'])
    loggers = get_loggers(config['loggers'])

    trainer = pl.Trainer(callbacks=callbacks, logger=loggers, **config['trainer'])

    if config['transfer_learning'] is not None:
        model = transfer_learning(model, config['transfer_learning'], callbacks, loggers, train_dataloader, valid_dataloader)

    if config['run']['train']:
        trainer.fit(model, train_dataloader, valid_dataloader, ckpt_path=config['resume_from_ckpt'])
        ckpt_path = None
    else:
        ckpt_path = config['resume_from_ckpt']

    if config['run']['valid']:
        trainer.validate(model, valid_dataloader, ckpt_path=ckpt_path)

    if config['run']['test']:
        trainer.test(model, test_dataloader, ckpt_path=ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='path to the config.yaml file to be used', default='config.yaml')
    args = parser.parse_args()

    run(args.config)
