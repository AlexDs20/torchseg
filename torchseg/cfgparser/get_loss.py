import copy
import torch.nn as nn
from torchseg.losses import FocalLoss
from torchseg.losses import DiceLoss


def get_loss(config):
    conv = {
        'CrossEntropyLoss': nn.CrossEntropyLoss,
        'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,
        'FocalLoss': FocalLoss,
        'DiceLoss': DiceLoss
    }
    try:
        config_copy = copy.deepcopy(config)

        all_losses = []
        for key, val in config_copy.items():
            if 'weight' in val:
                weight = val.pop('weight')
            else:
                weight = 1
            func = conv[key](**val) if val is not None else conv[key]()
            all_losses.append([weight, func])
        return all_losses
    except ValueError:
        print(f'Error with the input loss function(s) {config}')
