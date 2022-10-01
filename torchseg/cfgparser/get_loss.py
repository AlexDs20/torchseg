import copy
import torch.nn as nn
from torchseg.losses import FocalLoss, DiceLoss


def get_loss(config):
    conv = {
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
            attr = getattr(nn, key) if hasattr(nn, key) else conv[key]
            func = attr(**val) if val is not None else attr()
            all_losses.append([weight, func])
        return all_losses
    except ValueError:
        print(f'Error with the input loss function(s) {config}')
