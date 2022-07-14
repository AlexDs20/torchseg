from functools import partial
from torch import optim


def get_optimizer(config):
    conv = {
            'Adadelta': optim.Adadelta,
            'Adagrad': optim.Adagrad,
            'Adam': optim.Adam,
            'AdamW': optim.AdamW,
            'SparseAdam': optim.SparseAdam,
            'Adamax': optim.Adamax,
            'ASGD': optim.ASGD,
            'LBFGS': optim.LBFGS,
            'NAdam': optim.NAdam,
            'RAdam': optim.RAdam,
            'RMSprop': optim.RMSprop,
            'SGD': optim.SGD,
        }

    try:
        # Currently only support 1 entry
        for key, val in config.items():
            func = partial(conv[key], **val) if val is not None else conv[key]()
            return func
    except ValueError:
        print(f'Error with the optimizer: {config}')
