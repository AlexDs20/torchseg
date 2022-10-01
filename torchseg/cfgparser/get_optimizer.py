from functools import partial
from torch import optim


def get_optimizer(config):
    conv = {}

    try:
        # Currently only support 1 entry
        for key, val in config.items():
            attr = getattr(optim, key) if hasattr(optim, key) else conv[key]
            func = partial(attr, **val) if val is not None else attr()
            return func
    except ValueError:
        print(f'Error with the optimizer: {config}')
