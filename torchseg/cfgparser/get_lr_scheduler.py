from functools import partial
from torch.optim import lr_scheduler


def get_lr_scheduler(config):
    conv = {}

    if config is not None:
        for key, val in config.items():
            attr = getattr(lr_scheduler, key) if hasattr(lr_scheduler, key) else conv[key]
            func = partial(attr, **val) if val is not None else attr
            return func
    else:
        return None
