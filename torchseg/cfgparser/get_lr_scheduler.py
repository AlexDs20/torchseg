import sys
from functools import partial
from torch.optim import lr_scheduler

def get_lr_scheduler(config):
    conv = {
            'LambdaLR': lr_scheduler.LambdaLR,
            'MultiplicativeLR': lr_scheduler.MultiplicativeLR,
            'StepLR': lr_scheduler.StepLR,
            'MultiStepLR': lr_scheduler.MultiStepLR,
            'ConstantLR': lr_scheduler.ConstantLR,
            'LinearLR': lr_scheduler.LinearLR,
            'ExponentialLR': lr_scheduler.ExponentialLR,
            'CosineAnnealingLR': lr_scheduler.CosineAnnealingLR,
            'ChainedScheduler': lr_scheduler.ChainedScheduler,
            'SequentialLR:': lr_scheduler.SequentialLR,
            'ReduceLROnPlateau': lr_scheduler.ReduceLROnPlateau,
            'CyclicLR': lr_scheduler.CyclicLR,
            'OneCycleLR': lr_scheduler.OneCycleLR,
            'CosineAnnealingWarmRestarts': lr_scheduler.CosineAnnealingWarmRestarts,
        }

    if config is not None:
        for key, val in config.items():
            func = partial(conv[key], **val) if val is not None else conv[key]
            return func
    else:
        return None

