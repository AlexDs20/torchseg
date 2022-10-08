import torch
from functools import partial
import torchmetrics
from torchseg import metrics


def get_metrics(config):
    '''
    DOCSTRING
    '''
    conv = {
        "FBetaScore": metrics.FBetaScore,
        "F1Score": metrics.F1Score,
        "IoUScore": metrics.IoUScore,
        "Accuracy": metrics.Accuracy,
        "Precision": metrics.Precision,
        "Recall": metrics.Recall,
        "Sensitivity": metrics.Sensitivity,
        "Specificity": metrics.Specificity,
        "BalancedAccuracy": metrics.BalancedAccuracy
    }
    all_metrics = {}

    if config is not None:
        for key, val in config.items():
            attr = conv[key] if key in conv else getattr(torchmetrics, key)
            func = attr(**val) if val is not None else attr
            all_metrics.update({key: func})
    return all_metrics
