import torchseg.metrics.functional as MF
from functools import partial


def get_metrics(config):
    '''
    DOCSTRING
    '''
    conv = {
        "Accuracy": MF.accuracy,
        "Precision": MF.precision,
        "Recall": MF.recall
    }
    all_metrics = {}

    if config is not None:
        for key, val in config.items():
            attr = getattr(MF, key) if hasattr(MF, key) else conv[key]
            func = partial(attr, **val) if val is not None else attr
            all_metrics.update({key: func})
    return all_metrics
