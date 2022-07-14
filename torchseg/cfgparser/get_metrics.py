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
    try:
        if config is not None:

            for key, val in config.items():
                func = partial(conv[key], **val) if val is not None else conv[key]
                all_metrics.update({key:func})
        return all_metrics
    except:
        print(f'Error with the input metric(s) {config}')

