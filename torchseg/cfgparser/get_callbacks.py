from pytorch_lightning import callbacks
from torchseg.callbacks.ImageLogger import ImageLogger


def get_callbacks(config):
    conv = {
        'ImageLogger': ImageLogger,
    }

    cb = []
    try:
        if config is not None:
            for key, val in config.items():
                attr = getattr(callbacks, key) if hasattr(callbacks, key) else conv[key]
                func = attr(**val) if val is not None else attr()
                cb.append(func)
            return cb
    except ValueError:
        print(f'Error with the input callback(s) {config}')
