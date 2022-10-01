from pytorch_lightning import callbacks
from torchseg.callbacks.ImageLogger import ImageLogger


def get_callbacks(config):
    conv = {
        'ModelCheckpoint': callbacks.ModelCheckpoint,
        'LearningRateMonitor': callbacks.LearningRateMonitor,
        'EarlyStopping': callbacks.EarlyStopping,
        'DeviceStatsMonitor': callbacks.DeviceStatsMonitor,
        'ImageLogger': ImageLogger,
    }

    cb = []
    try:
        if config is not None:
            for key, val in config.items():
                func = conv[key](**val) if val is not None else conv[key]()
                cb.append(func)
            return cb
    except ValueError:
        print(f'Error with the input callback(s) {config}')
