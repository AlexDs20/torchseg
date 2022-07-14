from pytorch_lightning import loggers


def get_loggers(config):
    conv = {
        'CometLogger': loggers.CometLogger,
        'CSVLogger': loggers.CSVLogger,
        'MLFlowLogger': loggers.MLFlowLogger,
        'NeptuneLogger': loggers.NeptuneLogger,
        'TensorBoardLogger': loggers.TensorBoardLogger,
        'WandbLogger': loggers.WandbLogger,
    }
    if config is None:
        return loggers.MLFlowLogger()
    elif config is False:
        return False
    else:
        logs = []
        for key, val in config.items():
            func = conv[key](**val) if val is not None else conv[key]()
            logs.append(func)
        return logs
