from pytorch_lightning import loggers


def get_loggers(config):
    conv = {}

    if config is None:
        return loggers.MLFlowLogger()
    elif config is False:
        return False
    else:
        logs = []
        for key, val in config.items():
            attr = getattr(loggers, key) if hasattr(loggers, key) else conv[key]
            func = attr(**val) if val is not None else attr()
            logs.append(func)
        return logs
