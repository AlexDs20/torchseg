from torch.utils.data import DataLoader
import torchseg.dataset as dataset


def get_dataloader(config, stage):
    ds = get_dataset(config['data'][f'{stage}_folder'], config['dataset'][stage])
    dl = DataLoader(ds, **config['dataloader'][stage])
    return dl


def get_dataset(data_folder, config):
    conv = {}

    for key, val in config.items():
        cls = getattr(dataset, key) if hasattr(dataset, key) else conv[key]

        ds = cls(data_folder, **val)
        return ds
