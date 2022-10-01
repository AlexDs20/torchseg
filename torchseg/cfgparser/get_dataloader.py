from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchseg.dataset as dataset


def get_dataloader(config, stage):
    ds = get_dataset(config['data'][f'{stage}_folder'], config['dataset'][stage])
    dl = DataLoader(ds, **config['dataloader'][stage])
    return dl


def get_dataset(data_folder, config):
    conv = {}

    for key, val in config.items():
        img_tf = get_transforms(val['image_transforms'])
        lbl_tf = get_transforms(val['target_transforms'])
        cls = getattr(dataset, key) if hasattr(dataset, key) else conv[key]

        ds = cls(data_folder, img_tf, lbl_tf)
        return ds


def get_transforms(config):
    conv = {}
    transform_list = []

    for key, val in config.items():
        cls = getattr(transforms, key) if hasattr(transforms, key) else conv[key]
        if isinstance(val, dict):
            tf = cls(**val)
        else:
            tf = cls(val) if val is not None else cls()
        transform_list.append(tf)

    # TODO: Add support to run the transforms in random order
    return transforms.Compose(transform_list)
