import torch
import importlib


def make_subtiles(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """ Reshape a 3D or 4D tensor into a new 4D with the dimension 0 used to index the subtiles (and batch)
    Inputs:
     x: tensor to split into subtiles of size h, w. Size of x: (C,H,W) or (N,C,H,W)
     h: int indicating the height of the subtile
     w: int indicating the width of the subtile to use

    Output:
     Torch tensor of shape (N,C,H,W) with N = the total number of subtiles created.
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    elif x.dim() != 4:
        raise ValueError(f'Input dimension must be either 3 or 4. {x.dim} dimensions provided')

    b, c, m, n = x.shape

    if m // h != m / h:
        raise ValueError(f'Required subtile size of {h} for tensor size of {m}, does not create an exact number of subtiles. Use padding or change subtile size.')
    elif n // w != n / w:
        raise ValueError(f'Required subtile size of {w} for tensor size of {n}, does not create an exact number of subtiles. Use padding or change subtile size.')

    return x.reshape(b, c, m // h, h, n // w, w).permute(2, 4, 0, 1, 3, 5).reshape(-1, c, h, w)


def unmake_subtiles(x: torch.Tensor, m: int, n: int) -> torch.Tensor:
    """ Reshape the subtiles created by make_subtiles into a tensor of the original size

    Inputs:
     x: torch tensor with 4D with structure (N,C,H,W)
     m: int of the origial height of the image (height to recreate)
     n: into f the original width of the image (width to recreate)
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    elif x.dim() != 4:
        raise ValueError(f'Input dimension must be either 3 or 4. {x.dim} dimensions provided')

    c, h, w = x.shape[-3:]
    b = x.shape[0] // ((m // h) * (n // w))

    if m // h != m / h:
        raise ValueError(f'Required larger torch of size {m} while subtiles are of size {h} this does not work.')
    elif n // w != n / w:
        raise ValueError(f'Required larger torch of size {n} while subtiles are of size {w} this does not work.')

    return x.reshape(m // h, n // w, b, c, h, w).permute(2, 3, 0, 4, 1, 5).reshape(b, c, m, n)


def parse_kwargs(kwargs):
    for key, val in kwargs.items():
        if isinstance(val, dict):
            val = parse_kwargs(val)
        if isinstance(val, str):
            exist = importlib.util.find_spec('.'.join(val.split('.')[:-1])) is not None
            if exist:
                mymodule = importlib.import_module('.'.join(val.split('.')[:-1]))
                if hasattr(mymodule, val.split('.')[-1]):
                    kwargs[key] = getattr(mymodule, val.split('.')[-1])
    return kwargs
