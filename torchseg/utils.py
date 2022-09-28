import torch


def make_subtiles(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """ Reshape a 3D tensor into a 4D with the new dimension used to index the subtiles
    Inputs:
     x: tensor to split into subtiles of size h, w. Size of x: (C,H,W)
     h: int indicating the height of the subtile
     w: int indicating the width of the subtile to use

    Output:
     Torch tensor of shape (N,C,H,W) with N = the number of subtiles created.
    """
    c, m, n = x.shape

    if m // h != m / h:
        raise ValueError(f'Required subtile size of {h} for tensor size of {m}, does not create an exact number of subtiles. Use padding or change subtile size.')
    elif n // w != n / w:
        raise ValueError(f'Required subtile size of {w} for tensor size of {n}, does not create an exact number of subtiles. Use padding or change subtile size.')

    return x.reshape(c, m // h, h, n // w, w).permute(1, 3, 0, 2, 4).reshape(-1, c, h, w)


def unmake_subtiles(x: torch.Tensor, m: int, n: int) -> torch.Tensor:
    """ Reshape the subtiles created by make_subtiles into an tensor of the original size

    Inputs:
     x: torch tensor with 4D with structure (N,C,H,W)
     m: int of the origial height of the image (height to recreate)
     n: into f the original width of the image (width to recreate)
    """
    c, h, w = x.shape[-3:]

    if m // h != m / h:
        raise ValueError(f'Required larger torch of size {m} while subtiles are of size {h} this does not work.')
    elif n // w != n / w:
        raise ValueError(f'Required larger torch of size {n} while subtiles are of size {w} this does not work.')

    return x.reshape(m // h, n // w, c, h, w).permute(2, 0, 3, 1, 4).reshape(c, m, n)
