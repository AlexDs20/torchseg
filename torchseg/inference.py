import numpy as np
import torch
from torch.utils.data import DataLoader

from utilities.raster.operations import make_subtiles, unmake_subtiles

from torchseg.trainer import plModel


@torch.no_grad()
def inference(data: np.ndarray, config: dict, model: str, batch_size: int = None, cut_size: int = None, proba_output: bool = True, device: int = 'cpu') -> np.ndarray:
    """ General inference function

    Input:
    -----
    data: numpy array of the prepared data containing the right layers with the correct preprocessing for each layer, the image can be any size (C,H,W)
    config: dictionay configuration file that was used for training the model
    batch_size: the size of the batch to send through the model at once
    proba_output: Optional[default=True] to say whether to output probability (default) or classes (False)

    Output:
    ------
    numpy array containing the predictions. The array is of the same height and width but with a number of channel as dictated by the model output
    """
    # Device
    if device == 'gpu':
        # Use gpu if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # Check datatype
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)

    # Check data dim
    if data.dim() == 3:
        data = data.unsqueeze(0)
    elif data.dim() == 4:
        if cut_size is not None:
            raise ValueError('Batched inputs with cutsize is not implemented yet.')
        else:
            pass
    else:
        raise ValueError(f'Input data with {data.dim()} dimensions. Only 3 or 4 dimensions allowed.')

    data = data.to(device)

    # Load model
    model = plModel.load_from_checkpoint(model, config=config, device=device)
    model.eval()

    if cut_size is None:    # Predict on the whole image at once, this is OK as long as we have pure convolutional model
        logits = model(data, device=device)

    # If not pure convolutional model, we should use this, or better: smooth tile prediction (not implemented)
    else:
        # Here do some padding, for the moment, just doing minimal functional
        c, h, w = data.shape
        add_h = cut_size - (h - (h // cut_size) * cut_size)
        add_w = cut_size - (w - (w // cut_size) * cut_size)
        data = torch.nn.functional.pad(data, (0, add_w, 0, add_h, 0, 0), mode='reflect')

        # Split into subtiles
        subtiles = make_subtiles(data, cut_size, cut_size)

        # Run inference by batches
        logits = torch.zeros((subtiles.shape[0], config['model']['head']['out_channels'], cut_size, cut_size), device=device)

        for bi in range(0, subtiles.shape[0], batch_size):
            be = bi + batch_size if bi + batch_size < subtiles.shape[0] else subtiles.shape[0]
            logits[bi:be] = model(subtiles[bi:be])

        # Put back to original shape
        logits = unmake_subtiles(logits, data.shape[-2], data.shape[-1])

        # Remove padding
        logits = logits[:, :h, :w]

    # Get probabilities
    proba = model.logits_to_prob(logits)

    # Convert to classes if wanted
    if not proba_output:
        output = model.probs_to_classes(proba)

    return output.cpu().numpy()
