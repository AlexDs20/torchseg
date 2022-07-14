import torch
import pytorch_lightning as pl

def transfer_learning(model, config, callbacks, loggers, train_dataloader, valid_dataloader):
    """
    From a certain checkpoint: get the weights.
    From the new model arch.: get the weights.

    Loop through the checkpoint layers.
    If the layers name and shape matches the one for the new model arch -> use the weights from the checkpoints, otherwise keep as is.

    If max_epochs is set, then freeze all the layers that are already trained and train for max_epochs, then unfreeze everything.
    If max_epochs is set to None, then just return this model

    Documentation:
    -------------
    https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    """
    # Get current untrained layers
    model_layers = model.state_dict()

    # Get checkpoint layers
    checkpoint = torch.load(config['transfer_from_checkpoint'], map_location='cpu')
    checkpoint_layers = checkpoint['state_dict']

    # Keep layers with same name and shape
    keep_layers = {}
    for key, val in checkpoint_layers.items():
        if (key in model_layers) and val.shape == model_layers[key].shape:
            keep_layers[key] = val

    # Set the layers to keep in the model to train
    model_layers.update(keep_layers)
    model.load_state_dict(model_layers)

    # Train with frozen layers
    if 'trainer' in config and config['trainer']['max_epochs'] is not None and config['trainer']['max_epochs'] != 0:
        # Freeze layers
        for name, params in model.named_parameters():
            if name in keep_layers:
                params.requires_grad = False

        # start training with frozen layers
        trainer = pl.Trainer(callbacks=callbacks, logger=loggers, **config['trainer'])
        trainer.fit(model, train_dataloader, valid_dataloader)

        # Unfreeze layers
        for params in model.parameters():
            params.requires_grad = True

    return model
