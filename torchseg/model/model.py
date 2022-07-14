import torch.nn as nn
from .unet import Encoder, Decoder, Head


class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()

        self.encoder = Encoder(params['encoder'])
        self.decoder = Decoder(params['decoder'])
        self.head = Head(params['head'])

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        out = self.head(out)
        return out
