import torch.nn as nn
from .unet import Encoder, Middle, Decoder, Head


class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()

        self.encoder = Encoder(params['encoder'])
        self.middle = Middle(params['middle'])
        self.decoder = Decoder(params)
        self.head = Head(params['head'])

    def forward(self, x):
        features = self.encoder(x)
        features = self.middle(features)
        out = self.decoder(features)
        out = self.head(out)
        return out
