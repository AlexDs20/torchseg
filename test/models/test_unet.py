import pytest
import torch

from torchseg.models.Unet import DoubleConv, Encoder, Decoder, Unet

class TestUnet:
    def test_doubleconv(self):
        in_channels = 12
        out_channels = 64

        bs = 5
        w, h = 256, 256
        x = torch.ones((bs, in_channels, w, h))

        dc = DoubleConv(in_channels, out_channels)

        dc(x)
