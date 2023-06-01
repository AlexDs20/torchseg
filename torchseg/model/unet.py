from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 padding_mode='reflect',
                 residual=False,
                 batch_norm=False,
                 **kwargs):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.residual = residual
        self.batch_norm = batch_norm

        self.conv1 = nn.Conv2d(in_channels,
                               mid_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               padding_mode=padding_mode,
                               **kwargs)
        self.conv2 = nn.Conv2d(mid_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               padding_mode=padding_mode,
                               **kwargs)

        if self.residual:
            self.res = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        if self.batch_norm:
            out = self.bn1(out)
        out = F.relu(self.conv2(out))
        if self.batch_norm:
            out = self.bn2(out)
        if self.residual:
            out = out + self.res(x)
        return out


class Encoder(nn.Module):
    def __init__(self, parameters):
        super(Encoder, self).__init__()

        self.block = nn.ModuleDict({})
        self.down = nn.ModuleDict({})

        for i, (key, val) in enumerate(parameters.items()):
            if val['name'] == 'DoubleConv':
                v = deepcopy(val)
                del v['name']
                self.block[key] = DoubleConv(**v)

            if i != len(parameters):
                self.down[key] = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = {}
        out = x
        for key, val in self.block.items():
            out = self.block[key](out)
            features[key] = out
            if key in self.down:
                out = self.down[key](out)
        return features


class Empty(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 0, *x.shape[-2:], device=x.device)


class Middle(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.middle = nn.ModuleDict({})

        for key, val in params.items():
            v = deepcopy(val)
            name = v.pop('name')
            if name == 'skip':
                self.middle[key] = nn.Identity()
            else:
                self.middle[key] = Empty()

    def forward(self, features):
        out = features
        for key, val in self.middle.items():
            out[key] = val(features[key])

        return out


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()

        self.block = nn.ModuleDict({})
        self.up = nn.ModuleDict({})

        for key, val in params.items():
            if 'up' in val:
                v = deepcopy(val['up'])
                name = v.pop('name')
                if name == 'ConvTranspose2d':
                    self.up[key] = nn.ConvTranspose2d(**v, kernel_size=2, stride=2)

            if 'block' in val:
                v = deepcopy(val['block'])
                name = v.pop('name')
                if name == 'DoubleConv':
                    self.block[key] = DoubleConv(**v)

    def forward(self, features):
        out = features[sorted(features.keys())[-1]]

        for key in sorted(features.keys(), reverse=True):
            if key in self.block:
                out = torch.cat([features[key], out], dim=1)
                out = self.block[key](out)
            if key in self.up:
                out = self.up[key](out)

        return out


class Head(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.head = nn.Conv2d(**params, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, features):
        out = self.head(features)
        return out


class Unet(nn.Module):
    def __init__(self, parameters):
        super(Unet, self).__init__()

        self.encoder = Encoder(parameters['encoder'])
        self.decoder = Decoder(parameters)

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out
