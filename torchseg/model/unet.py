import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 padding_mode='reflect',
                 residual=False,
                 batch_norm=False,
                 **kwargs):
        super(DoubleConv, self).__init__()

        self.residual = residual
        self.batch_norm = batch_norm

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               padding_mode=padding_mode,
                               **kwargs)
        self.conv2 = nn.Conv2d(out_channels,
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

        self.Conv = nn.ModuleList()
        self.MaxPool = nn.ModuleList()

        for i, (key, val) in enumerate(parameters.items()):
            self.Conv.append(DoubleConv(**val))
            if i != len(parameters):
                self.MaxPool.append(nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        features = []
        out = x
        for i in range(len(self.Conv)):
            out = self.Conv[i](out)
            if i < len(self.MaxPool):
                features.append(out)
                out = self.MaxPool[i](out)
        return features


class Decoder(nn.Module):
    def __init__(self, parameters):
        super(Decoder, self).__init__()

        encoder_params = parameters['encoder']
        decoder_params = parameters['decoder']

        encoder_out_channels = [layer['out_channels'] for layer in encoder_params.values()]

        self.Conv = nn.ModuleList()
        self.Up = nn.ModuleList()

        for i, (key, val) in enumerate(decoder_params.items()):
            in_channels = val['in_channels']
            out_channels = val['out_channels']

            self.Up.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            self.Conv.append(DoubleConv(in_channels=out_channels + encoder_out_channels[-i - 2],
                                        out_channels=out_channels))

    def forward(self, features):
        out = features.pop()
        for i in range(len(features)):
            out = self.Up[i](out)
            out = torch.cat([features[-i - 1], out], dim=1)
            out = self.Conv[i](out)

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
