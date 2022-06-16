import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', residual=False, **kwargs):
        super(DoubleConv, self).__init__()

        self.residual = residual

        self.conv1 = nn.Conv2d(in_channels,  out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, **kwargs)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, **kwargs)

        if self.residual:
            self.res = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)


    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        if self.residual:
            out = out + self.res(x)
        return out


class Encoder(nn.Module):
    def __init__(self, parameters):
        super(Encoder, self).__init__()

        self.Conv = nn.ModuleList()
        self.MaxPool = nn.ModuleList()

        for i, (key, val) in enumerate(parameters.items()):
            self.Conv.append( DoubleConv(**val) )
            if i!=len(parameters):
                self.MaxPool.append( nn.MaxPool2d(kernel_size=2, stride=2) )


    def forward(self, x):
        features = []
        out = x
        for i in range(len(self.Conv)):
            out = self.Conv[i](out)
            if i<len(self.MaxPool):
                features.append(out)
                out = self.MaxPool[i](out)
        return features

class Decoder(nn.Module):
    def __init__(self, parameters):
        super(Decoder, self).__init__()

        encoder_params = parameters['Encoder']
        decoder_params = parameters['Decoder']

        encoder_out_channels = []
        for key, val in encoder_params.items():
            encoder_out_channels.append(val['out_channels'])


        self.Conv = nn.ModuleList()
        self.Up = nn.ModuleList()

        for i, (key, val) in enumerate(decoder_params.items()):
            in_channels = val['in_channels']
            out_channels = val['out_channels']

            self.Up.append( nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) )
            self.Conv.append( DoubleConv(in_channels=out_channels+encoder_out_channels[-i-2], out_channels=out_channels) )

    def forward(self, features):
        out = features.pop()
        for i in range(len(features)):
            out = self.Up[i](out)
            out = torch.cat([features[-i-1], out], dim=1)
            out = self.Conv[i](out)

        return out


class Unet(nn.Module):
    def __init__(self, parameters):
        super(Unet, self).__init__()

        self.encoder = Encoder(parameters['Encoder'])
        self.decoder = Decoder(parameters)

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out


if __name__=='__main__':
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    bs = 12
    features = 5
    w, h = 128, 128

    x = torch.ones((bs, features, w, h))

    if 0:
        out_channels = 64
        model = DoubleConv(features, out_channels)
        out = model(x)
        print(out.shape)

    if 0:
        parameters = {
            'layer1': {
                'in_channels': features,
                'out_channels': 64,
            },
            'layer2': {
                'in_channels': 64,
                'out_channels': 128,
            },
            'layer3': {
                'in_channels': 128,
                'out_channels': 256,
            },
            'layer4': {
                'in_channels': 256,
                'out_channels': 512,
            },
            'layer5': {
                'in_channels': 512,
                'out_channels': 1024,
            }
        }
        model = Encoder(parameters)
        out = model(x)
        print(out.shape)

    if 1:
        parameters = {
            'Encoder':{
                'layer1': {
                    'in_channels': features,
                    'out_channels': 64,
                    'residual': True,
                },
                'layer2': {
                    'in_channels': 64,
                    'out_channels': 128,
                    'residual': True,
                },
                'layer3': {
                    'in_channels': 128,
                    'out_channels': 256,
                    'residual': True,
                },
                'layer4': {
                    'in_channels': 256,
                    'out_channels': 512,
                    'residual': True,
                },
                'layer5': {
                    'in_channels': 512,
                    'out_channels': 1024,
                    'residual': True,
                }
            },
            'Decoder':{
                'layer4': {
                    'in_channels': 1024,
                    'out_channels': 512,
                    'residual': True,
                },
                'layer3': {
                    'in_channels': 512,
                    'out_channels': 256,
                    'residual': True,
                },
                'layer2': {
                    'in_channels': 256,
                    'out_channels': 128,
                    'residual': True,
                },
                'layer1': {
                    'in_channels': 128,
                    'out_channels': 64,
                    'residual': True,
                }
            }
        }
        model = Unet(parameters)
        #out = model(x)
        #print(out.shape)
        writer.add_graph(model, x)

    writer.close()

