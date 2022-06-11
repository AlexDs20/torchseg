import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SimplestRNN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(SimplestRNN, self).__init__()
        self.in_features = in_features
        self.hiden_features = hidden_features

        self.i2h = nn.Linear(in_features, hidden_features)
        self.h2h = nn.Linear(hidden_features, hidden_features)


    def forward(self, x):
        """
        x shape: [bs, T, N]
        """
        out = self._init_hidden()
        out = F.gelu(self.i2h(x) + self.h2h(hidden))
        return out

    def _init_hidden(self):
        return torch.zeros((self.hidden_features))


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.conv = nn.ModuleList(
                (
                nn.Conv2d(1, 64, kernel_size=3, stride=1, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(128*7**2, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
                )
                )

    def forward(self, x):
        out = x
        for layer in self.conv:
            out = layer(out)
        return out


if __name__ == '__main__':
    image = torch.ones((5,1,28,28))
    model = Classifier(1)
    model(image)

    bs = 5
    time_steps = 10
    features = 13
    hidden_features = 10

    x = torch.ones((bs, time_steps, features))

    model = SimplestRNN(features, hidden_features)

    for image in x:
        for i, step in enumerate(image):
            if i == 0:
                hidden = model._init_hidden(hidden_features)
            hidden = model(step, hidden)

    print(hidden)
