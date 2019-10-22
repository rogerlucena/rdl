from collections import deque

import torch
from torch import nn
from torch.nn import SmoothL1Loss


class NN(nn.Module):
    def __init__(self, inSize, outSize, layers=[]):
        super(NN, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
            self.layers.append(nn.Linear(inSize, outSize))

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.leaky_relu(x)
            x = self.layers[i](x)
        return x


class DQN():
    def __init__(self, capacity):
        self.D = deque(maxlen=capacity)


# ls0m la, ls1, l1, ld = map(list, zip(*p))
