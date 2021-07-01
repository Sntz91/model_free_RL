import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    """
    Neural Network, choosing actions
    """
    def __init__(self, n_in, n_out):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_in[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(n_in)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_out)
        )
        self.conv.apply(self.init_weights)
        self.fc.apply(self.init_weights)

        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor #here

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.type(self.dtype) 
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class DQN_cartpole(nn.Module):
    """
    Neural Network, choosing actions without conv
    """
    def __init__(self, n_in, n_out):
        super(DQN_cartpole, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(n_in[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_out)
        )


    def forward(self, x):
        return self.fc(x)