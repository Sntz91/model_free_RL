import torch
import torch.nn as nn
import numpy as np

class DuelingDQN(nn.Module):
    def __init__(self, n_in, n_out):
        super(DuelingDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(n_in[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(n_in)

        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_out)
        )

        self.conv.apply(self.init_weights)
        self.value_stream.apply(self.init_weights)
        self.advantage_stream.apply(self.init_weights)

        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor #here

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.type(self.dtype) #!!! not sure. I need this, because the images are now stored as uint8 and not float32 anymore
        conv_out = self.conv(x).view(x.size()[0], -1)
        values = self.value_stream(conv_out)
        advantages = self.advantage_stream(conv_out)
        q_vals = values + (advantages - advantages.mean())

        return q_vals

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)