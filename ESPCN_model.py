import torch
import torch.nn as nn


class ESPCN(nn.Module):
    def __init__(self, upscale_factor):
        super(ESPCN, self).__init__()
        self.ESPCN_layer = nn.Sequential(
            nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x):
        x = self.ESPCN_layer(x)
        return x
