import torch
import torch.nn as nn
import torch.nn.functional as F
from base_module import *


class SD_UNet(nn.Module):
    def __init__(self, n_head=8, channels=[320, 640, 1280]):
        super().__init__()
        self.encoders = nn.ModuleList(
            [
                SwitchSequential(nn.Conv2d(4, channels[0], kernel_size=3, padding=1)),
                SwitchSequential(
                    TimeResidualBlock(channels[0], channels[0]),
                    MultiModalAttentionBlock(n_head, channels[0]),
                ),
                SwitchSequential(
                    TimeResidualBlock(channels[0], channels[0]),
                    MultiModalAttentionBlock(n_head, channels[0]),
                ),
                SwitchSequential(
                    nn.Conv2d(
                        channels[0], channels[0], kernel_size=3, stride=2, padding=1
                    )
                ),
                SwitchSequential(
                    TimeResidualBlock(channels[0], channels[1]),
                    MultiModalAttentionBlock(n_head, channels[1]),
                ),
                SwitchSequential(
                    TimeResidualBlock(channels[1], channels[1]),
                    MultiModalAttentionBlock(n_head, channels[1]),
                ),
                SwitchSequential(
                    nn.Conv2d(
                        channels[1], channels[1], kernel_size=3, stride=2, padding=1
                    )
                ),
                SwitchSequential(
                    TimeResidualBlock(channels[1], channels[2]),
                    MultiModalAttentionBlock(n_head, channels[2]),
                ),
                SwitchSequential(
                    TimeResidualBlock(channels[2], channels[2]),
                    MultiModalAttentionBlock(n_head, channels[2]),
                ),
                SwitchSequential(
                    nn.Conv2d(
                        channels[2], channels[2], kernel_size=3, stride=2, padding=1
                    )
                ),
                SwitchSequential(TimeResidualBlock(channels[2], channels[2])),
                SwitchSequential(TimeResidualBlock(channels[2], channels[2])),
            ]
        )
        self.bottleneck = SwitchSequential(
            TimeResidualBlock(channels[2], channels[2]),
            MultiModalAttentionBlock(n_head, channels[2]),
            TimeResidualBlock(channels[2], channels[2]),
        )
        self.decoders = nn.ModuleList(
            [
                SwitchSequential(TimeResidualBlock(channels[2] * 2, channels[2])),
                SwitchSequential(TimeResidualBlock(channels[2] * 2, channels[2])),
                SwitchSequential(
                    TimeResidualBlock(channels[2] * 2, channels[2]),
                    Upsample(channels[2]),
                ),
                SwitchSequential(
                    TimeResidualBlock(channels[2] * 2, channels[2]),
                    MultiModalAttentionBlock(n_head, channels[2]),
                ),
                SwitchSequential(
                    TimeResidualBlock(channels[2] * 2, channels[2]),
                    MultiModalAttentionBlock(n_head, channels[2]),
                ),
                SwitchSequential(
                    TimeResidualBlock(channels[2] + channels[1], channels[2]),
                    MultiModalAttentionBlock(n_head, channels[2]),
                    Upsample(channels[2]),
                ),
                SwitchSequential(
                    TimeResidualBlock(channels[2] + channels[1], channels[1]),
                    MultiModalAttentionBlock(n_head, channels[1]),
                ),
                SwitchSequential(
                    TimeResidualBlock(channels[2], channels[1]),
                    MultiModalAttentionBlock(n_head, channels[1]),
                ),
                SwitchSequential(
                    TimeResidualBlock(channels[0] + channels[1], channels[1]),
                    MultiModalAttentionBlock(n_head, channels[1]),
                    Upsample(channels[1]),
                ),
                SwitchSequential(
                    TimeResidualBlock(channels[0] + channels[1], channels[0]),
                    MultiModalAttentionBlock(n_head, channels[0]),
                ),
                SwitchSequential(
                    TimeResidualBlock(channels[1], channels[0]),
                    MultiModalAttentionBlock(n_head, channels[0]),
                ),
                SwitchSequential(
                    TimeResidualBlock(channels[1], channels[0]),
                    MultiModalAttentionBlock(n_head, channels[0]),
                ),
            ]
        )

    def forward(self, x, context, time):
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)

        return x


class Diffusion(nn.Module):
    def __init__(self, n_head=8, channels=[320, 640, 1280]):
        super().__init__()
        self.time_embedding = TimeEmbedding(channels[0])
        self.unet = SD_UNet(n_head, channels)
        self.final_layer = nn.Sequential(
            nn.GroupNorm(32, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], 4, kernel_size=3, padding=1),
        )

    def forward(self, x, context, time):
        time = self.time_embedding(time)
        x = self.unet(x, context, time)
        x = self.final_layer(x)
        return x
