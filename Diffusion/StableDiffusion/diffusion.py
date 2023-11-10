import torch
import torch.nn as nn
import torch.nn.functional as F
from base_module import *


class TimeEmbedding(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.l1 = nn.Linear(n_embed, 4 * n_embed)
        self.l2 = nn.Linear(4 * n_embed, 4 * n_embed)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        return x


class TimeResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.time_embedding = nn.Linear(n_time, out_channels)

    def forward(self, x, time):
        residual = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        time = self.act(time)
        time = self.time_embedding(time)
        x = x + time.unsqueeze(-1).unsqueeze(-1)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x = x + self.residual_layer(residual)
        return x


class MultiModalAttentionBlock(nn.Module):
    def __init__(self, n_head, n_embed, d_context=512):
        super().__init__()
        self.gn = nn.GroupNorm(32, n_embed)
        self.conv_in = nn.Conv2d(n_embed, n_embed, kernel_size=1)
        self.conv_out = nn.Conv2d(n_embed, n_embed, kernel_size=1)

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ln3 = nn.LayerNorm(n_embed)

        self.att1 = SelfAttention(n_head, n_embed, in_proj_bias=False)
        self.att2 = CrossAttention(n_head, n_embed, d_context, in_proj_bias=False)

        self.l1 = nn.Linear(n_embed, 4 * n_embed * 2)
        self.l2 = nn.Linear(4 * n_embed, n_embed)

        self.act = nn.GELU()

    def forward(self, x, context):
        residual = x
        x = self.gn(x)
        x = self.conv_in(x)

        n, c, h, w = x.shape
        x = x.view(n, c, -1).transpose(1, 2)

        residual_t = x
        x = self.ln1(x)
        x = self.att1(x)
        x += residual_t

        residual_t = x
        x = self.ln2(x)
        x = self.att2(x, context)
        x += residual_t

        residual_t = x
        x = self.ln3(x)
        x, gate = self.l1(x).chunk(2, dim=-1)
        x = x * self.act(gate)
        x = self.l2(x)
        x += residual_t

        x = x.transpose(1, 2).view(n, c, h, w)

        x = self.conv_out(x) + residual
        return x


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        x = self.act(x)
        return x


# 这一段可以模仿，没有见过这种写法
class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, MultiModalAttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, TimeResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNet(nn.Module):
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
        self.unet = UNet(n_head, channels)
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
