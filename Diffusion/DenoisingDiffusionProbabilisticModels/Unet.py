from base_module import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPM_Unet(nn.Module):
    def __init__(self, in_channels=3, channels=[128, 256, 512]):
        super().__init__()
        self.encoder1 = SwitchSequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1),
            TimeResidualBlock(channels[0], channels[0]),
            # AttentionBlock(channels[0]),
            TimeResidualBlock(channels[0], channels[0]),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=2, padding=1),
        )
        self.encoder2 = SwitchSequential(
            TimeResidualBlock(channels[0], channels[1]),
            # AttentionBlock(channels[1]),
            TimeResidualBlock(channels[1], channels[1]),
            nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=2, padding=1),
        )
        self.encoder3 = SwitchSequential(
            TimeResidualBlock(channels[1], channels[2]),
            AttentionBlock(channels[2]),
            TimeResidualBlock(channels[2], channels[2]),
            nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=2, padding=1),
        )
        self.bottleneck = SwitchSequential(
            TimeResidualBlock(channels[2], channels[2]),
            AttentionBlock(channels[2]),
            TimeResidualBlock(channels[2], channels[2]),
        )

        self.decoder1 = SwitchSequential(
            TimeResidualBlock(channels[2] * 2, channels[2]),
            AttentionBlock(channels[2]),
            TimeResidualBlock(channels[2], channels[2]),
            nn.ConvTranspose2d(
                channels[2], channels[2], kernel_size=4, stride=2, padding=1
            ),
        )
        self.decoder2 = SwitchSequential(
            TimeResidualBlock(channels[1] + channels[2], channels[1]),
            # AttentionBlock(channels[1]),
            TimeResidualBlock(channels[1], channels[1]),
            nn.ConvTranspose2d(
                channels[1], channels[1], kernel_size=4, stride=2, padding=1
            ),
        )
        self.decoder3 = SwitchSequential(
            TimeResidualBlock(channels[1] + channels[0], channels[0]),
            # AttentionBlock(channels[0]),
            TimeResidualBlock(channels[0], channels[0]),
            nn.ConvTranspose2d(
                channels[0], channels[0], kernel_size=4, stride=2, padding=1
            ),
        )
        self.output = SwitchSequential(
            TimeResidualBlock(channels[0], channels[0]),
            # AttentionBlock(channels[0]),
            TimeResidualBlock(channels[0], channels[0]),
            nn.Conv2d(channels[0], in_channels, kernel_size=3, padding=1),
        )
        self.time_embedding = nn.Embedding(5001, 1280)

    def forward(self, x, t):
        t = self.time_embedding(t).view(-1, 1280)
        x1 = self.encoder1(x, t)
        x2 = self.encoder2(x1, t)
        x3 = self.encoder3(x2, t)
        x4 = self.bottleneck(x3, t)

        x = self.decoder1(torch.cat([x3, x4], dim=1), t)
        x = self.decoder2(torch.cat([x, x2], dim=1), t)
        x = self.decoder3(torch.cat([x, x1], dim=1), t)
        x = self.output(x, t)
        return x


if __name__ == "__main__":
    model = DDPM_Unet()
    x = torch.randn(1, 3, 32, 32)
    t = torch.randn(1, 1280)
    y = model(x, t)
    print(y.shape)
