from base_module import *


class Encoder(nn.Module):
    def __init__(self, channels=[128, 256, 512]):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1),
            ResidualBlock(channels[0], channels[0]),
            ResidualBlock(channels[0], channels[0]),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=2, padding=1),
            ResidualBlock(channels[0], channels[1]),
            ResidualBlock(channels[1], channels[1]),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=2, padding=1),
            ResidualBlock(channels[1], channels[2]),
            ResidualBlock(channels[2], channels[2]),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=2, padding=1),
            ResidualBlock(channels[2], channels[2]),
            ResidualBlock(channels[2], channels[2]),
            ResidualBlock(channels[2], channels[2]),
            AttentionBlock(channels[2]),
            ResidualBlock(channels[2], channels[2]),
        )
        self.norm = nn.GroupNorm(32, channels[2])
        self.act = nn.SiLU()
        self.get_static_feat = nn.Sequential(
            nn.Conv2d(channels[2], 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1),
        )

    def forward(self, x, noise):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.norm(x)
        x = self.act(x)
        static_feat = self.get_static_feat(x)
        mean, log_var = static_feat.chunk(2, dim=1)
        log_var = log_var.clamp(-30, 20)
        var = log_var.exp()
        stdev = var.sqrt()
        x = mean + stdev * noise
        x *= 0.18215
        return x
