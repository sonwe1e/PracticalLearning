from base_module import *


class Decoder(nn.Module):
    def __init__(self, channels=[512, 256, 128]):
        super().__init__()
        self.static_feat = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=1),
            nn.Conv2d(4, channels[0], kernel_size=3, padding=1),
        )
        self.block1 = nn.Sequential(
            ResidualBlock(channels[0], channels[0]),
            AttentionBlock(channels[0]),
            ResidualBlock(channels[0], channels[0]),
            ResidualBlock(channels[0], channels[0]),
            ResidualBlock(channels[0], channels[0]),
            ResidualBlock(channels[0], channels[0]),
            nn.ConvTranspose2d(
                channels[0], channels[0], kernel_size=3, stride=2, padding=1
            ),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            ResidualBlock(channels[0], channels[0]),
            ResidualBlock(channels[0], channels[0]),
            ResidualBlock(channels[0], channels[0]),
            nn.ConvTranspose2d(
                channels[0], channels[0], kernel_size=3, stride=2, padding=1
            ),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1),
        )
        self.block3 = nn.Sequential(
            ResidualBlock(channels[0], channels[1]),
            ResidualBlock(channels[1], channels[1]),
            ResidualBlock(channels[1], channels[1]),
            nn.ConvTranspose2d(
                channels[1], channels[1], kernel_size=3, stride=2, padding=1
            ),
            nn.Conv2d(channels[1], channels[1], kernel_size=3, padding=1),
        )
        self.block4 = nn.Sequential(
            ResidualBlock(channels[1], channels[2]),
            ResidualBlock(channels[2], channels[2]),
            ResidualBlock(channels[2], channels[2]),
        )
        self.norm = nn.GroupNorm(32, channels[2])
        self.act = nn.SiLU()
        self.get_rgb = nn.Sequential(
            nn.Conv2d(channels[2], 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x /= 0.18215
        static_feat = self.static_feat(x)
        x = self.block1(static_feat)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.get_rgb(x)
        return x
