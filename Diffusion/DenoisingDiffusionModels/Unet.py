from base_module import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, n=2, res="mid", attention=False):
        super().__init__()
        self.block = []
        for _ in range(n):
            self.block.append(TimeResidualBlock(in_channels, out_channels))
            self.block.append(
                AttentionBlock(out_channels) if attention else nn.Identity()
            )
            in_channels = out_channels
        if res == "mid":
            self.block.append(TimeResidualBlock(out_channels, out_channels))
        elif res == "up":
            self.block.append(Up(out_channels, out_channels))
        elif res == "down":
            self.block.append(Down(out_channels, out_channels))
        self.block = SwitchSequential(*self.block)

    def forward(self, x, t):
        return self.block(x, t)


class Unet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_res=[1, 2, 4],
        channels=[128, 256, 512, 1024],
        n=2,
        scale=1,
    ):
        super().__init__()
        # 64x64
        assert len(num_res) == len(channels) - 1
        self.stem = nn.Conv2d(in_channels * scale**2, channels[0], 3, 1, 1)
        self.encoder = nn.ModuleList()
        self.mid = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(len(num_res)):
            attn = True if i > 0 else False
            self.encoder.append(
                Block(channels[i], channels[i + 1], n=n, res="down", attention=attn)
            )
        self.mid.append(
            Block(channels[-1], channels[-1], n=2 * n, res="mid", attention=True)
        )
        self.decoder.append(
            Block(
                channels[-1] * 2,
                channels[-1],
                n=n,
                res="up",
                attention=True,
            )
        )
        for i in range(len(num_res) - 1, 0, -1):
            attn = True if i > 0 else False
            self.decoder.append(
                Block(
                    channels[i] + channels[i + 1],
                    channels[i],
                    n=n,
                    res="up",
                    attention=attn,
                )
            )
        block = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                block.append(TimeResidualBlock(channels[0], channels[0] * 2**2))
                block.append(nn.PixelShuffle(2))
                block.append(nn.BatchNorm2d(channels[0]))
                block.append(nn.LeakyReLU(0.2, inplace=True))
        self.up = SwitchSequential(*block)
        self.output = SwitchSequential(
            TimeResidualBlock(channels[1], channels[1]),
            TimeResidualBlock(channels[1], channels[1]),
            nn.Conv2d(channels[1], in_channels, kernel_size=3, padding=1),
        )
        self.time_embedding = nn.Embedding(251, 768)
        self.scale = scale

    def forward(self, x, t):
        t = self.time_embedding(t).view(-1, 768)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h // self.scale, self.scale, w // self.scale, self.scale)
        x = x.permute(0, 3, 5, 1, 2, 4).reshape(b, -1, h // self.scale, w // self.scale)
        x = self.stem(x)
        e = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x, t)
            e.append(x)
        x = self.mid[0](x, t)
        for i in range(len(self.decoder)):
            x = self.decoder[i](torch.cat([x, e.pop()], dim=1), t)
        x = self.up(x, t)
        x = self.output(x, t)
        return x


if __name__ == "__main__":
    model = Unet(3).cuda()
    x = torch.randn(4, 3, 32, 32).cuda()
    t = torch.randint(0, 250, (4,)).cuda()
    with torch.no_grad():
        y = model(x, t)
    print(y.shape)

# 128 32
# 256 16
# 512 8
# 1024 4

# 1024 4

# 2048->1024 8
# 1536->512 16
# 768->256 32
# 384->128 64
