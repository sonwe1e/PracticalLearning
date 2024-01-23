import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up1_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.up2_conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.up2_conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.up2_bn1 = nn.BatchNorm2d(out_channels)
        self.up2_bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x1 = self.up1(x)
        x1 = self.up1_conv(x1)

        x = self.up2_bn1(x)
        x = self.act(x)
        x = self.up2(x)
        x = self.up2_conv1(x)
        x = self.up2_bn2(x)
        x = self.act(x)
        x = self.up2_conv2(x)
        return x + x1


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down1_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.down2_conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.down2_conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.act = nn.ReLU()
        self.down1 = nn.AvgPool2d(2)
        self.down2 = nn.AvgPool2d(2)

    def forward(self, x):
        x1 = self.down1_conv(x)
        x1 = self.down1(x1)

        x = self.act(x)
        x = self.down2_conv1(x)
        x = self.act(x)
        x = self.down2_conv2(x)
        x = self.down2(x)
        return x + x1


class LineAttention(nn.Module):
    def __init__(self, n_head, n_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(n_embed, n_embed * 3, bias=in_proj_bias)
        self.out_proj = nn.Linear(n_embed, n_embed, bias=out_proj_bias)
        self.n_head = n_head
        self.d_head = n_embed // n_head

    def forward(self, x):
        batch_size, seq_len, n_embed = x.size()
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(attn, dim=-1)
        attn = attn @ v
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, n_embed)
        attn = self.out_proj(attn)
        return attn


class ConvAttention(nn.Module):
    def __init__(self, n_head, n_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q = nn.Conv2d(n_embed, n_embed, 1, bias=in_proj_bias)
        self.k = nn.Conv2d(n_embed, n_embed, 1, bias=in_proj_bias)
        self.v = nn.Conv2d(n_embed, n_embed, 1, bias=in_proj_bias)
        self.out_proj = nn.Conv2d(n_embed, n_embed, 1, bias=out_proj_bias)
        self.n_head = n_head
        self.d_head = n_embed // n_head

    def forward(self, x):
        batch_size, channel, h, w = x.size()
        q = self.q(x).reshape(batch_size, self.n_head, self.d_head, h * w)
        k = self.k(x).reshape(batch_size, self.n_head, self.d_head, h * w)
        v = self.v(x).reshape(batch_size, self.n_head, self.d_head, h * w)
        attn = (q.transpose(-2, -1) @ k) / math.sqrt(self.d_head)
        attn = F.softmax(attn, dim=-1)
        attn = attn @ v.transpose(-2, -1)
        attn = attn.transpose(1, 2).reshape(batch_size, self.n_head * self.d_head, h, w)
        attn = self.out_proj(attn)
        return attn


class AttentionBlock(nn.Module):
    def __init__(self, channels, head=8, attn_type="conv"):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn_type = attn_type
        self.attn = (
            ConvAttention(head, channels)
            if attn_type == "conv"
            else LineAttention(head, channels)
        )

    def forward(self, x):
        residual = x
        x = self.norm(x)

        if self.attn_type == "conv":
            x = self.attn(x)
        else:
            n, c, h, w = x.size()
            x = x.view(n, c, -1).transpose(1, 2)
            x = self.attn(x)
            x = x.transpose(1, 2).view(n, c, h, w)
        x += residual
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gn1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1)
        self.gn2 = nn.GroupNorm(32, out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, 3, 1, 1)
        self.gn3 = nn.GroupNorm(32, out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, 1)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.gn1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.gn2(x)
        x = self.act(x)
        x = self.conv2(x)

        x = self.gn3(x)
        x = self.act(x)
        x = self.conv3(x)

        x = x + self.residual_layer(residual)
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.l1 = nn.Linear(n_embed, n_embed)
        self.l2 = nn.Linear(n_embed, n_embed)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        return x


class TimeResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=768):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv1 = ResidualBlock(in_channels, out_channels)
        self.conv2 = ResidualBlock(out_channels, out_channels)
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


# 这一段可以模仿，没有见过这种写法
class SwitchSequential(nn.Sequential):
    def forward(self, x, time):
        for layer in self:
            if isinstance(layer, TimeResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x
