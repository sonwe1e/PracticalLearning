import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
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


class CrossAttention(nn.Module):
    def __init__(self, n_head, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_in_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.kv_in_proj = nn.Linear(d_cross, d_embed * 2, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_head = n_head
        self.d_head = d_embed // n_head

    def forward(self, x, y):
        batch_size, seq_len, d_embed = x.size()
        q = (
            self.q_in_proj(x)
            .view(batch_size, -1, self.n_head, self.d_head)
            .transpose(1, 2)
        )
        k, v = self.kv_in_proj(y.float()).chunk(2, dim=-1)
        k = k.view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(attn, dim=-1)
        attn = attn @ v
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, d_embed)
        attn = self.out_proj(attn)
        return attn


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = SelfAttention(1, channels)

    def forward(self, x):
        residual = x
        x = self.norm(x)

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
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.gn1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.gn2(x)
        x = self.act(x)
        x = self.conv2(x)

        x = x + self.residual_layer(residual)
        return x


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
