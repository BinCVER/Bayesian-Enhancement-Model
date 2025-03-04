import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numbers
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
from einops import rearrange
from basicsr.archs.arch_util import SAM
from basicsr.utils.gaussian_downsample import downsample_fft
from basicsr.utils.poisson_gaussian import add_poisson_gaussian_noise
from basicsr.utils.registry import ARCH_REGISTRY
from functools import partial
from timm.models.layers import DropPath
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"



def _no_grad_trunc_normal_(
    tensor, mean, std, a, b
):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)

        return tensor


def trunc_normal_(
    tensor, mean=0.0, std=1.0, a=-2.0, b=2.0
):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)



def conv_down(in_channels):
    return nn.Conv2d(in_channels, in_channels * 2, 4, 2, 1, bias=False)


class PatchMerging(nn.Module):
    """Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, dim, norm_type='LayerNorm2d', norm_bias=True):
        super().__init__()
        self.dim = dim
        if norm_type == 'LayerNorm2d':
            if not norm_bias:
                norm_layer = partial(LayerNormS, LayerNorm_type='BiasFree')
            else:
                norm_layer = LayerNorm2d
        elif norm_type == 'InstanceNorm2d':
            norm_layer = partial(nn.InstanceNorm2d, affine=True, track_running_stats=False)
        else:
            raise ValueError('Not Imp!')
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Conv2d(4 * dim, 2 * dim, 1, 1, 0, bias=False)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        x0 = x[:, :, 0::2, 0::2]  # B, C, H/2, W/2
        x1 = x[:, :, 1::2, 0::2]  # B, C, H/2, W/2
        x2 = x[:, :, 0::2, 1::2]  # B, C, H/2, W/2
        x3 = x[:, :, 1::2, 1::2]  # B, C, H/2, W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # B, 4C, H/2, W/2

        x = self.reduction(self.norm(x))

        return x


def deconv_up(in_channels):
    return nn.ConvTranspose2d(
        in_channels,
        in_channels // 2,
        stride=2,
        kernel_size=2,
        padding=0,
        output_padding=0,
    )


# Dual Up-Sample
class DualUpSample(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(DualUpSample, self).__init__()
        self.factor = scale_factor

        if self.factor == 2:
            self.conv = nn.Conv2d(in_channels, in_channels // 2, 1, 1, 0, bias=False)
            self.up_p = nn.Sequential(
                nn.Conv2d(in_channels, 2 * in_channels, 1, 1, 0, bias=False),
                nn.PReLU(),
                nn.PixelShuffle(scale_factor),
                nn.Conv2d(
                    in_channels // 2,
                    in_channels // 2,
                    1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
            )

            self.up_b = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                nn.PReLU(),
                nn.Upsample(
                    scale_factor=scale_factor, mode="bilinear", align_corners=False
                ),
                nn.Conv2d(
                    in_channels, in_channels // 2, 1, stride=1, padding=0, bias=False
                ),
            )
        elif self.factor == 4:
            self.conv = nn.Conv2d(2 * in_channels, in_channels, 1, 1, 0, bias=False)
            self.up_p = nn.Sequential(
                nn.Conv2d(in_channels, 16 * in_channels, 1, 1, 0, bias=False),
                nn.PReLU(),
                nn.PixelShuffle(scale_factor),
                nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False),
            )

            self.up_b = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1, 1, 0),
                nn.PReLU(),
                nn.Upsample(
                    scale_factor=scale_factor, mode="bilinear", align_corners=False
                ),
                nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False),
            )

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        # breakpoint()
        x_p = self.up_p(x)  # pixel shuffle
        x_b = self.up_b(x)  # bilinear
        out = self.conv(torch.cat([x_p, x_b], dim=1))

        return out


##########################################################################
## HalfInstanceNorm2d

class HalfInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        """
        Args:
            num_features (int): Number of channels in the input.
            eps (float): Small value added to variance for numerical stability.
            momentum (float): Value for updating running mean and variance.
            affine (bool): If True, learnable weight and bias are added.
            track_running_stats (bool): If True, running mean and variance are tracked.
        """
        super(HalfInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.half_features = num_features // 2  # Only normalize the first half of the channels

        # InstanceNorm2d for the first half of the channels
        self.instance_norm = nn.InstanceNorm2d(
            self.half_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats
        )

    def forward(self, input):
        # Split the input into two parts: normalize the first half, leave the second half unchanged
        input1, input2 = input[:, :self.half_features], input[:, self.half_features:]

        # Apply InstanceNorm2d to the first half
        input1_normalized = self.instance_norm(input1)

        # Concatenate the normalized and unnormalized parts
        output = torch.cat([input1_normalized, input2], dim=1)

        return output

##########################################################################
## Layer Norm

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNormS(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNormS, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)



    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, norm_type='LayerNorm2d', norm_bias=True):
        super(TransformerBlock, self).__init__()
        if norm_type == 'LayerNorm2d':
            if not norm_bias:
                norm_layer = partial(LayerNormS, LayerNorm_type='BiasFree')
            else:
                norm_layer = LayerNorm2d
        elif norm_type == 'InstanceNorm2d':
            norm_layer = partial(nn.InstanceNorm2d, affine=True, track_running_stats=False)
        else:
            raise ValueError('Not Imp!')
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = norm_layer(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        MLP Module for (B, C, H, W) input.

        Args:
            in_features (int): Number of input channels.
            hidden_features (int): Number of hidden channels. Default: in_features.
            out_features (int): Number of output channels. Default: in_features.
            act_layer (nn.Module): Activation layer. Default: nn.GELU.
            drop (float): Dropout rate. Default: 0.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)  # 1x1 Conv for dense connection
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)  # 1x1 Conv
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Forward pass for (B, C, H, W) input.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).
        Returns:
            torch.Tensor: Output tensor with shape (B, C, H, W).
        """
        x = self.fc1(x)  # Apply first linear transformation
        x = self.act(x)  # Apply activation
        x = self.drop(x)  # Apply dropout
        x = self.fc2(x)  # Apply second linear transformation
        x = self.drop(x)  # Apply dropout
        return x



def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): Window size.

    Returns:
        windows: (num_windows * B, window_size, window_size, C)
        H_pad, W_pad: Padded height and width.
    """
    B, H, W, C = x.shape

    # 计算需要填充的高度和宽度
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # 在 H 和 W 方向填充
    H_pad, W_pad = x.shape[1], x.shape[2]

    # 划分窗口
    x = x.view(B, H_pad // window_size, window_size, W_pad // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, num_h, num_w, window_size, window_size, C)
    windows = x.view(-1, window_size, window_size, C)  # (num_windows * B, window_size, window_size, C)

    return windows



def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Original height of image
        W (int): Original width of image

    Returns:
        x: (B, H, W, C)
    """
    B, C = windows.shape[0] // ((H + (window_size - H % window_size) % window_size) *
                                (W + (window_size - W % window_size) % window_size) //
                                window_size // window_size), windows.shape[-1]
    H_pad = H + (window_size - H % window_size) % window_size
    W_pad = W + (window_size - W % window_size) % window_size

    # Reshape to recover windows
    x = windows.view(B, H_pad//window_size, W_pad//window_size, window_size, window_size, -1)
    # Rearrange dimensions
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    # Merge dimensions
    x = x.view(B, H_pad, W_pad, C)

    # Remove padding if necessary
    if H_pad != H or W_pad != W:
        x = x[:, :H, :W, :]
    return x



class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
        relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.stack((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = self.qkv(x)
        qkv = qkv.view(B_, N, 3, -1)
        # 分别添加不同的 bias
        if self.q_bias is not None:
            qkv += qkv_bias.view(1, 1, 3, -1)  # (1, 1, dim)

        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01).to(self.logit_scale.device))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=LayerNorm2d):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # 训练时的固定分辨率
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(window_size, window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 在训练时提前生成 img_mask
        self.H, self.W = input_resolution
        if self.shift_size > 0:
            self.attn_mask = self.create_mask(self.H, self.W)
        else:
            self.attn_mask = None

    def create_mask(self, H, W):
        """
        动态生成 SW-MSA 的 attention mask。
        """
        img_mask = torch.zeros((1, H, W, 1), device="cuda")  # 创建初始掩码
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # 划分窗口
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入特征。
        """
        B, C, H, W = x.shape
        assert C == self.dim, f"Input channel size ({C}) must match block dimension ({self.dim})."

        shortcut = x

        # Norm and Attention
        x = self.norm1(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # 转换为 (B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # (num_windows * B, window_size, window_size, C)
        # 转换为 (num_windows * B, window_size * window_size, C)
        x_windows = x_windows.flatten(1, 2)



        # 在推理阶段动态生成 mask
        if self.shift_size > 0 and (H != self.H or W != self.W):
            attn_mask = self.create_mask(H, W)
        else:
            attn_mask = self.attn_mask

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # breakpoint()
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.permute(0, 3, 1, 2).contiguous()  # 转换回 (B, C, H, W)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x



class BasicBlock(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_blocks=2,
        num_heads=8,
        ffn_expansion_factor=2.66,
        bias=False,
        sam=False,
        norm_type='LayerNorm2d',
        spatial_attention=False,
        window_size=7,
        bayesian=False,
        norm_bias=True,
    ):

        super().__init__()
        self.bayesian = bayesian
        self.sam = sam
        self.blocks = nn.ModuleList([])
        if sam:
            self.sam_blocks = nn.ModuleList([])
        for i in range(num_blocks):
            if spatial_attention:
                self.blocks.append(
                    SwinTransformerBlock(dim, input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=ffn_expansion_factor, qkv_bias=bias, norm_layer=LayerNorm2d)
                )
            else:
                self.blocks.append(
                TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, norm_type=norm_type, norm_bias=norm_bias)
            )
            if sam:
                self.sam_blocks.append(
                    SAM(in_channel=dim, d_list=(1, 2, 3, 2, 1), inter_num=24)
                )

    def forward(self, x):
        for _idx, block in enumerate(self.blocks):
            x = block(x)
            if self.sam:
                x = self.sam_blocks[_idx](x)
        return x


class SubNetwork(nn.Module):
    """
    The main module representing as a shallower UNet
    args:
        dim (int): number of channels of input and output
        num_blocks (list): each element defines the number of basic blocks in a scale level
        d_state (int): dimension of the hidden state in S6
        ssm_ratio(int): expansion ratio of SSM in S6
        mlp_ratio (float): expansion ratio of MLP in S6
        use_pixelshuffle (bool): when true, use pixel(un)shuffle for up(down)-sampling, otherwise, use Transposed Convolution.
        drop_path (float): ratio of droppath beween two subnetwork
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_blocks=[1,1,3,3,3],
        heads=[1,2,4],
        ffn_expansion_factor=2.66,
        bias=False,
        use_pixelshuffle=False,
        drop_path=0.0,
        spatial_attention=False,
        window_size=7,
        sam=False,
        global_shortcut=True,
        norm_type='LayerNorm2d',
        norm_bias=True,
    ):
        super(SubNetwork, self).__init__()
        self.dim = dim
        self.global_shortcut = global_shortcut
        level = len(num_blocks) // 2
        self.level = level
        self.encoder_layers = nn.ModuleList([])
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        curr_dim = dim
        down_layer = partial(PatchMerging, norm_type=norm_type) if use_pixelshuffle else conv_down
        up_layer = (
            partial(DualUpSample, scale_factor=2) if use_pixelshuffle else deconv_up
        )

        train_h, train_w = input_resolution
        curr_depth = 0
        for i in range(level):
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        BasicBlock(
                            dim=curr_dim,
                            input_resolution=(train_h, train_w),
                            num_blocks=num_blocks[curr_depth],
                            num_heads=heads[i],
                            ffn_expansion_factor=ffn_expansion_factor,
                            spatial_attention=spatial_attention,
                            window_size=window_size,
                            bias=bias,
                            sam=sam,
                            norm_type=norm_type,
                            bayesian=True,
                            norm_bias=norm_bias,
                        ),
                        down_layer(curr_dim),
                    ]
                )
            )
            curr_dim *= 2
            curr_depth += 1
            train_h, train_w = train_h//2, train_w//2

        self.bottleneck = BasicBlock(
            dim=curr_dim,
            input_resolution=(train_h, train_w),
            num_blocks=num_blocks[curr_depth],
            num_heads=heads[level],
            ffn_expansion_factor=ffn_expansion_factor,
            spatial_attention=spatial_attention,
            window_size=window_size,
            bias=bias,
            sam=sam,
            norm_type=norm_type,
            bayesian=True,
            norm_bias=norm_bias,
        )
        curr_depth += 1

        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            train_h, train_w = int(train_h*2), int(train_w*2)
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        up_layer(curr_dim),
                        nn.Conv2d(curr_dim, curr_dim // 2, 1, 1, bias=False),
                        BasicBlock(
                            dim=curr_dim // 2,
                            input_resolution=(train_h, train_w),
                            num_blocks=num_blocks[curr_depth],
                            num_heads=heads[level - 1 - i],
                            ffn_expansion_factor=ffn_expansion_factor,
                            spatial_attention=spatial_attention,
                            window_size=window_size,
                            bias=bias,
                            sam=sam,
                            norm_type=norm_type,
                            bayesian=True,
                            norm_bias=norm_bias,
                        ),
                    ]
                )
            )
            curr_dim //= 2
            curr_depth += 1

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):

        fea = x
        #####Encoding Process-------------------------------------------------------------------------------------------
        fea_encoder = []
        for en_block, down_layer in self.encoder_layers:
            fea = en_block(fea)
            fea_encoder.append(fea)
            fea = down_layer(fea)
        fea = self.bottleneck(fea)
        ######----------------------------------------------------------------------------------------------------------
        ######Decoding Process------------------------------------------------------------------------------------------
        for i, (up_layer, fusion, de_block) in enumerate(self.decoder_layers):
            fea = up_layer(fea)
            fea = fusion(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            fea = de_block(fea)

        if self.global_shortcut:
            out = x + self.drop_path(fea)
        else:
            out = self.drop_path(fea)
        return out

@ARCH_REGISTRY.register()
class UTransformer(nn.Module):
    """
    The Model
    args:
        in_channels (int): input channel number
        out_channels (int): output channel number
        n_feat (int): channel number of intermediate features
        stage (int): number of stages。
        num_blocks (list): each element defines the number of basic blocks in a scale level
        d_state (int): dimension of the hidden state in S6
        ssm_ratio(int): expansion ratio of SSM in S6
        mlp_ratio (float): expansion ratio of MLP in S6
        use_pixelshuffle (bool): when true, use pixel(un)shuffle for up(down)-sampling, otherwise, use Transposed Convolution.
        drop_path (float): ratio of droppath beween two subnetwork
        use_illu (bool): true to include an illumination layer
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        n_feat=40,
        input_resolution=(48, 48),
        stage=1,
        num_blocks=[1, 1, 1],
        heads=[1,2,4],
        ffn_expansion_factor=2.66,
        spatial_attention=False,
        window_size=7,
        bias=False,
        use_pixelshuffle=False,
        drop_path=0.0,
        sam=False,
        last_act=None,
        downscale=8,
        global_shortcut=True,
        norm_type='LayerNorm2d',
        norm_bias=True,
    ):
        super(UTransformer, self).__init__()
        self.stage = stage
        self.downscale = downscale

        self.mask_token = nn.Parameter(torch.zeros(1, n_feat, 1, 1))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)

        # 11/07/2024 set bias True, following MoCov3; Meanwhile, bias can help ajust input's mean
        self.first_conv = nn.Conv2d(in_channels, n_feat, 3, 1, 1, bias=True)
        nn.init.kaiming_normal_(
            self.first_conv.weight, mode="fan_out", nonlinearity="linear"
        )
        # nn.init.xavier_normal_(self.conv_proj.weight)
        if self.first_conv.bias is not None:
            nn.init.zeros_(self.first_conv.bias)
        # nn.init.xavier_normal_(self.dynamic_emblayer.weight)

        # # freeze embedding layer
        # for param in self.static_emblayer.parameters():
        #     param.requires_grad = False

        self.alpha = 0.0238

        self.subnets = nn.ModuleList([])

        self.proj = nn.Conv2d(n_feat, out_channels, 3, 1, 1, bias=True)
        # nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        if last_act is None:
            self.last_act = nn.Identity()
        elif last_act == "relu":
            self.last_act = nn.ReLU()
        elif last_act == "softmax":
            self.last_act = nn.Softmax(dim=1)
        else:
            raise NotImplementedError

        for i in range(stage):
            self.subnets.append(
                SubNetwork(
                    dim=n_feat,
                    input_resolution=input_resolution,
                    num_blocks=num_blocks,
                    heads=heads,
                    ffn_expansion_factor=ffn_expansion_factor,
                    spatial_attention=spatial_attention,
                    window_size=window_size,
                    bias=bias,
                    use_pixelshuffle=use_pixelshuffle,
                    drop_path=drop_path,
                    sam=sam,
                    global_shortcut=global_shortcut,
                    norm_type=norm_type,
                    norm_bias=norm_bias,
                )
            )


    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): [batch_size, channels, height, width]
        return:
            out (Tensor): return reconstructed images
        """

        if self.downscale > 1:
            return self.__forwardI(x, mask)
        else:
            return self.__forwardII(x, mask)

    def __forwardI(self, x, mask=None):

        x_down = downsample_fft(x)
        fea = self.first_conv(x_down)

        B, C, H, W = fea.size()
        if self.training and mask is not None:
            mask_tokens = self.mask_token.expand(B, -1, H, W)
            w = mask.unsqueeze(1).type_as(mask_tokens)
            fea = fea * (1.0 - w) + mask_tokens * w

        for _idx, subnet in enumerate(self.subnets):
            fea = subnet(fea)
        coarse = self.last_act(self.proj(fea))
        coarse_up = F.interpolate(coarse, scale_factor=self.downscale, mode='bilinear', align_corners=False)

        out = (x + self.alpha * coarse_up) *  coarse_up

        return coarse, coarse_up, out

    def __forwardII(self, x, mask=None):

        fea = x
        fea = self.first_conv(fea)

        B, C, H, W = fea.size()
        if self.training and mask is not None:
            mask_tokens = self.mask_token.expand(B, -1, H, W)
            w = mask.unsqueeze(1).type_as(mask_tokens)
            fea = fea * (1.0 - w) + mask_tokens * w

        for _idx, subnet in enumerate(self.subnets):
            fea = subnet(fea)
        fea = self.last_act(self.proj(fea))
        out = fea
        return x, out



def build_model() -> UTransformer:
    return UTransformer(
        in_channels=3,
        out_channels=3,
        n_feat=40,
        input_resolution=(256, 256),
        stage=1,
        num_blocks=[1,1,3,3,3],
        heads=[1,2,4],
        ffn_expansion_factor=2.66,
        spatial_attention=False,
        window_size=7,
        bias=False,
        use_pixelshuffle=True,
        drop_path=0.0,
        sam=False,
        last_act=None,
        downscale=8,
        global_shortcut=False,
        norm_type='LayerNorm2d',
        norm_bias=True,
    )
