import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import math
import warnings
from basicsr.archs.arch_util import SAM
from basicsr.utils.registry import ARCH_REGISTRY
try:
    from ..vmamba.models.vmamba import VSSBlock, LayerNorm2d
except:
    from vmamba.models.vmamba import VSSBlock, LayerNorm2d
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

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = LayerNorm2d(4 * dim)
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


class LN2DLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(LN2DLinear, self).__init__()
        self.norm = LayerNorm2d(in_channels)
        self.linear = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.constant_(m.weight, 0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.linear(self.norm(x))


class BasicBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_blocks=2,
        d_state=1,
        ssm_ratio=1,
        mlp_ratio=4,
        mlp_type="gdmlp",
        sam=False,
        condition=False,
        bayesian=False,
    ):

        super().__init__()
        self.bayesian = bayesian
        self.sam = sam
        self.condition = condition
        self.blocks = nn.ModuleList([])
        if sam:
            self.sam_blocks = nn.ModuleList([])
        if self.condition:
            self.condition_blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                # In our model, the forward way is SS2Dv2 -> forwardv2 -> forward_corev2
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=0,
                    norm_layer=LayerNorm2d,
                    channel_first=True,
                    # =============================
                    ssm_d_state=d_state,  # the state dimension of SSM
                    ssm_ratio=ssm_ratio,  # the rate of data dimension of SSM compared to the data dimension outside SSM
                    ssm_dt_rank="auto",
                    ssm_act_layer=nn.SiLU,
                    ssm_conv=3,
                    ssm_conv_bias=False,  # RetinexMamba sets it True
                    ssm_drop_rate=0,
                    ssm_init="v0",
                    forward_type="v05_noz",  # Vmamba use"v05_noz"
                    # =============================
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=nn.GELU,
                    mlp_drop_rate=0.0,
                    mlp_type=mlp_type,
                    # =============================
                    use_checkpoint=False,
                    post_norm=False,
                )
            )
            if sam:
                self.sam_blocks.append(
                    SAM(in_channel=dim, d_list=(1, 2, 3, 2, 1), inter_num=24)
                )
            if condition:
                self.condition_blocks.append(None)

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
        dim=31,
        num_blocks=[2, 4, 4],
        d_state=1,
        ssm_ratio=1,
        mlp_ratio=4,
        mlp_type="gdmlp",
        use_pixelshuffle=False,
        drop_path=0.0,
        sam=False,
    ):
        super(SubNetwork, self).__init__()
        self.dim = dim
        level = len(num_blocks) - 1
        self.level = level
        self.encoder_layers = nn.ModuleList([])
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        curr_dim = dim
        down_layer = PatchMerging if use_pixelshuffle else conv_down
        up_layer = (
            partial(DualUpSample, scale_factor=2) if use_pixelshuffle else deconv_up
        )

        for i in range(level):
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        BasicBlock(
                            dim=curr_dim,
                            num_blocks=num_blocks[i],
                            d_state=d_state[i],
                            ssm_ratio=ssm_ratio,
                            mlp_ratio=mlp_ratio,
                            mlp_type=mlp_type,
                            sam=sam,
                            bayesian=True,
                        ),
                        down_layer(curr_dim),
                    ]
                )
            )
            curr_dim *= 2

        self.bottleneck = BasicBlock(
            dim=curr_dim,
            num_blocks=num_blocks[-1],
            d_state=d_state[level],
            ssm_ratio=ssm_ratio,
            mlp_ratio=mlp_ratio,
            sam=sam,
            bayesian=True,
        )

        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        up_layer(curr_dim),
                        nn.Conv2d(curr_dim, curr_dim // 2, 1, 1, bias=False),
                        BasicBlock(
                            dim=curr_dim // 2,
                            num_blocks=num_blocks[level - 1 - i],
                            d_state=d_state[level - 1 - i],
                            ssm_ratio=ssm_ratio,
                            mlp_ratio=mlp_ratio,
                            sam=sam,
                            bayesian=True,
                        ),
                    ]
                )
            )
            curr_dim //= 2

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

        return x + self.drop_path(fea)


@ARCH_REGISTRY.register()
class Network(nn.Module):
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
        stage=1,
        num_blocks=[1, 1, 1],
        d_state=1,
        ssm_ratio=1,
        mlp_ratio=4,
        mlp_type="gdmlp",
        use_pixelshuffle=False,
        drop_path=0.0,
        use_illu=False,
        sam=False,
        last_act=None,
    ):
        super(Network, self).__init__()
        self.stage = stage

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
                    num_blocks=num_blocks,
                    d_state=d_state,
                    ssm_ratio=ssm_ratio,
                    mlp_ratio=mlp_ratio,
                    mlp_type=mlp_type,
                    use_pixelshuffle=use_pixelshuffle,
                    drop_path=drop_path,
                    sam=sam,
                )
            )

    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): [batch_size, channels, height, width]
        return:
            out (Tensor): return reconstrcuted images
        """

        out_list = []
        out_list.append(x)

        fea = self.first_conv(x)

        B, C, H, W = fea.size()
        if self.training and mask is not None:
            mask_tokens = self.mask_token.expand(B, -1, H, W)
            w = mask.unsqueeze(1).type_as(mask_tokens)
            fea = fea * (1.0 - w) + mask_tokens * w

        for _idx, subnet in enumerate(self.subnets):
            fea = subnet(fea)
            out = self.proj(fea)
            out = self.last_act(out)
            out_list.append(out)
        return out_list


def build_model():
    return Network(
        stage=1,
        n_feat=40,
        num_blocks=[2, 2, 2],
        d_state=[1, 1, 1],
        ssm_ratio=1,
        mlp_ratio=4,
        mlp_type="gdmlp",
        use_pixelshuffle=True,
    )
