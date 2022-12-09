# Copyright (c) 2014-2021 Alibaba PAI-Teams and GOATmessi7. All rights reserved.
import torch
import torch.nn.functional as F
import torch.nn as nn
from mmyolo.registry import MODELS


def get_activation(name='silu', inplace=True):
    if name == 'silu':
        module = nn.SiLU(inplace=inplace)
    elif name == 'relu':
        module = nn.ReLU(inplace=inplace)
    elif name == 'lrelu':
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == 'identity':
        module = nn.Identity(inplace=inplace)
    else:
        raise AttributeError('Unsupported act type: {}'.format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 groups=1,
                 bias=False,
                 act='silu'):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.03, eps=0.001)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class ASFF(nn.Module):

    def __init__(self,
                 level,
                 type='ASFF',
                 asff_channel=2,
                 expand_kernel=3,
                 multiplier=1,
                 act='silu'):
        """
        Args:
            level(int): the level of the input feature
            type(str): ASFF or ASFF_sim
            asff_channel(int): the hidden channel of the attention layer in ASFF
            expand_kernel(int): expand kernel size of the expand layer
            multiplier: should be the same as width in the backbone
        """
        super(ASFF, self).__init__()
        self.level = level
        self.type = type

        self.dim = [
            int(1024 * multiplier),
            int(512 * multiplier),
            int(256 * multiplier)
        ]

        Conv = BaseConv

        self.inter_dim = self.dim[self.level]

        if self.type == 'ASFF':
            if level == 0:
                self.stride_level_1 = Conv(
                    int(512 * multiplier), self.inter_dim, 3, 2, act=act)

                self.stride_level_2 = Conv(
                    int(256 * multiplier), self.inter_dim, 3, 2, act=act)

            elif level == 1:
                self.compress_level_0 = Conv(
                    int(1024 * multiplier), self.inter_dim, 1, 1, act=act)
                self.stride_level_2 = Conv(
                    int(256 * multiplier), self.inter_dim, 3, 2, act=act)

            elif level == 2:
                self.compress_level_0 = Conv(
                    int(1024 * multiplier), self.inter_dim, 1, 1, act=act)
                self.compress_level_1 = Conv(
                    int(512 * multiplier), self.inter_dim, 1, 1, act=act)
            else:
                raise ValueError('Invalid level {}'.format(level))

        # add expand layer
        self.expand = Conv(
            self.inter_dim, self.inter_dim, expand_kernel, 1, act=act)

        self.weight_level_0 = Conv(self.inter_dim, asff_channel, 1, 1, act=act)
        self.weight_level_1 = Conv(self.inter_dim, asff_channel, 1, 1, act=act)
        self.weight_level_2 = Conv(self.inter_dim, asff_channel, 1, 1, act=act)

        self.weight_levels = Conv(asff_channel * 3, 3, 1, 1, act=act)

    def expand_channel(self, x):
        # [b,c,h,w]->[b,c*4,h/2,w/2]
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return x

    def mean_channel(self, x):
        # [b,c,h,w]->[b,c/4,h*2,w*2]
        x1 = x[:, ::2, :, :]
        x2 = x[:, 1::2, :, :]
        return (x1 + x2) / 2

    def forward(self, x):  # l,m,s
        """
        #
        256, 512, 1024
        from small -> large
        """
        x_level_0 = x[2]  # max feature level [512,20,20]
        x_level_1 = x[1]  # mid feature level [256,40,40]
        x_level_2 = x[0]  # min feature level [128,80,80]

        if self.type == 'ASFF':
            if self.level == 0:
                level_0_resized = x_level_0
                level_1_resized = self.stride_level_1(x_level_1)
                level_2_downsampled_inter = F.max_pool2d(
                    x_level_2, 3, stride=2, padding=1)
                level_2_resized = self.stride_level_2(
                    level_2_downsampled_inter)
            elif self.level == 1:
                level_0_compressed = self.compress_level_0(x_level_0)
                level_0_resized = F.interpolate(
                    level_0_compressed, scale_factor=2, mode='nearest')
                level_1_resized = x_level_1
                level_2_resized = self.stride_level_2(x_level_2)
            elif self.level == 2:
                level_0_compressed = self.compress_level_0(x_level_0)
                level_0_resized = F.interpolate(
                    level_0_compressed, scale_factor=4, mode='nearest')
                x_level_1_compressed = self.compress_level_1(x_level_1)
                level_1_resized = F.interpolate(
                    x_level_1_compressed, scale_factor=2, mode='nearest')
                level_2_resized = x_level_2
        else:
            if self.level == 0:
                level_0_resized = x_level_0
                level_1_resized = self.expand_channel(x_level_1)
                level_1_resized = self.mean_channel(level_1_resized)
                level_2_resized = self.expand_channel(x_level_2)
                level_2_resized = F.max_pool2d(
                    level_2_resized, 3, stride=2, padding=1)
            elif self.level == 1:
                level_0_resized = F.interpolate(
                    x_level_0, scale_factor=2, mode='nearest')
                level_0_resized = self.mean_channel(level_0_resized)
                level_1_resized = x_level_1
                level_2_resized = self.expand_channel(x_level_2)
                level_2_resized = self.mean_channel(level_2_resized)

            elif self.level == 2:
                level_0_resized = F.interpolate(
                    x_level_0, scale_factor=4, mode='nearest')
                level_0_resized = self.mean_channel(
                    self.mean_channel(level_0_resized))
                level_1_resized = F.interpolate(
                    x_level_1, scale_factor=2, mode='nearest')
                level_1_resized = self.mean_channel(level_1_resized)
                level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:
                                                               1, :, :] + level_1_resized * levels_weight[:,
                                                                                            1:
                                                                                            2, :,
                                                                                            :] + level_2_resized * levels_weight[
                                                                                                                   :,
                                                                                                                   2:,
                                                                                                                   :, :]
        out = self.expand(fused_out_reduced)

        return out


@MODELS.register_module()
class ASFFNeck(nn.Module):
    def __init__(self, widen_factor, use_att='ASFF', asff_channel=2, expand_kernel=3, act='silu'):
        super().__init__()
        self.asff_1 = ASFF(
            level=0,
            type=use_att,
            asff_channel=asff_channel,
            expand_kernel=expand_kernel,
            multiplier=widen_factor,
            act=act,
        )
        self.asff_2 = ASFF(
            level=1,
            type=use_att,
            asff_channel=asff_channel,
            expand_kernel=expand_kernel,
            multiplier=widen_factor,
            act=act,
        )
        self.asff_3 = ASFF(
            level=2,
            type=use_att,
            asff_channel=asff_channel,
            expand_kernel=expand_kernel,
            multiplier=widen_factor,
            act=act,
        )

    def forward(self, x):
        pan_out0 = self.asff_1(x)
        pan_out1 = self.asff_2(x)
        pan_out2 = self.asff_3(x)
        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
