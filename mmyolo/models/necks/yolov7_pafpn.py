# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from .base_yolo_neck import BaseYOLONeck
from ..layers import SPPCSPBlock, ELANBlock, MaxPoolBlock, RepVGGBlock


@MODELS.register_module()
class YOLOv7PAFPN(BaseYOLONeck):
    """Path Aggregation Network used in YOLOv7.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        freeze_all(bool): Whether to freeze the model. Defaults to False.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: List[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):

        super().__init__(
            in_channels=[
                int(channel * widen_factor) for channel in in_channels
            ],
            out_channels=[
                int(channel * widen_factor) for channel in out_channels
            ],
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        if idx == 2:
            layer = SPPCSPBlock(self.in_channels[idx],
                                self.out_channels[idx],
                                expand_ratio=0.5,
                                kernel_sizes=(5, 9, 13),
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg)
        else:
            layer = ConvModule(self.in_channels[idx],
                               self.out_channels[idx],
                               1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg)

        return layer

    def build_upsample_layer(self, idx: int) -> nn.Module:
        """build upsample layer."""
        return nn.Sequential(ConvModule(self.out_channels[idx],
                                        self.out_channels[idx - 1],
                                        1,
                                        norm_cfg=self.norm_cfg,
                                        act_cfg=self.act_cfg),
                             nn.Upsample(scale_factor=2, mode='nearest')
                             )

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        return ELANBlock(self.out_channels[idx - 1] * 2,
                         'type3',
                         num_blocks=4,
                         norm_cfg=self.norm_cfg,
                         act_cfg=self.act_cfg)

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        return MaxPoolBlock(self.out_channels[idx],
                            mode='type2',
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        return ELANBlock(self.out_channels[idx+1] * 2,
                         'type3',
                         num_blocks=4,
                         norm_cfg=self.norm_cfg,
                         act_cfg=self.act_cfg)

    def build_out_layer(self, idx: int) -> nn.Module:
        """build out layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The out layer.
        """
        return RepVGGBlock(
            self.out_channels[idx],
            self.out_channels[idx] * 2,
            3,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
