# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmyolo.registry import MODELS


@MODELS.register_module()
class YOLOV7Backbone(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
