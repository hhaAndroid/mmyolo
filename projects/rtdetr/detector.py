# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.detectors import DetectionTransformer
from mmengine.dist import get_world_size
from mmengine.logging import print_log

from mmyolo.registry import MODELS


@MODELS.register_module()
class RTDetector(DetectionTransformer):
    def __init__(self, *args, use_syncbn: bool = True,**kwargs):
        super().__init__(*args, **kwargs)
        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
            print_log('Using SyncBatchNorm()', 'current')

    def _init_layers(self) -> None:
        super()._init_layers()
        self.bbox_head.init_weights()
