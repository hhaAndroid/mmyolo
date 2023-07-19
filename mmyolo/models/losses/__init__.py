# Copyright (c) OpenMMLab. All rights reserved.
from .iou_loss import IoULoss, bbox_overlaps
from .varifocal_loss import VarifocalLoss

__all__ = ['IoULoss', 'bbox_overlaps']
