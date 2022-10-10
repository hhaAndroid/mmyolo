# Copyright (c) OpenMMLab. All rights reserved.
import functools
import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from mmengine.dist import get_dist_info

from mmyolo.registry import MODELS
from ..layers import yolov7_brick as vn_layer
from .yolov5_head import YOLOv5Head


def _make_divisible(x, divisor, width_multiple):
    return math.ceil(x * width_multiple / divisor) * divisor


def make_divisible(divisor, width_multiple=1.0):
    return functools.partial(
        _make_divisible, divisor=divisor, width_multiple=width_multiple)


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


@MODELS.register_module()
class YOLOV7Head(YOLOv5Head):

    def __init__(self,
                 num_classes,
                 in_channels,
                 depth_multiple=1.0,
                 width_multiple=1.0,
                 anchor_generator=dict(
                     type='YOLOAnchorGenerator',
                     base_sizes=[[(116, 90), (156, 198), (373, 326)],
                                 [(30, 61), (62, 45), (59, 119)],
                                 [(10, 13), (16, 30), (33, 23)]],
                     strides=[32, 16, 8]),
                 bbox_coder=dict(type='YOLOv5BBoxCoder'),
                 featmap_strides=[8, 16, 32],
                 loss_cls=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 **kwargs):
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        super().__init__(None,
                         prior_generator=anchor_generator,
                         bbox_coder=bbox_coder,
                         loss_cls=loss_cls,
                         loss_bbox=loss_bbox,
                         **kwargs)

        ch = [256, 512, 1024]
        self.no = 5 + self.num_classes
        self.na = 3
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.num_base_priors, 1) for x in ch)  # output conv

        # Can be removed when aligning inference accuracy
        # self.ia = nn.ModuleList(vn_layer.ImplicitA(x) for x in ch)
        # self.im = nn.ModuleList(vn_layer.ImplicitM(self.no * self.na) for _ in ch)

        # self.loss_fun = ComputeLossOTA(self, self.prior_generator)

    def special_init(self):
        layers, self.save = vn_layer.parse_model(vn_layer.v7l_backbone,
                                                 vn_layer.v7l_head,
                                                 self.depth_multiple,
                                                 self.width_multiple,
                                                 [3])

        self.det = nn.Sequential(*layers)
        # self.save = save
        self.return_index = [102, 103, 104]

    def forward(self, x):
        y, out = [], []  # outputs
        for i, m in enumerate(self.det):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)  # save output

            if i in self.return_index:
                out.append(x)
        del y

        x = out
        cls_scores=[]
        bbox_preds=[]
        objectnesss=[]
        for i in range(3):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape

            pred_map = x[i].view(bs, self.num_base_priors, 85,
                                     ny, nx)

            cls_score = pred_map[:, :, 5:, ...].reshape(bs, -1, ny, nx)
            bbox_pred = pred_map[:, :, :4, ...].reshape(bs, -1, ny, nx)
            objectness = pred_map[:, :, 4:5, ...].reshape(bs, -1, ny, nx)

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            objectnesss.append(objectness)

            # # Can be removed when aligning inference accuracy
            # x[i] = self.m[i](self.ia[i](x[i]))  # conv
            # x[i] = self.im[i](x[i])

        return cls_scores, bbox_preds, objectnesss

    def init_weights(self):
        pass

    def loss(self, pred_maps, data_samples):
        pass