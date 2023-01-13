# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.utils import multi_apply
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig)
from mmengine.dist import get_dist_info
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS, TASK_UTILS
from ..utils import make_divisible
from .v8_utils import (BboxLoss, TaskAlignedAssigner, bbox2dist, dist2bbox,
                       make_anchors)
from .yolov5_head import YOLOv5Head


@MODELS.register_module()
class YOLOv8HeadModule(BaseModule):
    """YOLOv8HeadModule head module used in `YOLOv8`.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (Union[int, Sequence]): Number of channels in the input
            feature map.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_base_priors (int): The number of priors (points) at a point
            on the feature grid.
        featmap_strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to [8, 16, 32].
        reg_max (int): Max value of integral set :math: ``{0, ..., reg_max-1}``
            in QFL setting. Defaults to 16.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 num_base_priors: int = 1,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 reg_max: int = 16,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.num_base_priors = num_base_priors
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_channels = in_channels
        self.reg_max = reg_max

        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels

        self._init_layers()

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        super().init_weights()
        for a, b, s in zip(self.reg_preds, self.cls_preds,
                           self.featmap_strides):
            a[-1].bias.data[:] = 1.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            b[-1].bias.data[:self.num_classes] = math.log(
                5 / self.num_classes / (640 / s)**2)

    def _init_layers(self):
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=reg_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    ConvModule(
                        in_channels=reg_out_channels,
                        out_channels=reg_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    nn.Conv2d(
                        in_channels=reg_out_channels,
                        out_channels=4 * self.reg_max,
                        kernel_size=1)))
            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=cls_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    ConvModule(
                        in_channels=cls_out_channels,
                        out_channels=cls_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    nn.Conv2d(
                        in_channels=cls_out_channels,
                        out_channels=self.num_classes,
                        kernel_size=1)))

        # proj = torch.linspace(0, self.reg_max - 1, self.reg_max)
        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions
        """
        assert len(x) == self.num_levels
        return multi_apply(self.forward_single, x, self.cls_preds,
                           self.reg_preds)

    def forward_single(self, x: torch.Tensor, cls_pred: nn.ModuleList,
                       reg_pred: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = x.shape
        cls_logit = cls_pred(x)
        bbox_dist_preds = reg_pred(x)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)
            # TODO: Test whether use matmul instead of conv can
            #  speed up training.
            bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds


@MODELS.register_module()
class YOLOv8Head(YOLOv5Head):
    """YOLOv8Head head used in `YOLOv8`.

    Args:
        head_module(:obj:`ConfigDict` or dict): Base module used for YOLOv8Head
        prior_generator(dict): Points generator feature maps
            in 2D points-based detectors.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_dfl (:obj:`ConfigDict` or dict): Config of Distribution Focal
            Loss.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            anchor head. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            anchor head. Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 head_module: ConfigType,
                 prior_generator: ConfigType = dict(
                     type='mmdet.MlvlPointGenerator',
                     offset=0.5,
                     strides=[8, 16, 32]),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='none',
                     loss_weight=0.5),
                 loss_bbox: ConfigType = dict(
                     type='IoULoss',
                     iou_mode='ciou',
                     bbox_format='xyxy',
                     reduction='sum',
                     loss_weight=7.5,
                     return_iou=False),
                 loss_dfl=dict(
                     type='mmdet.DistributionFocalLoss',
                     reduction='mean',
                     loss_weight=1.5 / 4),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            head_module=head_module,
            prior_generator=prior_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.loss_dfl = MODELS.build(loss_dfl)
        # yolov8 doesn't need loss_obj
        self.loss_obj = None

        self.v8_assigner = TaskAlignedAssigner(
            topk=10, num_classes=self.num_classes, alpha=0.5, beta=6.0)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.bbox_loss = BboxLoss(
            self.head_module.reg_max - 1, use_dfl=True).cuda()
        self.proj = torch.arange(
            self.head_module.reg_max, dtype=torch.float, device='cuda')

    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)

            # Add common attributes to reduce calculation
            self.featmap_sizes_train = None
            self.num_level_priors = None
            self.flatten_priors_train = None
            self.stride_tensor = None

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            bbox_dist_preds (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """

        # 进行结果对齐时候使用
        # o_preds = torch.load('/data/gulingrui/code/ultralytics/feats.pth')
        # o_targets, o_pred_distri, o_pred_bboxes, o_pred_scores = torch.load(
        #     '/data/gulingrui/code/ultralytics/inputs.pth')
        # o_loss = torch.load('/data/gulingrui/code/ultralytics/loss.pth')
        # print(o_loss)
        #
        # # 准备数据
        # cls_scores = [i[:, -80:] for i in o_preds]
        # # (16, 64, h, w) -> (16, 64, h*w) -> (16, h*w, 64)
        # -> (16, h*w, 4, 16)
        # bbox_dist_preds = [
        #     i[:, :64].view(16, 64, -1).permute(0, 2,
        #     1).view(16, -1, 4,
        #     16).permute(0, 3, 1, 2)
        #     for i in o_preds
        # ]
        # # -> (16, 16, h*w, 4)
        # assert bbox_dist_preds[
        # 0].shape[-1] == 4 and bbox_dist_preds[0].shape[
        #     1] == 16
        # bbox_preds = []
        # for i in bbox_dist_preds:
        #     tempres = F.conv2d(
        #         F.softmax(i.float(), dim=1), self.head_module.proj)
        #     bbox_preds.append(tempres)

        num_imgs = len(batch_img_metas)
        # num_imgs = 16

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(
                mlvl_priors_with_stride, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = self.gt_instances_preprocess(batch_gt_instances, num_imgs)
        # 对齐时候使用
        # gt_info = o_targets
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # (bs, reg_max+1, n, 4) -> (bs, n, 4, reg_max+1)
        flatten_pred_dists = [
            bbox_pred_org.permute(0, 2, 3,
                                  1).reshape(num_imgs, -1,
                                             self.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[..., 0])
        pred_scores = torch.sigmoid(flatten_cls_preds)

        # o_1, o_2, o_3, o_4, o_5, o_6 = torch.load(
        #     '/data/gulingrui/code/ultralytics/assigner.pth')
        # o_assigned_bboxes, o_assigned_scores, o_fg_mask_pre_prior
        # = torch.load(
        #     '/data/gulingrui/code/ultralytics/assignerres.pth')

        assigned_result = self.assigner(flatten_pred_bboxes.detach(),
                                        pred_scores.detach(),
                                        self.flatten_priors_train, gt_labels,
                                        gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

        assigned_scores_sum = assigned_scores.sum()

        loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            # dfl loss
            dist_mask = fg_mask_pre_prior.unsqueeze(-1).repeat(
                [1, 1, self.head_module.reg_max * 4])

            pred_dist_pos = torch.masked_select(
                flatten_dist_preds,
                dist_mask).reshape([-1, 4, self.head_module.reg_max])
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            print('pre', flatten_dist_preds.sum(), dist_mask.sum(),
                  pred_dist_pos.sum())
            loss_dfl = self.loss_dfl(
                pred_dist_pos.reshape(-1, self.head_module.reg_max),
                assigned_ltrb_pos.reshape(-1),
                weight=bbox_weight.expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum)
            # loss_dfl_v8 = self.loss_dfl_v8(
            #     flatten_dist_preds,
            #     self.flatten_priors_train[..., :2] / self.stride_tensor,
            #     assigned_bboxes,
            #     bbox_weight,
            #     prior_bbox_mask,
            #     assigned_scores_sum
            # )

            # target_ltrb = bbox2dist(
            #     self.flatten_priors_train[..., :2] / self.stride_tensor,
            #     assigned_bboxes,
            #     15)
            # fg_mask = prior_bbox_mask[..., 0]
            # loss_dfl_v8 = self._df_loss(flatten_dist_preds[fg_mask].view(-1, 16), target_ltrb[fg_mask]) * bbox_weight # noqa
            # loss_dfl_v8 = loss_dfl_v8.sum() / assigned_scores_sum
            #
            # print('loss_dfl', loss_dfl, loss_dfl_v8)
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * num_imgs * world_size,
            loss_bbox=loss_bbox * num_imgs * world_size,
            loss_dfl=loss_dfl * num_imgs * world_size), dist_mask

    def loss_dfl_v8(self, pred_dist, anchors, target_bboxes, weight, fg_mask,
                    target_scores_sum):
        # weight = torch.masked_select(target_scores.sum(-1),
        # fg_mask).unsqueeze(-1)
        target_ltrb = bbox2dist(anchors, target_bboxes, 15)
        fg_mask = fg_mask[..., 0]
        # print(pred_dist.shape, target_ltrb.shape, fg_mask.shape)
        # pred_dist[fg_mask]
        # target_ltrb[fg_mask]
        loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, 16),
                                 target_ltrb[fg_mask]) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
        return loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(
            tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(
                    tl.shape) * wr).mean(
                        -1, keepdim=True)

    def loss_by_feat_v8(self,
                        cls_scores: Sequence[Tensor],
                        bbox_preds: Sequence[Tensor],
                        bbox_dist_preds: Sequence[Tensor],
                        batch_gt_instances: Sequence[InstanceData],
                        batch_img_metas: Sequence[dict],
                        batch_gt_instances_ignore: OptInstanceList = None,
                        dist_mask_=None) -> dict:

        # 1111 torch.Size([8, 80, 60, 60])
        # 2222 torch.Size([8, 1, 3600, 4])
        # 3333 torch.Size([8, 17, 3600, 4])

        # v8
        # pred_scores [16, 8400, 80]
        # pred_bboxes [16, 8400, 4]
        # pred_distri [16, 8400, 64]
        # view 用的是view(b, a, 4, c // 4), 16在后

        # mmyolo 下面的转成上面的
        # cls_scores [(bs, 80, 80, 80), (bs, 80, 40, 40), (bs, 80, 20, 20)]
        # bbox_preds [(8, 1, 6400, 4), ]
        # bbox_dist_preds [(8, 16, 3600, 4), ]
        # bbox_dist_preds (bs, reg_max, n, 4)

        bs = cls_scores[0].shape[0]
        # device = cls_scores[0].device
        pred_scores_list = []
        for cls_score in cls_scores:
            # (bs, num_classes, 80, 80)
            # -> (bs, num_classes, 6400)
            # -> (bs, 6400, num_classes)
            cls_score1 = cls_score.view(bs, self.num_classes,
                                        -1).permute(0, 2, 1)
            pred_scores_list.append(cls_score1)
        pred_distri_list = []
        for pred_d in bbox_dist_preds:
            # [bs, hw, 4, regmax]
            pred_d1 = pred_d

            # (bs, reg_max, hw, 4) -> (bs, hw, 4, 16)
            # pred_d1 = pred_d.permute(0, 2, 3, 1)
            assert pred_d1.shape[-1] == 16 and pred_d1.shape[-2] == 4
            # (bs, hw, 4, 16) -> (bs, hw, 64)
            pred_d2 = pred_d1.view(bs, -1, 4 * self.head_module.reg_max)
            pred_distri_list.append(pred_d2)

        pred_scores = torch.cat(pred_scores_list, dim=1)
        pred_distri = torch.cat(pred_distri_list, dim=1)
        assert pred_distri.shape[1] == 8400 and pred_scores.shape[1] == 8400

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        # imgsz = torch.tensor(cls_scores[0].shape[2:], device=device, dtype=dtype) * self.self.head_module.featmap_strides[0]  # image size (h,w)   # noqa
        anchor_points, stride_tensor = make_anchors(
            cls_scores, self.head_module.featmap_strides, 0.5)

        # targets
        # targets = torch.cat((batch["batch_idx"].view(-1, 1),
        # batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(batch_gt_instances, batch_size)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points,
                                       pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.v8_assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_bboxes /= stride_tensor
        target_scores_sum = target_scores.sum()

        # cls loss
        loss_cls = self.bce(
            pred_scores,
            target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            loss_bbox, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes,
                                                 anchor_points, target_bboxes,
                                                 target_scores,
                                                 target_scores_sum, fg_mask,
                                                 batch_gt_instances_ignore)
        else:
            loss_bbox = pred_scores.sum() * 0
            loss_dfl = pred_scores.sum() * 0

        loss_bbox *= 7.5
        loss_cls *= 0.5
        loss_dfl *= 1.5

        _, world_size = get_dist_info()

        return dict(
            loss_cls=loss_cls * bs * world_size,
            loss_bbox=loss_bbox * bs * world_size,
            loss_dfl=loss_dfl * bs * world_size)

    def preprocess(self, batch_gt_instances, batch_size):
        # 只支持fast version
        assert isinstance(batch_gt_instances, Tensor)
        i = batch_gt_instances[:, 0]  # image index
        _, counts = i.unique(return_counts=True)
        out = torch.zeros(batch_size, counts.max(), 5, device='cuda')
        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n:
                out[j, :n] = batch_gt_instances[matches, 1:]
            # 不需要mul
            # out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    @staticmethod
    def gt_instances_preprocess(batch_gt_instances: Union[Tensor, Sequence],
                                batch_size: int) -> Tensor:
        """Split batch_gt_instances with batch size, from [all_gt_bboxes, 6]
        to.

        [batch_size, number_gt, 5]. If some shape of single batch smaller than
        gt bbox len, then using [-1., 0., 0., 0., 0.] to fill.

        Args:
            batch_gt_instances (Sequence[Tensor]): Ground truth
                instances for whole batch, shape [all_gt_bboxes, 6]
            batch_size (int): Batch size.

        Returns:
            Tensor: batch gt instances data, shape [batch_size, number_gt, 5]
        """
        if isinstance(batch_gt_instances, Sequence):
            max_gt_bbox_len = max(
                [len(gt_instances) for gt_instances in batch_gt_instances])
            # fill [-1., 0., 0., 0., 0.] if some shape of
            # single batch not equal max_gt_bbox_len
            batch_instance_list = []
            for index, gt_instance in enumerate(batch_gt_instances):
                bboxes = gt_instance.bboxes
                labels = gt_instance.labels
                batch_instance_list.append(
                    torch.cat((labels[:, None], bboxes), dim=-1))

                if bboxes.shape[0] >= max_gt_bbox_len:
                    continue

                fill_tensor = bboxes.new_full(
                    [max_gt_bbox_len - bboxes.shape[0], 5], 0)
                fill_tensor[:, 0] = -1.
                batch_instance_list[index] = torch.cat(
                    (batch_instance_list[-1], fill_tensor), dim=0)

            return torch.stack(batch_instance_list)
        else:
            # faster version
            # sqlit batch gt instance [all_gt_bboxes, 6] ->
            # [batch_size, number_gt_each_batch, 5]
            batch_instance_list = []
            max_gt_bbox_len = 0
            for i in range(batch_size):
                single_batch_instance = \
                    batch_gt_instances[batch_gt_instances[:, 0] == i, :]
                single_batch_instance = single_batch_instance[:, 1:]
                batch_instance_list.append(single_batch_instance)
                if len(single_batch_instance) > max_gt_bbox_len:
                    max_gt_bbox_len = len(single_batch_instance)

            # fill [-1., 0., 0., 0., 0.] if some shape of
            # single batch not equal max_gt_bbox_len
            for index, gt_instance in enumerate(batch_instance_list):
                if gt_instance.shape[0] >= max_gt_bbox_len:
                    continue
                fill_tensor = batch_gt_instances.new_full(
                    [max_gt_bbox_len - gt_instance.shape[0], 5], 0)
                fill_tensor[:, 0] = -1.
                batch_instance_list[index] = torch.cat(
                    (batch_instance_list[index], fill_tensor), dim=0)

            return torch.stack(batch_instance_list)

    def bbox_decode(self, anchor_points, pred_dist):
        # if self.use_dfl:
        b, a, c = pred_dist.shape  # batch, anchors, channels
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
            self.proj.type(pred_dist.dtype))
        # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))     # noqa
        # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)  # noqa
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def loss(self, x: Tuple[Tensor], batch_data_samples: Union[list,
                                                               dict]) -> dict:
        # if isinstance(batch_data_samples, list):
        #     losses = super().loss(x, batch_data_samples)
        # else:
        assert not isinstance(batch_data_samples, list)
        outs = self(x)
        # Fast version
        loss_inputs = outs + (batch_data_samples['bboxes_labels'],
                              batch_data_samples['img_metas'])
        losses, dist_mask = self.loss_by_feat(*loss_inputs)
        loss_inputs = loss_inputs + (dist_mask, )
        losses_v8 = self.loss_by_feat_v8(*loss_inputs)
        print('mmyolo', losses)
        print('v8', losses_v8)

        return losses
