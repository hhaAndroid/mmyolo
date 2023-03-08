# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from mmdet.models.utils import mask2ndarray
from mmdet.structures.bbox import BaseBoxes
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar

from mmyolo.registry import DATASETS, VISUALIZERS


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register all modules in mmdet into the registries
    init_default_scope('mmyolo')

    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo

    progress_bar = ProgressBar(len(dataset))
    for i in range(len(dataset)):
        item = dataset[i]
        img = item['inputs'].permute(1, 2, 0).numpy()

        data_sample = item['data_samples']
        gt_instances = data_sample.gt_instances
        assert not ('masks' in gt_instances and
                    'gt_panoptic_seg' in data_sample), \
            'masks and gt_panoptic_seg are not allowed to exist ' \
            'at the same time.'

        if 'masks' in gt_instances and gt_instances.masks.shape[
                1:] != img.shape[:2]:
            masks = F.interpolate(
                gt_instances.masks[None].float(),
                size=img.shape[:2],
                mode='bilinear',
                align_corners=False)[0]
            gt_instances.masks = masks

        if 'gt_panoptic_seg' in data_sample:
            pan_seg = data_sample.gt_panoptic_seg.pan_seg
            max_len = torch.arange(torch.max(pan_seg)) + 1
            gt_mask = torch.where(pan_seg == max_len.view(-1, 1, 1), 1.0, 0.0)
            if pan_seg.shape[1:] != img.shape[:2]:
                gt_mask = F.interpolate(
                    gt_mask[None],
                    size=img.shape[:2],
                    mode='bilinear',
                    align_corners=False)[0]
            gt_instances.masks = gt_mask
            del data_sample.gt_panoptic_seg

        gt_bboxes = gt_instances.get('bboxes', None)
        if gt_bboxes is not None and isinstance(gt_bboxes, BaseBoxes):
            gt_instances.bboxes = gt_bboxes.tensor

        gt_masks = gt_instances.get('masks', None)
        if gt_masks is not None:
            masks = mask2ndarray(gt_masks)
            gt_instances.masks = masks.astype(bool)

        data_sample.gt_instances = gt_instances

        img_path = osp.basename(item['data_samples'].img_path)
        out_file = osp.join(
            args.output_dir,
            osp.basename(img_path)) if args.output_dir is not None else None

        img = img[..., [2, 1, 0]]  # bgr to rgb

        visualizer.add_datasample(
            osp.basename(img_path),
            img,
            data_sample.numpy(),
            draw_pred=False,
            show=not args.not_show,
            wait_time=args.show_interval,
            out_file=out_file)

        progress_bar.update()


if __name__ == '__main__':
    main()