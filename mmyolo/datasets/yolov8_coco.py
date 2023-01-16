from .v5augmentations import Albumentations, random_perspective, letterbox, augment_hsv
import mmengine
import copy
import math
import random

import cv2
import mmcv
import numpy as np
import torch

from mmyolo.registry import DATASETS
from mmdet.datasets import CocoDataset


def clip_boxes(boxes, shape):
    """
    > It takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the
    shape

    Args:
      boxes: the bounding boxes to clip
      shape: the shape of the image
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    """
    > It takes in a list of bounding boxes, and returns a list of bounding boxes, but with the x and y
    coordinates normalized to the width and height of the image

    Args:
      x: the bounding box coordinates
      w: width of the image. Defaults to 640
      h: height of the image. Defaults to 640
      clip: If True, the boxes will be clipped to the image boundaries. Defaults to False
      eps: the minimum value of the box's width and height.

    Returns:
      the xywhn format of the bounding boxes.
    """
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """
    > It converts the normalized coordinates to the actual coordinates [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right

    Args:
      x: the bounding box coordinates
      w: width of the image. Defaults to 640
      h: height of the image. Defaults to 640
      padw: padding width. Defaults to 0
      padh: height of the padding. Defaults to 0

    Returns:
      the xyxy coordinates of the bounding box.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    """
    > It converts normalized segments into pixel segments of shape (n,2)

    Args:
      x: the normalized coordinates of the bounding box
      w: width of the image. Defaults to 640
      h: height of the image. Defaults to 640
      padw: padding width. Defaults to 0
      padh: padding height. Defaults to 0

    Returns:
      the x and y coordinates of the top left corner of the bounding box.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * x[..., 0] + padw  # top left x
    y[..., 1] = h * x[..., 1] + padh  # top left y
    return y


@DATASETS.register_module()
class YOLOv8CocoDataset(CocoDataset):

    def __init__(self,
                 *args,
                 img_size=640,
                 batch_size=1,
                 stride=32,
                 pad=0.0,
                 file_client_args=None,
                 albu=True,
                 **kwargs):
        super(YOLOv8CocoDataset, self).__init__(*args, **kwargs)
        self.img_size = img_size
        self.indices = range(len(self))
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.hyp = {
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        }
        if albu:
            self.albumentations = Albumentations()
        else:
            self.albumentations = None

        self.file_client = None
        if file_client_args is not None:
            self.file_client = mmengine.FileClient(**file_client_args)

    def prepare_img(self, idx):
        """Get training data and annotations after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        results = copy.deepcopy(self.data_list[idx])
        return self.pipeline(results)

    def _train(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = True
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            # shapes = None

            # MixUp augmentation
            # if random.random() < hyp['mixup']:
            #     img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))
        else:
            # Load image
            # TODO
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels = random_perspective(img,
                                                 labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        if True:
            # Albumentations
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout

        if nl:
            labels[:, 1:5] = xywhn2xyxy(labels[:, 1:5])

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img, labels

    def load_image(self, index):
        # loads 1 image from dataset, returns img, original hw, resized hw
        path = self.data_list[index]['img_path']

        if self.file_client:
            img_bytes = self.file_client.get(path)
            img = mmcv.imfrombytes(img_bytes)
        else:
            img = cv2.imread(path)  # BGR

        if img is None:
            print('Image Not Found ' + path)
            return None, (0, 0), (0, 0)

        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img = cv2.resize(img, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=cv2.INTER_LINEAR)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

    def load_mosaic(self, index):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            img, (orig_h, orig_w), (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            gt_bboxes = []
            gt_bboxes_labels = []
            for instance in self.data_list[index]['instances']:
                if instance['ignore_flag'] == 0:
                    gt_bboxes.append(instance['bbox'])
                    gt_bboxes_labels.append(instance['bbox_label'])
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_bboxes_labels = np.array(gt_bboxes_labels)
            bboxes = gt_bboxes  # xyxy
            labels = gt_bboxes_labels

            # 对 bbox 进行处理
            if bboxes.shape[0] > 0:
                bboxes[:, 0::2] *= w / orig_w
                bboxes[:, 1::2] *= h / orig_h
                bboxes[:, 0::2] += padw
                bboxes[:, 1::2] += padh
            else:
                bboxes = np.empty((0, 4))

            labels = np.concatenate((labels[:, None], bboxes), axis=1)
            segments = []

            # if labels.size:
            #     labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            #     segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate

        # Augment
        # img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
        img4, labels4 = random_perspective(img4,
                                           labels4,
                                           segments4,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

        return img4, labels4

    # @staticmethod
    # def collate_fn(batch):
    #     # YOLOv8 collate function, outputs dict
    #     im, label, path, shapes = zip(*batch)  # transposed
    #     for i, lb in enumerate(label):
    #         lb[:, 0] = i  # add target image index for build_targets()
    #     batch_idx, cls, bboxes = torch.cat(label, 0).split((1, 1, 4), dim=1)
    #     return {
    #         'ori_shape': tuple((x[0] if x else None) for x in shapes),
    #         'resized_shape': tuple(tuple(x.shape[1:]) for x in im),
    #         'im_file': path,
    #         'img': torch.stack(im, 0),
    #         'cls': cls,
    #         'bboxes': bboxes,
    #         'batch_idx': batch_idx.view(-1)}


    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_img(idx)
        else:
            img, labels = self._train(idx)

            nl = len(labels)
            labels_out = torch.zeros((nl, 6))
            if nl:
                labels_out[:, 1:] = torch.from_numpy(labels)

            # Convert
            # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            # img = np.ascontiguousarray(img)
            # print(labels_out.shape)

            return torch.from_numpy(img), labels_out

            # label 是归一化的 cxcywh 坐标，需要转化为 xyxy
            # labels[:, 1:] = xywhn2xyxy(labels[:, 1:], img.shape[0],
            #                            img.shape[1])
            #
            # results = {
            #     'img': img,
            #     'gt_bboxes': labels[:, 1:].astype(np.float),
            #     'gt_bboxes_labels': labels[:, 0].astype(np.int),
            #     'img_shape': img.shape,
            #     'img_id': self.data_list[idx]['img_id'],
            #     'img_path': self.data_list[idx]['img_path'],
            #     'ori_height': 2,
            #     'ori_width': 2,
            #     'scale_factor': 1,
            #     'flip': 1,
            #     'flip_direction': 1,
            # }
            # return self.pipeline(results)




