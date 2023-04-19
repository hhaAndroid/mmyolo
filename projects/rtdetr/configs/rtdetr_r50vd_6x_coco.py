_base_ = '../../../configs/_base_/default_runtime.py'

model = dict(
    type='RTDetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=None),
    backbone=dict(
        type='mmdet.ResNetV1d',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
       type='HybridEncoder'
    )

)