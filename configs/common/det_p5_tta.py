tta_model = dict(
    type='mmdet.DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.65), max_per_img=100))

img_scales = [(640, 640), (320, 320), (960, 960)]

#                                LoadImageFromFile
#                     /                 |                     \
# (RatioResize,LetterResize) (RatioResize,LetterResize) (RatioResize,LetterResize) # noqa
#        /      \                    /      \                    /        \
#  RandomFlip RandomFlip      RandomFlip RandomFlip        RandomFlip RandomFlip # noqa
#      |          |                |         |                  |         |
#  LoadAnn    LoadAnn           LoadAnn    LoadAnn           LoadAnn    LoadAnn
#      |          |                |         |                  |         |
#  PackDetIn  PackDetIn         PackDetIn  PackDetIn        PackDetIn  PackDetIn # noqa

tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[[[
            dict(type='YOLOv5KeepRatioResize', scale=s),
            dict(
                type='LetterResize',
                scale=s,
                allow_scale_up=False,
                pad_val=dict(img=114))
        ] for s in img_scales],
                    [
                        dict(type='mmdet.RandomFlip', prob=1.),
                        dict(type='mmdet.RandomFlip', prob=0.)
                    ], [dict(type='mmdet.LoadAnnotations', with_bbox=True)],
                    [
                        dict(
                            type='mmdet.PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor',
                                       'pad_param', 'flip', 'flip_direction'))
                    ]])
]
