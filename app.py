# Copyright (c) OpenMMLab. All rights reserved.
import os

os.system('python -m mim install "mmcv>=2.0.0rc4"')
os.system('python -m mim install mmengine')
os.system('python -m mim install "mmdet>=3.0.0"')
os.system('python -m mim install -e .')

from argparse import ArgumentParser

import gradio as gr
from mmdet.apis import inference_detector, init_detector

from mmyolo.registry import VISUALIZERS
from mmyolo.utils import switch_to_deploy


def inference(input):
    parser = ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco.py',
        help='Config file')
    parser.add_argument(
        '--checkpoint',
        default=
        'https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco/rtmdet_l_syncbn_fast_8xb32-300e_coco_20230102_135928-ee3abdc4.pth',
        help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    args = parser.parse_args()

    model = init_detector(args.config, args.checkpoint, device=args.device)
    switch_to_deploy(model)

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    result = inference_detector(model, input)

    visualizer.add_datasample(
        'image',
        input,
        data_sample=result,
        draw_gt=False,
        show=False,
        wait_time=0,
        pred_score_thr=args.score_thr)

    return visualizer.get_image()


gr.Interface(
    fn=inference,
    title='MMYOLO Demo',
    description=
    'Note: The first time running requires downloading the weights, please wait a moment.',  # noqa
    inputs=gr.Image(type='numpy'),
    outputs=gr.Image(type='pil'),
    examples=['demo/dog.jpg']).launch()
