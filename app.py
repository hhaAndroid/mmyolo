# Copyright (c) OpenMMLab. All rights reserved.
import os

os.system('python -m mim install "mmcv>=2.0.0rc4"')
os.system('python -m mim install mmengine')
os.system('python -m mim install "mmdet>=3.0.0"')
os.system('python -m mim install -e .')

from argparse import ArgumentParser

import gradio as gr
from mmdet.apis import DetInferencer, inference_detector

from mmyolo.utils import switch_to_deploy

model_dict = {
    'RTMDet-L': 'rtmdet_l_syncbn_fast_8xb32-300e_coco',
    'YOLOv5-L': 'yolov5_l-v61_syncbn_fast_8xb16-300e_coco',
    'YOLOv6-L': 'yolov6_l_syncbn_fast_8xb32-300e_coco',
    'YOLOv7-L': 'yolov7_l_syncbn_fast_8x16b-300e_coco',
    'YOLOv8-L': 'yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco',
    'YOLOX-L': 'yolox_l_fast_8xb8-300e_coco',
    'PPYOLOE-L': 'ppyoloe_plus_L_fast_8xb8-80e_coco',
}


def inference(input, model_str):
    parser = ArgumentParser()
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    args = parser.parse_args()

    det_inferencer = DetInferencer(model_dict[model_str], scope='mmyolo')
    switch_to_deploy(det_inferencer.model)

    result = inference_detector(det_inferencer.model, input)

    det_inferencer.visualizer.add_datasample(
        'image',
        input,
        data_sample=result,
        draw_gt=False,
        show=False,
        wait_time=0,
        pred_score_thr=args.score_thr)

    return det_inferencer.visualizer.get_image()


image = gr.Image(type='numpy', label='input')
input_model = gr.inputs.Dropdown(
    choices=list(model_dict.keys()), label='select model', default='RTMDet-L')

gr.Interface(
    fn=inference,
    title='MMYOLO Demo',
    description=
    'Note: The first time running requires downloading the weights, please wait a moment.',  # noqa
    inputs=[image, input_model],
    outputs=gr.Image(type='pil'),
    examples=[['demo/dog.jpg', 'RTMDet-L']]).launch()
