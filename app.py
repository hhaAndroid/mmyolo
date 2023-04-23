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
from mmyolo.utils.misc import get_file_list


mmyolo_det_dict = {
    'RTMDet-S': 'rtmdet_s_syncbn_fast_8xb32-300e_coco',
    'YOLOv5-S': 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco',
    'YOLOv6-S': 'yolov6_s_syncbn_fast_8xb32-400e_coco',
    'YOLOv7-T': 'yolov7_tiny_syncbn_fast_8x16b-300e_coco',
    'YOLOv8-S': 'yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco',
    'YOLOX-S': 'yolox_s_fast_8xb8-300e_coco',
    'PPYOLOE-S': 'ppyoloe_plus_s_fast_8xb8-80e_coco'
}
mmdet_det_dict = {
    'RetinaNet-50': 'retinanet_r50-caffe_fpn_1x_coco',
    'FasterRCNN-50': 'faster-rcnn_r50-caffe_fpn_1x_coco',
    'DINO': 'dino-5scale_swin-l_8xb2-12e_coco.py',
}
mmdet_ins_dict = {
    'MaskRCNN-50': 'mask-rcnn_r50-caffe_fpn_1x_coco',
    'SOLOv2-50': 'solov2_r50_fpn_1x_coco',
}

mmdet_pan_dict = {
    'PanopticFPN-50': 'panoptic_fpn_r50_fpn_1x_coco',
    'Mask2Former-Swin':
    'mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco-panoptic',
}

DEFAULT_MMDet_Det = 'FasterRCNN-50'
DEFAULT_MMYOLO_Det = 'RTMDet-S'
DEFAULT_MMDet_Ins = 'MaskRCNN-50'
DEFAULT_MMDet_PAN = 'PanopticFPN-50'

merged_dict = {}
merged_dict.update(mmyolo_det_dict)
merged_dict.update(mmdet_det_dict)
merged_dict.update(mmdet_ins_dict)
merged_dict.update(mmdet_pan_dict)


def update_mmdet_model_name(model_type: str) -> dict:
    if model_type == 'detection':
        model_dict = mmdet_det_dict
        default = DEFAULT_MMDet_Det
    elif model_type == 'instance_segmentation':
        model_dict = mmdet_ins_dict
        default = DEFAULT_MMDet_Ins
    else:
        model_dict = mmdet_pan_dict
        default = DEFAULT_MMDet_PAN
    model_names = list(model_dict.keys())
    return gr.Dropdown.update(choices=model_names, value=default)


def set_example_image(example: list) -> dict:
    return gr.Image.update(value=example[0])


def inference(input, model_str):
    parser = ArgumentParser()
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    args = parser.parse_args()

    if model_str in mmyolo_det_dict:
        scope = 'mmyolo'
    else:
        scope = 'mmdet'

    det_inferencer = DetInferencer(merged_dict[model_str], scope=scope)
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


DESCRIPTION = '''# MMDetection & MMYOLO
<div align="center">
<img src="https://user-images.githubusercontent.com/45811724/190993591-bd3f1f11-1c30-4b93-b5f4-05c9ff64ff7f.gif" width="50%"/>
</div>

#### This is an official demo for MMDet and MMYOLO. \n

Note: The first time running requires downloading the weights, please wait a moment.
'''

with gr.Blocks() as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tab('MMDet Demo'):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(label='Input Image', type='numpy')
                with gr.Group():
                    with gr.Row():
                        model_type = gr.Radio([
                            'detection', 'instance_segmentation',
                            'panoptic_segmentation'
                        ],
                                              value='detection',
                                              label='task Type')
                    with gr.Row():
                        model_name = gr.Dropdown(
                            list(list(mmdet_det_dict.keys())),
                            value=DEFAULT_MMDet_Det,
                            label='Model')
                with gr.Row():
                    run_button = gr.Button(value='Run')
            with gr.Column():
                with gr.Row():
                    visualization = gr.Image(label='Result', type='numpy')

        with gr.Row():
            paths, _ = get_file_list('demo')
            example_images = gr.Dataset(
                components=[input_image], samples=[[path] for path in paths])

        model_type.change(
            fn=update_mmdet_model_name, inputs=model_type, outputs=model_name)

        run_button.click(
            fn=inference,
            inputs=[input_image, model_name],
            outputs=[
                visualization,
            ])
        example_images.click(
            fn=set_example_image, inputs=example_images, outputs=input_image)

    with gr.Tab('MMYOLO Demo'):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(label='Input Image', type='numpy')
                with gr.Row():
                    model_name = gr.Dropdown(
                        list(list(mmyolo_det_dict.keys())),
                        value=DEFAULT_MMYOLO_Det,
                        label='Model')
                with gr.Row():
                    run_button = gr.Button(value='Run')
            with gr.Column():
                visualization = gr.Image(label='Result', type='numpy')

        with gr.Row():
            paths, _ = get_file_list('demo')
            example_images = gr.Dataset(
                components=[input_image], samples=[[path] for path in paths])

        run_button.click(
            fn=inference,
            inputs=[input_image, model_name],
            outputs=[
                visualization,
            ])
        example_images.click(
            fn=set_example_image, inputs=example_images, outputs=input_image)

demo.queue().launch(show_api=False)
