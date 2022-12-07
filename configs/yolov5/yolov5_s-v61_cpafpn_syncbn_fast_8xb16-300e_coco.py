_base_ = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

model = dict(neck=dict(type='YOLOv5CPAFPN'))



