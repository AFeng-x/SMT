_base_ = [
    './mask_rcnn_ressa_fpn.py','../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


optimizer = dict(type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.)}))

data = dict(
    samples_per_gpu=2, workers_per_gpu=2,)