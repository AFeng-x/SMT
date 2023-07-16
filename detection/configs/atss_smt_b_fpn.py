# model settings
model = dict(
    type='ATSS',
    pretrained=None,
    backbone=dict(
        type='SMT',
        embed_dims=[64, 128, 256, 512], 
        ca_num_heads=[4, 4, 4, -1], 
        sa_num_heads=[-1, -1, 8, 16],
        mlp_ratios=[8, 6, 4, 2], 
        qkv_bias=True,
        drop_path_rate=0.3,
        depths=[4, 6, 28, 2],
        ca_attentions=[1, 1, 1, 0],
        num_stages=4,
        head_conv=7,
        expand_ratio=2,
        init_cfg=dict(type='Pretrained', checkpoint='path/to/ckpt.pth'),
        ),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='ATSSHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))