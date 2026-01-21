crop_size = (
    256,
    256,
)
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
        123.675,
        116.28,
        103.53,
    ],
    size=(
        256,
        256,
    ),
    std=[
        58.395,
        57.12,
        57.375,
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = '/home/dell/gitrepos/MdaCD/Dataset/LEVIRCD'
dataset_type = 'TXTCDDatasetJSON'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=10000,
        save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=10, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw=True, type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
find_unused_parameters = True
launcher = 'pytorch'
load_from = 'work_dirs/MdaCD_LEVIRCD_3/best_mIoU_iter_21000.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
metainfo = dict(
    classes=(
        'background',
        'building',
    ),
    palette=[
        [
            0,
            0,
            0,
        ],
        [
            255,
            255,
            255,
        ],
    ])
model = dict(
    backbone=dict(
        input_resolution=512,
        layers=[
            3,
            4,
            6,
            3,
        ],
        output_dim=1024,
        style='pytorch',
        type='CLIPResNetWithAttention'),
    context_decoder=dict(
        context_length=16,
        dropout=0.1,
        outdim=1024,
        style='pytorch',
        transformer_heads=4,
        transformer_layers=3,
        transformer_width=256,
        type='ContextDecoder',
        visual_dim=1024),
    context_length=64,
    decode_head=dict(
        align_corners=False,
        channels=256,
        dropout_ratio=0.1,
        feature_strides=[
            4,
            8,
            16,
            32,
        ],
        in_channels=[
            256,
            256,
            256,
            256,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        type='SwinTextDecode'),
    neck=dict(
        in_channels=[
            512,
            1024,
            2048,
            4100,
        ],
        num_outs=4,
        out_channels=256,
        type='FPN'),
    pretrained='/home/dell/gitrepos/MdaCD/RN50.pt',
    test_cfg=dict(mode='whole'),
    text_encoder=dict(
        context_length=77,
        embed_dim=1024,
        style='pytorch',
        transformer_heads=8,
        transformer_layers=12,
        transformer_width=512,
        type='CLIPTextContextEncoder'),
    text_head=False,
    train_cfg=dict(),
    type='AugCD')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0))),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1500, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1500,
        by_epoch=False,
        end=30000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='test.txt',
        data_prefix=dict(
            img_path='test/A', img_path2='test/B', seg_map_path='test/label'),
        data_root='/home/dell/gitrepos/MdaCD/Dataset/LEVIRCD',
        metainfo=dict(
            classes=(
                'background',
                'building',
            ),
            palette=[
                [
                    0,
                    0,
                    0,
                ],
                [
                    255,
                    255,
                    255,
                ],
            ]),
        pipeline=[
            dict(type='LoadMultipleRSImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='ConcatCDInput'),
            dict(type='PackCDInputs'),
        ],
        type='TXTCDDatasetJSON'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadMultipleRSImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='ConcatCDInput'),
    dict(type='PackCDInputs'),
]
train_cfg = dict(max_iters=30000, type='IterBasedTrainLoop', val_interval=1000)
train_dataloader = dict(
    batch_size=20,
    dataset=dict(
        ann_file='train.txt',
        data_prefix=dict(
            img_path='train/A',
            img_path2='train/B',
            seg_map_path='train/label'),
        data_root='/home/dell/gitrepos/MdaCD/Dataset/LEVIRCD',
        metainfo=dict(
            classes=(
                'background',
                'building',
            ),
            palette=[
                [
                    0,
                    0,
                    0,
                ],
                [
                    255,
                    255,
                    255,
                ],
            ]),
        pipeline=[
            dict(type='LoadMultipleRSImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(degree=180, prob=0.5, type='MultiImgRandomRotate'),
            dict(
                cat_max_ratio=0.75,
                crop_size=(
                    256,
                    256,
                ),
                type='MultiImgRandomCrop'),
            dict(direction='horizontal', prob=0.5, type='MultiImgRandomFlip'),
            dict(direction='vertical', prob=0.5, type='MultiImgRandomFlip'),
            dict(
                brightness_delta=10,
                contrast_range=(
                    0.8,
                    1.2,
                ),
                hue_delta=10,
                saturation_range=(
                    0.8,
                    1.2,
                ),
                type='MultiImgPhotoMetricDistortion'),
            dict(type='ConcatCDInput'),
            dict(type='PackCDInputs'),
        ],
        type='TXTCDDatasetJSON'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadMultipleRSImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(degree=180, prob=0.5, type='MultiImgRandomRotate'),
    dict(
        cat_max_ratio=0.75, crop_size=(
            256,
            256,
        ), type='MultiImgRandomCrop'),
    dict(direction='horizontal', prob=0.5, type='MultiImgRandomFlip'),
    dict(direction='vertical', prob=0.5, type='MultiImgRandomFlip'),
    dict(
        brightness_delta=10,
        contrast_range=(
            0.8,
            1.2,
        ),
        hue_delta=10,
        saturation_range=(
            0.8,
            1.2,
        ),
        type='MultiImgPhotoMetricDistortion'),
    dict(type='ConcatCDInput'),
    dict(type='PackCDInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(type='LoadMultipleRSImageFromFile'),
    dict(
        transforms=[
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='ConcatCDInput'),
            ],
            [
                dict(type='PackCDInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='val.txt',
        data_prefix=dict(
            img_path='test/A', img_path2='test/B', seg_map_path='test/label'),
        data_root='/home/dell/gitrepos/MdaCD/Dataset/LEVIRCD',
        metainfo=dict(
            classes=(
                'background',
                'building',
            ),
            palette=[
                [
                    0,
                    0,
                    0,
                ],
                [
                    255,
                    255,
                    255,
                ],
            ]),
        pipeline=[
            dict(type='LoadMultipleRSImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='ConcatCDInput'),
            dict(type='PackCDInputs'),
        ],
        type='TXTCDDatasetJSON'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    save_dir='work_dirs/MdaCD_LEVIRCD_3/vis_result',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/MdaCD_LEVIRCD'
