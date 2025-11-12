"""
MMAction2 configuration for thermal action detection with SlowFast model.
"""

# Model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,  # We'll add pretrained weights later
        resample_rate=8,  # Non-local res-sampling rate
        speed_ratio=8,  # Speed ratio for fast pathway
        channel_ratio=8,  # Channel ratio for fast pathway
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            spatial_strides=(1, 2, 2, 2)),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            spatial_strides=(1, 2, 2, 2))),
    cls_head=dict(
        type='SlowFastHead',
        in_channels=2304,  # SlowFast feature dimension
        num_classes=14,  # Thermal action classes (only use first 14)
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[15., 15., 15.],  # Thermal mean temperature (~15°C room temp)
        std=[10., 10., 10.],  # Thermal std (~10°C variation)
        format_shape='NCTHW'))

# Dataset settings
dataset_type = 'ThermalActionDataset'
data_root = 'ThermalDataGen/thermal_action_dataset'
hdf5_root = 'ThermalDataGen/thermal_action_dataset/frames'
ann_file_train = 'ThermalDataGen/thermal_action_dataset/annotations/train.json'
ann_file_val = 'ThermalDataGen/thermal_action_dataset/annotations/val.json'

# Thermal video has 40x60 resolution
img_norm_cfg = dict(
    mean=[15., 15., 15.],  # Thermal temperature mean
    std=[10., 10., 10.],  # Thermal temperature std
    to_bgr=False)  # Already in RGB-like format

train_pipeline = [
    dict(type='DecordInit', num_threads=4),
    dict(type='SampleFrames', clip_len=64, frame_interval=1, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(384, 256)),  # Resize thermal 40x60 -> 256x384
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit', num_threads=4),
    dict(type='SampleFrames', clip_len=64, frame_interval=1, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(384, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        hdf5_root=hdf5_root,
        pipeline=train_pipeline,
        num_classes=14))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        hdf5_root=hdf5_root,
        pipeline=val_pipeline,
        test_mode=True,
        num_classes=14))

test_dataloader = val_dataloader

# Evaluation settings
val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

# Optimizer settings
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=40, norm_type=2))

# Learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[25, 40],
        gamma=0.1)
]

# Training settings
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,
    val_begin=1,
    val_interval=5)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Runtime settings
default_scope = 'mmaction'
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best='auto',
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

log_level = 'INFO'
load_from = None  # Path to pretrained checkpoint
resume = False
work_dir = './work_dirs/thermal_slowfast'

