_base_ = './_base_/models/unet_r50.py'
# Используем базовая конфигурация для U-Net с ResNet-50

model = dict(
    type='UNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    decode_head=dict(
        type='FCNHead',
        in_channels=1024,
        channels=256,
        num_classes=3,
        loss_decode=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
        activation='relu'),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        channels=256,
        num_classes=3,
        loss_decode=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=0.4),
        activation='relu')
)

# Гиперпараметры
batch_size = 4
learning_rate = 0.001
epochs = 50
optimizer = dict(type='Adam', lr=learning_rate)
