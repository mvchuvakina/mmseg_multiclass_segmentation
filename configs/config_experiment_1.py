_base_ = './_base_/models/deeplabv3plus_r50-d8.py'
# Используем базовая конфигурация для DeepLabV3+ с ResNet-50

model = dict(
    type='EncoderDecoder',
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
        type='DeepLabV3PlusHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        num_classes=3,
        loss_decode=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
        activation='relu'),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_classes=3,
        loss_decode=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=0.4),
        activation='relu')
)

# Гиперпараметры
batch_size = 8
learning_rate = 0.0001
epochs = 30
optimizer = dict(type='Adam', lr=learning_rate)
