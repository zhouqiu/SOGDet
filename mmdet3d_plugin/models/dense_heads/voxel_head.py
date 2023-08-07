# Copyright (c) https://github.com/FANG-MING/occupancy-for-nuscenes. All rights reserved.
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import HEADS
import numpy as np
from .lovasz_losses import lovasz_softmax



@HEADS.register_module()
class OccFuserHead(nn.Module):
    def __init__(
            self, grid_size, nbr_classes=18,
            in_dims=64, hidden_dims=128, out_dims=None,
            ignore_label=0,loss_weight=1,balance=1,
            weight=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
    ):
        super().__init__()
        self.bev_h = grid_size[0]
        self.bev_w = grid_size[1]
        self.bev_z = grid_size[2]
        # self.scale_h = scale_h
        # self.scale_w = scale_w
        # self.scale_z = scale_z

        out_dims = in_dims if out_dims is None else out_dims

        self.decoder = nn.Sequential(
            nn.Conv3d(in_dims, hidden_dims, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm3d(hidden_dims),
            nn.Conv3d(hidden_dims, out_dims, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_dims),
        )
        self.z_embeding = nn.Embedding(self.bev_z, in_dims)
        self.classifier = nn.Conv3d(out_dims, nbr_classes, 1)
        self.classes = nbr_classes
        self.ignore_label = ignore_label

        self.weight = torch.Tensor(weight)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=self.weight, ignore_index=ignore_label)
        self.lovasz_softmax_loss = lovasz_softmax
        self.loss_weight=loss_weight
        self.balance=balance

    def forward(self, bev_feature):
        bs, c, _, _ = bev_feature.shape
        bev_feature = bev_feature.reshape(bs, c, self.bev_h, self.bev_w)
        bev_feature = bev_feature[..., None] + self.z_embeding.weight.permute(1, 0)[None, :, None, None, :]
        bev_feature = self.decoder(bev_feature)
        logits = self.classifier(bev_feature)
        return logits

    def loss(self, x, target):#B, C, h,w,z
        loss_dict = dict()

        # voxel_loss = self.cross_entropy_loss(x.squeeze(), target.squeeze().long())# val_vox_label.type(torch.LongTensor).cuda()
        voxel_loss = self.cross_entropy_loss(x, target.long())# val_vox_label.type(torch.LongTensor).cuda()
        lovasz_softmax_loss = self.lovasz_softmax_loss(torch.nn.functional.softmax(x, dim=1), target, ignore=self.ignore_label)

        loss_dict['voxel_bev_loss'] = voxel_loss * self.loss_weight * self.balance
        loss_dict['lovasz_softmax_loss'] = lovasz_softmax_loss * self.loss_weight
        return loss_dict

