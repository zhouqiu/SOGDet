# Copyright (c) Phigent Robotics. All rights reserved.

import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet.models import NECKS


@NECKS.register_module()
class FPN_LSS(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=4,
                 input_feature_index = (0,2),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None):
        super().__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        # assert norm_cfg['type'] in ['BN', 'SyncBN']
        channels_factor = 2 if self.extra_upsample else 1
        self.conv = nn.Sequential(
            build_conv_layer(conv_cfg, in_channels=in_channels, out_channels=out_channels * channels_factor,
                             kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(conv_cfg, in_channels=out_channels * channels_factor, out_channels=out_channels * channels_factor,
                             kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=extra_upsample , mode='bilinear', align_corners=True),
                build_conv_layer(conv_cfg, in_channels=out_channels * channels_factor, out_channels=out_channels,
                                 kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, in_channels=out_channels, out_channels=out_channels,
                                 kernel_size=1, padding=0),
            )
        self.lateral=  lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=lateral, out_channels=lateral,
                                 kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral, postfix=0)[1],
                nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        x2, x1 = feats[self.input_feature_index[0]], feats[self.input_feature_index[1]]
        if self.lateral:
            x2 = self.lateral_conv(x2)
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        x = self.conv(x1)
        if self.extra_upsample:
            x = self.up2(x)
        return x


# @NECKS.register_module()
# class FPN_LSS_Fusion(FPN_LSS):

#     def forward(self, feats, feat_voxel):
#         x2, x1 = feats[self.input_feature_index[0]
#                        ], feats[self.input_feature_index[1]] #x1:640,16,16 x2:160,64,64
#         if self.lateral:
#             x2 = self.lateral_conv(x2)
#         x1 = x1 + feat_voxel[0]
#         x1 = self.up(x1) #640,64,64
#         x1 = torch.cat([x2, x1], dim=1) #800,64,64
#         x = self.conv(x1) #512, 64,64
#         x = x + feat_voxel[1]
#         if self.extra_upsample:
#             x = self.up2(x)   #256,128,128
#             x = x + feat_voxel[2]
#         return x


    
# @NECKS.register_module()
# class FPN_LSS_Fusion_DiffMomentum(FPN_LSS):
#     def __init__(self, momentum_weight:list = [0.9,0.9,0.9], **kwargs):
#         super(FPN_LSS_Fusion_DiffMomentum, self).__init__(**kwargs)

#         self.momentum_weight = momentum_weight

#     def momentum_add(self, feat1, feat2, layer=0):
#         return feat1 * self.momentum_weight[layer] + feat2 * (1-self.momentum_weight[layer])

#     def forward(self, feats, feat_voxel):
#         x2, x1 = feats[self.input_feature_index[0]
#                        ], feats[self.input_feature_index[1]] #x1:640,16,16 x2:160,64,64
#         if self.lateral:
#             x2 = self.lateral_conv(x2)
#         x1 = self.momentum_add(x1, feat_voxel[0], 0) #x1 + feat_voxel[0]
#         x1 = self.up(x1) #640,64,64
#         x1 = torch.cat([x2, x1], dim=1) #800,64,64
#         x = self.conv(x1) #512, 64,64
#         x = self.momentum_add(x, feat_voxel[1], 1) #x + feat_voxel[1]
#         if self.extra_upsample:
#             x = self.up2(x)   #256,128,128
#             x = self.momentum_add(x, feat_voxel[2], 2) #x + feat_voxel[2]
#         return x


# @NECKS.register_module()
# class FPN_LSS_Fusion_Filter(FPN_LSS):

#     def __init__(self, channel_indices=[512, 512, 256], use_filter=False, **kwargs):
#         super(FPN_LSS_Fusion_Filter, self).__init__(**kwargs)

#         self.channel_indices = channel_indices
#         self.use_filter = use_filter

#         for i in range(len(channel_indices)):
#             filter_conv = nn.Sequential(
#                 nn.Conv2d(channel_indices[i], channel_indices[i], kernel_size=3, padding=1, bias=False),
#                 nn.ReLU(inplace=True))
#             setattr(self, 'filter_conv_{}'.format(i), filter_conv)


#     def forward(self, feats, feat_voxel):
#         x2, x1 = feats[self.input_feature_index[0]
#                        ], feats[self.input_feature_index[1]] #x1:640,16,16 x2:160,64,64
#         if self.lateral:
#             x2 = self.lateral_conv(x2)
#         if self.use_filter:
#             for i in range(len(self.channel_indices)):
#                 feat_voxel[i] = getattr(self, 'filter_conv_{}'.format(i))(feat_voxel[i])
#                 # print("{}:{}".format(i, feat_voxel[i].shape))
#         x1 = x1 + feat_voxel[0]
#         x1 = self.up(x1) #640,64,64
#         x1 = torch.cat([x2, x1], dim=1) #800,64,64
#         x = self.conv(x1) #512, 64,64
#         x = x + feat_voxel[1]
#         if self.extra_upsample:
#             x = self.up2(x)   #256,128,128
#             x = x + feat_voxel[2]
#         return x


# @NECKS.register_module()
# class FPN_LSS_Fusion_Filter_Momentum(FPN_LSS_Fusion_Filter):

#     def __init__(self, momentum_weight=0.9, **kwargs):
#         super(FPN_LSS_Fusion_Filter_Momentum, self).__init__(**kwargs)

#         self.momentum_weight = momentum_weight

#     def momentum_add(self, feat1, feat2):
#         return feat1 * self.momentum_weight + feat2 * (1-self.momentum_weight)

#     def forward(self, feats, feat_voxel):
#         x2, x1 = feats[self.input_feature_index[0]
#                        ], feats[self.input_feature_index[1]] #x1:640,16,16 x2:160,64,64
#         if self.lateral:
#             x2 = self.lateral_conv(x2)
#         if self.use_filter:
#             for i in range(len(self.channel_indices)):
#                 feat_voxel[i] = getattr(self, 'filter_conv_{}'.format(i))(feat_voxel[i])
#                 # print("{}:{}".format(i, feat_voxel[i].shape))
#         x1 = self.momentum_add(x1, feat_voxel[0]) #x1 + feat_voxel[0]
#         x1 = self.up(x1) #640,64,64
#         x1 = torch.cat([x2, x1], dim=1) #800,64,64
#         x = self.conv(x1) #512, 64,64
#         x = self.momentum_add(x, feat_voxel[1]) #x + feat_voxel[1]
#         if self.extra_upsample:
#             x = self.up2(x)   #256,128,128
#             x = self.momentum_add(x, feat_voxel[2]) #x + feat_voxel[2]
#         return x





# @NECKS.register_module()
# class FPN_LSS_Voxel_OnlyOcc(nn.Module):
#     def __init__(self, in_channels, out_channels, scale_factor=4,
#                  input_feature_index=(0, 2),
#                  conv_cfg=None,
#                  norm_cfg=dict(type='BN'),
#                  extra_upsample=2,
#                  lateral=None,
#                  freeze=False,):
#         super().__init__()
#         self.freeze = freeze
#         self.input_feature_index = input_feature_index
#         self.extra_upsample = extra_upsample is not None
#         self.up = nn.Upsample(scale_factor=scale_factor,
#                               mode='bilinear', align_corners=True)
#         # assert norm_cfg['type'] in ['BN', 'SyncBN']
#         channels_factor = 2 if self.extra_upsample else 1
#         self.conv = nn.Sequential(
#             # nn.Conv2d(in_channels, out_channels * channels_factor,
#             #           kernel_size=3, padding=1, bias=False),
#             build_conv_layer(conv_cfg, in_channels=in_channels, out_channels=out_channels * channels_factor,
#                                  kernel_size=3, padding=1, bias=False),
#             build_norm_layer(norm_cfg, out_channels *
#                              channels_factor, postfix=0)[1],
#             nn.ReLU(inplace=True),
#             # nn.Conv2d(out_channels * channels_factor, out_channels * channels_factor,
#             #           kernel_size=3, padding=1, bias=False),
#             build_conv_layer(conv_cfg, in_channels=out_channels * channels_factor, out_channels=out_channels * channels_factor,
#                                  kernel_size=3, padding=1, bias=False),
#             build_norm_layer(norm_cfg, out_channels *
#                              channels_factor, postfix=0)[1],
#             nn.ReLU(inplace=True),
#         )
#         self.up2 = nn.Sequential(
#             nn.Upsample(scale_factor=extra_upsample,
#                         mode='bilinear', align_corners=True),
#             # nn.Conv2d(out_channels * channels_factor, out_channels,
#             #           kernel_size=3, padding=1, bias=False),
#             build_conv_layer(conv_cfg, in_channels=out_channels * channels_factor, out_channels=out_channels,
#                                  kernel_size=3, padding=1, bias=False),
#             build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
#             nn.ReLU(inplace=True),
#         )
#         self.lateral = lateral is not None
#         if self.lateral:
#             self.lateral_conv = nn.Sequential(
#                 # nn.Conv2d(lateral, lateral,
#                 #           kernel_size=1, padding=0, bias=False),
#                 build_conv_layer(conv_cfg, in_channels=lateral, out_channels=lateral,
#                                  kernel_size=1, padding=0, bias=False),
#                 build_norm_layer(norm_cfg, lateral, postfix=0)[1],
#                 nn.ReLU(inplace=True),
#             )
        
#     def forward(self, feats):
#         x2, x1 = feats[self.input_feature_index[0]
#                        ], feats[self.input_feature_index[1]] #x1:640,16,16 x2:160,64,64
#         if self.lateral:
#             x2 = self.lateral_conv(x2)
#         # x_fuse1 = self.conv_fuse1(x1) # 640, 16,16
#         x1_up = self.up(x1) #640,64,64
#         x1_up = torch.cat([x2, x1_up], dim=1) #800,64,64
#         x = self.conv(x1_up)  #512,64,64
#         # x_fuse2 = self.conv_fuse2(x) #512,64,64
#         x_up = self.up2(x) #512,128,128
#         # x_fuse3 = self.conv_fuse3(x_up)

#         return [x_up] 
    
