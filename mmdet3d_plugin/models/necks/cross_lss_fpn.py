import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet.models import NECKS
from .lss_fpn import FPN_LSS

@NECKS.register_module()
class FPN_LSS_Fusion_Momentum(FPN_LSS):
    def __init__(self, momentum_weight=0.9, **kwargs):
        super(FPN_LSS_Fusion_Momentum, self).__init__(**kwargs)

        self.momentum_weight = momentum_weight

    def momentum_add(self, feat1, feat2):
        return feat1 * self.momentum_weight + feat2 * (1-self.momentum_weight)

    def forward(self, feats, feat_voxel):
        x2, x1 = feats[self.input_feature_index[0]
                       ], feats[self.input_feature_index[1]] #x1:640,16,16 x2:160,64,64
        if self.lateral:
            x2 = self.lateral_conv(x2)
        x1 = self.momentum_add(x1, feat_voxel[0]) #x1 + feat_voxel[0]
        x1 = self.up(x1) #640,64,64
        if x1.shape[-1] != x2.shape[-1] and x1.shape[-2] != x2.shape[-2]:
            x1 = x1[...,1:1+x2.shape[-2], 1:1+x2.shape[-1]]
        x1 = torch.cat([x2, x1], dim=1) #800,64,64
        x = self.conv(x1) #512, 64,64
        x = self.momentum_add(x, feat_voxel[1]) #x + feat_voxel[1]
        if self.extra_upsample:
            x = self.up2(x)   #256,128,128
            x = self.momentum_add(x, feat_voxel[2]) #x + feat_voxel[2]
        return x
    
@NECKS.register_module()
class FPN_LSS_Voxel(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=4,
                 input_feature_index=(0, 2),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 freeze=False,
                 use_up=True,):
        super().__init__()
        self.use_up = use_up
        self.freeze = freeze
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.up = nn.Upsample(scale_factor=scale_factor,
                              mode='bilinear', align_corners=True)
        # assert norm_cfg['type'] in ['BN', 'SyncBN']
        channels_factor = 2 if self.extra_upsample else 1
        self.conv = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels * channels_factor,
            #           kernel_size=3, padding=1, bias=False),
            build_conv_layer(conv_cfg, in_channels=in_channels, out_channels=out_channels * channels_factor,
                                 kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels *
                             channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels * channels_factor, out_channels * channels_factor,
            #           kernel_size=3, padding=1, bias=False),
            build_conv_layer(conv_cfg, in_channels=out_channels * channels_factor, out_channels=out_channels * channels_factor,
                                 kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels *
                             channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=extra_upsample,
                        mode='bilinear', align_corners=True),
            # nn.Conv2d(out_channels * channels_factor, out_channels,
            #           kernel_size=3, padding=1, bias=False),
            build_conv_layer(conv_cfg, in_channels=out_channels * channels_factor, out_channels=out_channels,
                                 kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                # nn.Conv2d(lateral, lateral,
                #           kernel_size=1, padding=0, bias=False),
                build_conv_layer(conv_cfg, in_channels=lateral, out_channels=lateral,
                                 kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral, postfix=0)[1],
                nn.ReLU(inplace=True),
            )
        self.conv_fuse1 = nn.Sequential(
            # nn.Conv2d(in_channels//10*8, in_channels//10*8, kernel_size=1, padding=0, bias=False),
            build_conv_layer(conv_cfg, in_channels=in_channels//10*8, out_channels=in_channels//10*8,
                                 kernel_size=1, padding=0, bias=False),
            build_norm_layer(norm_cfg, in_channels//10*8, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        self.conv_fuse2 = nn.Sequential(
            # nn.Conv2d(out_channels * channels_factor, out_channels *
            #           channels_factor, kernel_size=1, padding=0, bias=False),
            build_conv_layer(conv_cfg, in_channels=out_channels * channels_factor, out_channels=out_channels * channels_factor,
                                 kernel_size=1, padding=0, bias=False),
            build_norm_layer(norm_cfg, out_channels *
                             channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        self.conv_fuse3 = nn.Sequential(
            # nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0,
            #           bias=False),
            build_conv_layer(conv_cfg, in_channels=out_channels, out_channels=out_channels,
                                 kernel_size=1, padding=0, bias=False),
            build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
            nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        x2, x1 = feats[self.input_feature_index[0]
                       ], feats[self.input_feature_index[1]] #x1:640,16,16 x2:160,64,64
        if self.lateral:
            x2 = self.lateral_conv(x2)
        x_fuse1 = self.conv_fuse1(x1) # 640, 16,16
        x1_up = self.up(x1) #640,64,64
        #for reso0512
        if x1_up.shape[-1] != x2.shape[-1] and x1_up.shape[-2] != x2.shape[-2]:
            x1_up = x1_up[...,1:x2.shape[-2]+1,1:x2.shape[-1]+1]
        x1_up = torch.cat([x2, x1_up], dim=1) #800,64,64
        x = self.conv(x1_up)  #512,64,64
        x_fuse2 = self.conv_fuse2(x) #512,64,64
        x_up = self.up2(x) #512,128,128
        x_fuse3 = self.conv_fuse3(x_up)
        if not self.use_up:
            return [x_fuse1, x_fuse2, x_fuse3] #, x_up
        return [x_fuse1, x_fuse2, x_fuse3, x_up] 

@NECKS.register_module()
class Cross_FPN_LSS(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=[4,2],
                 input_feature_index = (0,2),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 momentum_weight=0.9, ):
        super().__init__()
        self.input_feature_index = input_feature_index
        self.up = nn.Upsample(scale_factor=scale_factor[0], mode='bilinear', align_corners=True)
        
        channels_factor = 2 
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
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor[1] , mode='bilinear', align_corners=True),
            build_conv_layer(conv_cfg, in_channels=out_channels * channels_factor, out_channels=out_channels,
                            kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(conv_cfg, in_channels=out_channels, out_channels=out_channels,
                            kernel_size=1, padding=0),
            )
        
        self.conv_occ = nn.Sequential(
            build_conv_layer(conv_cfg, in_channels=in_channels, out_channels=out_channels * channels_factor,
                             kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(conv_cfg, in_channels=out_channels * channels_factor, out_channels=out_channels * channels_factor,
                             kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        self.up2_occ = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor[1] , mode='bilinear', align_corners=True),
            build_conv_layer(conv_cfg, in_channels=out_channels * channels_factor, out_channels=out_channels,
                            kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(conv_cfg, in_channels=out_channels, out_channels=out_channels,
                            kernel_size=1, padding=0),
            )
        
        self.conv_fuse1 = nn.Sequential(
            build_conv_layer(conv_cfg, in_channels=out_channels * channels_factor, out_channels=out_channels * channels_factor,
                                 kernel_size=1, padding=0, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        self.conv_fuse2 = nn.Sequential(
            build_conv_layer(conv_cfg, in_channels=out_channels * channels_factor, out_channels=out_channels * channels_factor,
                                 kernel_size=1, padding=0, bias=False),
            build_norm_layer(norm_cfg, out_channels *
                             channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        self.conv_fuse3 = nn.Sequential(
            build_conv_layer(conv_cfg, in_channels=out_channels, out_channels=out_channels,
                                 kernel_size=1, padding=0, bias=False),
            build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
            nn.ReLU(inplace=True),
        )

        self.conv_fuse1_occ = nn.Sequential(
            build_conv_layer(conv_cfg, in_channels=out_channels * channels_factor, out_channels=out_channels * channels_factor,
                                 kernel_size=1, padding=0, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        self.conv_fuse2_occ = nn.Sequential(
            build_conv_layer(conv_cfg, in_channels=out_channels * channels_factor, out_channels=out_channels * channels_factor,
                                 kernel_size=1, padding=0, bias=False),
            build_norm_layer(norm_cfg, out_channels *
                             channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        self.conv_fuse3_occ = nn.Sequential(
            build_conv_layer(conv_cfg, in_channels=out_channels, out_channels=out_channels,
                                 kernel_size=1, padding=0, bias=False),
            build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
            nn.ReLU(inplace=True),
        )

        self.momentum_weight = momentum_weight

    def momentum_add(self, feat1, feat2):
        return feat1 * self.momentum_weight + feat2 * (1-self.momentum_weight)



    def forward(self, featsOD, featsOCC):
        
        x2, x1 = featsOD[self.input_feature_index[0]], featsOD[self.input_feature_index[1]]
        y2, y1 = featsOCC[self.input_feature_index[0]], featsOCC[self.input_feature_index[1]]
        #1
        x_fuse1 = self.conv_fuse1(x1) 
        y_fuse1 = self.conv_fuse1_occ(y1) 
        #2
        x1 = self.momentum_add(x1, y_fuse1)
        y1 = self.momentum_add(y1, x_fuse1)
        #3
        x1 = self.up(x1)
        y1 = self.up(y1)
        #4
        x1 = torch.cat([x2, x1], dim=1)
        y1 = torch.cat([y2, y1], dim=1)
        #5
        x = self.conv(x1)
        y = self.conv_occ(y1)
        #6
        x_fuse2 = self.conv_fuse2(x)
        y_fuse2 = self.conv_fuse2_occ(y)
        #7
        x = self.momentum_add(x, y_fuse2)
        y = self.momentum_add(y, x_fuse2)
        #8
        x = self.up2(x)
        y = self.up2_occ(y)
        #9
        x_fuse3 = self.conv_fuse3(x)
        y_fuse3 = self.conv_fuse3_occ(y)
        #10
        x = self.momentum_add(x, y_fuse3)
        y = self.momentum_add(y, x_fuse3)
        return x, y
