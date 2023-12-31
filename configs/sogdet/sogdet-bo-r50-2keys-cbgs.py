# Copyright (c) Phigent Robotics. All rights reserved.
_base_ = ['../_base_/default_runtime.py']

plugin = True
plugin_dir = 'mmdet3d_plugin/'

# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config={
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test':0.04,
}

# Model
grid_config={
        'xbound': [-51.2, 51.2, 0.8],
        'ybound': [-51.2, 51.2, 0.8],
        'zbound': [-10.0, 10.0, 20.0],
        'dbound': [1.0, 60.0, 1.0],}

voxel_size = [0.1, 0.1, 0.2]
voxel_grid_num = [128, 128, 10]

learning_map = {
                1: 0,   5: 0,   7: 0,   8: 0,
                10: 0,  11: 0,  13: 0,  19: 0,
                20: 0,  0: 0,   29: 0,  31: 0,
                9: 1,   14: 2,  15: 3,  16: 3,
                17: 4,  18: 5,  21: 6,  2: 7,
                3: 7,   4: 7,   6: 7,   12: 8,
                22: 9,  23: 10, 24: 11, 25: 12,
                26: 13, 27: 14, 28: 15, 30: 16,
                32: 17
}

class_names_seg16=['noise','barrier','bicycle','bus','car','construction_vehicle','motorcycle','pedestrian','traffic_cone','trailer','truck',
'driveable_surface','other_flat','sidewalk','terrain','manmade','vegetation','empty']

numC_Trans=64

model = dict(
    type='BEVDepth4DOccu',
    aligned=True,
    detach=True,
    before=True,
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='FPNForBEVDet',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(type='ViewTransformerLSSBEVDepth',
                              loss_depth_weight=100.0,
                              grid_config=grid_config,
                              data_config=data_config,
                              numC_Trans=numC_Trans,
                              extra_depth_net=dict(type='ResNetForBEVDet', numC_input=256,
                                                   num_layer=[3,], num_channels=[256,], stride=[1,],)),
    img_bev_encoder_backbone = dict(type='ResNetForBEVDet',  numC_input=numC_Trans*2,
                                    num_channels=[128, 256, 512]),
    img_bev_encoder_neck = dict(type='FPN_LSS_Fusion_Momentum',
                                momentum_weight=0.9,
                                in_channels=numC_Trans*8+numC_Trans*2,
                                out_channels=256),
    devoxel_backbone=dict(
        type='ResNetForBEVDet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    devoxel_neck=dict(
        type='FPN_LSS_Voxel',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    pre_process = dict(type='ResNetForBEVDet',numC_input=numC_Trans,
                      num_layer=[2,], num_channels=[64,], stride=[1,],
                      backbone_output_ids=[0,]),
    voxel_bev_head=dict(type='OccFuserHead',
                        grid_size=voxel_grid_num,
                        nbr_classes=2,
                        in_dims=256,
                        out_dims=64,
                        loss_weight=10,
                        balance=6,
                        ignore_label=-1,
                        weight=[1.0, 2.0]),
    pts_bbox_head=dict(
        type='TaskSpecificCenterHead',
        task_specific=True,
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            # nms_type='circle',
            pre_max_size=1000,
            post_max_size=83,
            # nms_thr=0.2

            # Scale-NMS
            nms_type=['rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0,1.0], [4.5, 9.0]]
        )))


# Data
dataset_type = 'SequentialNuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', is_train=True, data_config=data_config,
         sequential=True, aligned=True, trans_only=False),
    dict(
        type='LoadPointsFromFileOccupancy',
        occupancy_root="./data/nuscenes/pc_panoptic/",
        learning_map=learning_map,
        label_from='panoptic',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans_BEVDet',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        update_img2lidar=True),
    dict(
        type='RandomFlip3D_BEVDet',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(type='GenVoxelLabel',
         grid_size=voxel_grid_num,
         fixed_volume_space=True,
         max_volume_space=point_cloud_range[3:],
         min_volume_space=point_cloud_range[:3],
         is_binary=True
         ),
    dict(type='PointToMultiViewDepth', grid_config=grid_config),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_voxel_bev'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'img_info'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', data_config=data_config,
         sequential=True, aligned=True, trans_only=False),
    # load lidar points for --show in test.py only
    dict(
        type='LoadPointsFromFileOccupancy',
        occupancy_root="./data/nuscenes/pc_panoptic/",
        learning_map=learning_map,
        label_from='panoptic',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='GenVoxelLabel',
         grid_size=voxel_grid_num,
         fixed_volume_space=True,
         max_volume_space=point_cloud_range[3:],
         min_volume_space=point_cloud_range[:3],
         is_binary=True
         ),
    dict(type='PointToMultiViewDepth', grid_config=grid_config),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs', 'gt_voxel_bev'],
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'adjacent', 'adjacent_type',)
                 )
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_train_4d_interval3_max60.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            modality=input_modality,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            img_info_prototype='bevdet_sequential',
            speed_mode='abs_dis',
            max_interval=9,
            min_interval=2,
            prev_only=True,
            fix_direction=True)),
    val=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_val_4d_interval3_max60.pkl',
            pipeline=test_pipeline, 
            classes=class_names,
            modality=input_modality, 
            test_mode=True,
            img_info_prototype='bevdet_sequential',
            speed_mode='abs_dis',
            max_interval=10,
            fix_direction=True,),
    test=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_val_4d_interval3_max60.pkl',
            pipeline=test_pipeline, 
            classes=class_names,
            modality=input_modality,
            test_mode=True,
            img_info_prototype='bevdet_sequential',
            speed_mode='abs_dis',
            max_interval=10,
            fix_direction=True,))

# Optimizer
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)

