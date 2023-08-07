# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import RandomFlip

from mmdet3d.datasets.pipelines import RandomFlip3D, GlobalRotScaleTrans


@PIPELINES.register_module()
class RandomFlip3D_BEVDet(RandomFlip3D):

    def __init__(self,
                #  sync_2d=True,
                #  flip_ratio_bev_horizontal=0.0,
                #  flip_ratio_bev_vertical=0.0,
                 update_img2lidar=False,
                 **kwargs):
        super(RandomFlip3D_BEVDet, self).__init__(**kwargs)
        # super(RandomFlip3D, self).__init__(
        #     flip_ratio=flip_ratio_bev_horizontal, **kwargs)
        # self.sync_2d = sync_2d
        # self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        # if flip_ratio_bev_horizontal is not None:
        #     assert isinstance(
        #         flip_ratio_bev_horizontal,
        #         (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        # if flip_ratio_bev_vertical is not None:
        #     assert isinstance(
        #         flip_ratio_bev_vertical,
        #         (int, float)) and 0 <= flip_ratio_bev_vertical <= 1
        self.update_img2lidar = update_img2lidar

    # def random_flip_data_3d(self, input_dict, direction='horizontal'):
    #     """Flip 3D data randomly.

    #     Args:
    #         input_dict (dict): Result dict from loading pipeline.
    #         direction (str): Flip direction. Default: horizontal.

    #     Returns:
    #         dict: Flipped results, 'points', 'bbox3d_fields' keys are \
    #             updated in the result dict.
    #     """
    #     assert direction in ['horizontal', 'vertical']
    #     if len(input_dict['bbox3d_fields']) == 0:  # test mode
    #         input_dict['bbox3d_fields'].append('empty_box3d')
    #         input_dict['empty_box3d'] = input_dict['box_type_3d'](
    #             np.array([], dtype=np.float32))
    #     assert len(input_dict['bbox3d_fields']) == 1
    #     for key in input_dict['bbox3d_fields']:
    #         if 'points' in input_dict:
    #             input_dict['points'] = input_dict[key].flip(
    #                 direction, points=input_dict['points'])
    #         else:
    #             input_dict[key].flip(direction)
    #     if 'centers2d' in input_dict:
    #         assert self.sync_2d is True and direction == 'horizontal', \
    #             'Only support sync_2d=True and horizontal flip with images'
    #         w = input_dict['ori_shape'][1]
    #         input_dict['centers2d'][..., 0] = \
    #             w - input_dict['centers2d'][..., 0]
    #         # need to modify the horizontal position of camera center
    #         # along u-axis in the image (flip like centers2d)
    #         # ['cam2img'][0][2] = c_u
    #         # see more details and examples at
    #         # https://github.com/open-mmlab/mmdetection3d/pull/744
    #         input_dict['cam2img'][0][2] = w - input_dict['cam2img'][0][2]

    def update_transform(self, input_dict):
        transform = torch.zeros((input_dict['img_inputs'][1].shape[0],4,4)).float()
        transform[:,:3,:3] = input_dict['img_inputs'][1]
        transform[:,:3,-1] = input_dict['img_inputs'][2]
        transform[:, -1, -1] = 1.0

        aug_transform = torch.eye(4).float()
        if input_dict['pcd_horizontal_flip']:
            aug_transform[1,1] = -1
        if input_dict['pcd_vertical_flip']:
            aug_transform[0,0] = -1
        aug_transform = aug_transform.view(1,4,4)
        new_transform = aug_transform.matmul(transform)
        input_dict['img_inputs'][1][...] = new_transform[:,:3,:3]
        input_dict['img_inputs'][2][...] = new_transform[:,:3,-1]


    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and \
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction', \
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added \
                into result dict.
        """
        # filp 2D image and its annotations
        super(RandomFlip3D, self).__call__(input_dict)

        if self.sync_2d:
            input_dict['pcd_horizontal_flip'] = input_dict['flip']
            input_dict['pcd_vertical_flip'] = False
        else:
            if 'pcd_horizontal_flip' not in input_dict:
                flip_horizontal = True if np.random.rand(
                ) < self.flip_ratio else False
                input_dict['pcd_horizontal_flip'] = flip_horizontal
            if 'pcd_vertical_flip' not in input_dict:
                flip_vertical = True if np.random.rand(
                ) < self.flip_ratio_bev_vertical else False
                input_dict['pcd_vertical_flip'] = flip_vertical

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
            input_dict['transformation_3d_flow'].extend(['HF'])
        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')
            input_dict['transformation_3d_flow'].extend(['VF'])

        if 'img_inputs' in input_dict:
            assert self.update_img2lidar
            self.update_transform(input_dict)
        return input_dict

    # def __repr__(self):
    #     """str: Return a string that describes the module."""
    #     repr_str = self.__class__.__name__
    #     repr_str += f'(sync_2d={self.sync_2d},'
    #     repr_str += f' flip_ratio_bev_vertical={self.flip_ratio_bev_vertical})'
    #     return repr_str


@PIPELINES.register_module()
class GlobalRotScaleTrans_BEVDet(GlobalRotScaleTrans):

    def __init__(self,
                #  rot_range=[-0.78539816, 0.78539816],
                #  scale_ratio_range=[0.95, 1.05],
                #  translation_std=[0, 0, 0],
                #  shift_height=False,
                 update_img2lidar=False,
                 **kwargs):
        super(GlobalRotScaleTrans_BEVDet, self).__init__(**kwargs)
        # seq_types = (list, tuple, np.ndarray)
        # if not isinstance(rot_range, seq_types):
        #     assert isinstance(rot_range, (int, float)), \
        #         f'unsupported rot_range type {type(rot_range)}'
        #     rot_range = [-rot_range, rot_range]
        # self.rot_range = rot_range

        # assert isinstance(scale_ratio_range, seq_types), \
        #     f'unsupported scale_ratio_range type {type(scale_ratio_range)}'
        # self.scale_ratio_range = scale_ratio_range

        # if not isinstance(translation_std, seq_types):
        #     assert isinstance(translation_std, (int, float)), \
        #         f'unsupported translation_std type {type(translation_std)}'
        #     translation_std = [
        #         translation_std, translation_std, translation_std
        #     ]
        # assert all([std >= 0 for std in translation_std]), \
        #     'translation_std should be positive'
        # self.translation_std = translation_std
        # self.shift_height = shift_height
        self.update_img2lidar = update_img2lidar

    # def _trans_bbox_points(self, input_dict):
    #     """Private function to translate bounding boxes and points.

    #     Args:
    #         input_dict (dict): Result dict from loading pipeline.

    #     Returns:
    #         dict: Results after translation, 'points', 'pcd_trans' \
    #             and keys in input_dict['bbox3d_fields'] are updated \
    #             in the result dict.
    #     """
    #     translation_std = np.array(self.translation_std, dtype=np.float32)
    #     trans_factor = np.random.normal(scale=translation_std, size=3).T

    #     input_dict['points'].translate(trans_factor)
    #     input_dict['pcd_trans'] = trans_factor
    #     for key in input_dict['bbox3d_fields']:
    #         input_dict[key].translate(trans_factor)

    # def _rot_bbox_points(self, input_dict):
    #     """Private function to rotate bounding boxes and points.

    #     Args:
    #         input_dict (dict): Result dict from loading pipeline.

    #     Returns:
    #         dict: Results after rotation, 'points', 'pcd_rotation' \
    #             and keys in input_dict['bbox3d_fields'] are updated \
    #             in the result dict.
    #     """
    #     rotation = self.rot_range
    #     noise_rotation = np.random.uniform(rotation[0], rotation[1])

    #     # if no bbox in input_dict, only rotate points
    #     if len(input_dict['bbox3d_fields']) == 0:
    #         rot_mat_T = input_dict['points'].rotate(noise_rotation)
    #         input_dict['pcd_rotation'] = rot_mat_T
    #         return

    #     # rotate points with bboxes
    #     for key in input_dict['bbox3d_fields']:
    #         if len(input_dict[key].tensor) != 0:
    #             points, rot_mat_T = input_dict[key].rotate(
    #                 noise_rotation, input_dict['points'])
    #             input_dict['points'] = points
    #             input_dict['pcd_rotation'] = rot_mat_T

    # def _scale_bbox_points(self, input_dict):
    #     """Private function to scale bounding boxes and points.

    #     Args:
    #         input_dict (dict): Result dict from loading pipeline.

    #     Returns:
    #         dict: Results after scaling, 'points'and keys in \
    #             input_dict['bbox3d_fields'] are updated in the result dict.
    #     """
    #     scale = input_dict['pcd_scale_factor']
    #     points = input_dict['points']
    #     points.scale(scale)
    #     if self.shift_height:
    #         assert 'height' in points.attribute_dims.keys(), \
    #             'setting shift_height=True but points have no height attribute'
    #         points.tensor[:, points.attribute_dims['height']] *= scale
    #     input_dict['points'] = points

    #     for key in input_dict['bbox3d_fields']:
    #         input_dict[key].scale(scale)

    # def _random_scale(self, input_dict):
    #     """Private function to randomly set the scale factor.

    #     Args:
    #         input_dict (dict): Result dict from loading pipeline.

    #     Returns:
    #         dict: Results after scaling, 'pcd_scale_factor' are updated \
    #             in the result dict.
    #     """
    #     scale_factor = np.random.uniform(self.scale_ratio_range[0],
    #                                      self.scale_ratio_range[1])
    #     input_dict['pcd_scale_factor'] = scale_factor

    def update_transform(self, input_dict):
        transform = torch.zeros((input_dict['img_inputs'][1].shape[0],4,4)).float()
        transform[:,:3,:3] = input_dict['img_inputs'][1]
        transform[:,:3,-1] = input_dict['img_inputs'][2]
        transform[:, -1, -1] = 1.0

        aug_transform = torch.zeros((input_dict['img_inputs'][1].shape[0],4,4)).float()
        if 'pcd_rotation' in input_dict:
            aug_transform[:,:3,:3] = input_dict['pcd_rotation'].T * input_dict['pcd_scale_factor']
        else:
            aug_transform[:, :3, :3] = torch.eye(3).view(1,3,3) * input_dict['pcd_scale_factor']
        aug_transform[:,:3,-1] = torch.from_numpy(input_dict['pcd_trans']).reshape(1,3)
        aug_transform[:, -1, -1] = 1.0

        new_transform = aug_transform.matmul(transform)
        input_dict['img_inputs'][1][...] = new_transform[:,:3,:3]
        input_dict['img_inputs'][2][...] = new_transform[:,:3,-1]


    def __call__(self, input_dict):
        """Private function to rotate, scale and translate bounding boxes and \
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
                'pcd_scale_factor', 'pcd_trans' and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        self._trans_bbox_points(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'S', 'T'])
        if 'img_inputs' in input_dict:
            assert self.update_img2lidar
            self.update_transform(input_dict)
        return input_dict

    # def __repr__(self):
    #     """str: Return a string that describes the module."""
    #     repr_str = self.__class__.__name__
    #     repr_str += f'(rot_range={self.rot_range},'
    #     repr_str += f' scale_ratio_range={self.scale_ratio_range},'
    #     repr_str += f' translation_std={self.translation_std},'
    #     repr_str += f' shift_height={self.shift_height})'
    #     return repr_str


# @PIPELINES.register_module()
# class ObjectRangeFilter(object):
#     """Filter objects by the range.

#     Args:
#         point_cloud_range (list[float]): Point cloud range.
#     """

#     def __init__(self, point_cloud_range):
#         self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

#     def __call__(self, input_dict):
#         """Call function to filter objects by the range.

#         Args:
#             input_dict (dict): Result dict from loading pipeline.

#         Returns:
#             dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
#                 keys are updated in the result dict.
#         """
#         # Check points instance type and initialise bev_range
#         if isinstance(input_dict['gt_bboxes_3d'],
#                       (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
#             bev_range = self.pcd_range[[0, 1, 3, 4]]
#         elif isinstance(input_dict['gt_bboxes_3d'], CameraInstance3DBoxes):
#             bev_range = self.pcd_range[[0, 2, 3, 5]]

#         gt_bboxes_3d = input_dict['gt_bboxes_3d']
#         gt_labels_3d = input_dict['gt_labels_3d']
#         mask = gt_bboxes_3d.in_range_bev(bev_range)
#         gt_bboxes_3d = gt_bboxes_3d[mask]
#         # mask is a torch tensor but gt_labels_3d is still numpy array
#         # using mask to index gt_labels_3d will cause bug when
#         # len(gt_labels_3d) == 1, where mask=1 will be interpreted
#         # as gt_labels_3d[1] and cause out of index error
#         gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]

#         # limit rad to [-pi, pi]
#         gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
#         input_dict['gt_bboxes_3d'] = gt_bboxes_3d
#         input_dict['gt_labels_3d'] = gt_labels_3d

#         return input_dict

#     def __repr__(self):
#         """str: Return a string that describes the module."""
#         repr_str = self.__class__.__name__
#         repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
#         return repr_str


# @PIPELINES.register_module()
# class ObjectNameFilter(object):
#     """Filter GT objects by their names.

#     Args:
#         classes (list[str]): List of class names to be kept for training.
#     """

#     def __init__(self, classes):
#         self.classes = classes
#         self.labels = list(range(len(self.classes)))

#     def __call__(self, input_dict):
#         """Call function to filter objects by their names.

#         Args:
#             input_dict (dict): Result dict from loading pipeline.

#         Returns:
#             dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
#                 keys are updated in the result dict.
#         """
#         gt_labels_3d = input_dict['gt_labels_3d']
#         gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
#                                   dtype=np.bool_)
#         input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][gt_bboxes_mask]
#         input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][gt_bboxes_mask]

#         return input_dict

#     def __repr__(self):
#         """str: Return a string that describes the module."""
#         repr_str = self.__class__.__name__
#         repr_str += f'(classes={self.classes})'
#         return repr_str

