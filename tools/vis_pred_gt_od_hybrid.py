import os
import json
import numpy as np
from nuscenes import NuScenes

from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from pyquaternion import Quaternion

import numpy as np
import os
import os.path as osp
from nuscenes import NuScenes,NuScenesExplorer
import math
from typing import List
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.axes import Axes
from pyquaternion import Quaternion

from nuscenes.lidarseg.lidarseg_utils import create_lidarseg_legend
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, BoxVisibility, transform_matrix




l32tol16 = {1: 0,
        5: 0,
        7: 0,
        8: 0,
        10: 0,
        11: 0,
        13: 0,
        19: 0,
        20: 0,
        0: 0,
        29: 0,
        31: 0,
        9: 1,
        14: 2,
        15: 3,
        16: 3,
        17: 4,
        18: 5,
        21: 6,
        2: 7,
        3: 7,
        4: 7,
        6: 7,
        12: 8,
        22: 9,
        23: 10,
        24: 11,
        25: 12,
        26: 13,
        27: 14,
        28: 15,
        30: 16}
colormap16={
'noise': (0, 0, 0),  # Black.
"barrier": (112, 128, 144),  # Slategrey
"bicycle": (220, 20, 60),  # Crimson
"bus": (255, 127, 80),  # Coral
"car": (255, 158, 0),  # Orange
"construction_vehicle": (233, 150, 70),  # Darksalmon
"motorcycle": (255, 61, 99),  # Red
"pedestrian": (0, 0, 230),  # Blue
"traffic_cone": (47, 79, 79),  # Darkslategrey
"trailer": (255, 140, 0),  # Darkorange
"truck": (255, 99, 71),  # Tomato
"drive.surf.": (0, 207, 191),  # nuTonomy green
"other flat": (175, 0, 75),
"sidewalk": (75, 0, 75),
"terrain": (112, 180, 60),
"manmade": (222, 184, 135),  # Burlywood
"vegetation": (0, 175, 0),  # Green
}
name2idx16 = {
'noise': 0,
"barrier": 1,
"bicycle": 2,
"bus": 3,
"car": 4,
"construction_vehicle": 5,
"motorcycle": 6,
"pedestrian": 7,
"traffic_cone": 8,
"trailer": 9,
"truck": 10,
"drive.surf.": 11,
"other flat": 12,
"sidewalk":13,
"terrain": 14,
"manmade": 15,
"vegetation": 16
}
idx2name16 = {
0:'noise',
1:"barrier",
2:"bicycle",
3:"bus",
4:"car",
5:"construction_vehicle",
6:"motorcycle",
7:"pedestrian",
8:"traffic_cone",
9:"trailer",
10:"truck",
11:"drive.surf.",
12:"other flat",
13:"sidewalk",
14:"terrain",
15:"manmade",
16:"vegetation"
}

def filter_colors(colors: np.array, classes_to_display: np.array) -> np.ndarray:
    """
    Given an array of RGB colors and a list of classes to display, return a colormap (in RGBA) with the opacity
    of the labels to be display set to 1.0 and those to be hidden set to 0.0
    :param colors: [n x 3] array where each row consist of the RGB values for the corresponding class index
    :param classes_to_display: An array of classes to display (e.g. [1, 8, 32]). The array need not be ordered.
    :return: (colormap <np.float: n, 4)>).

    colormap = np.array([[R1, G1, B1],             colormap = np.array([[1.0, 1.0, 1.0, 0.0],
                         [R2, G2, B2],   ------>                        [R2,  G2,  B2,  1.0],
                         ...,                                           ...,
                         Rn, Gn, Bn]])                                  [1.0, 1.0, 1.0, 0.0]])
    """
    for i in range(len(colors)):
        if i not in classes_to_display:
            colors[i] = [1.0, 1.0, 1.0]  # Mask labels to be hidden with 1.0 in all channels.

    # Convert the RGB colormap to an RGBA array, with the alpha channel set to zero whenever the R, G and B channels
    # are all equal to 1.0.
    alpha = np.array([~np.all(colors == 1.0, axis=1) * 1.0])
    colors = np.concatenate((colors, alpha.T), axis=1)

    return colors

class NuScenesExplorerCustom(NuScenesExplorer):
    """ Helper class to list and visualize NuScenes data. These are meant to serve as tutorials and templates for
    working with the data. """

    def __init__(self, nusc: NuScenes):
        self.nusc = nusc

    def render_sample_data(self,
                           sample_data_token: str,
                           with_anns: bool = True,
                           box_vis_level: BoxVisibility = BoxVisibility.ANY,
                           axes_limit: float = 40,
                           ax: Axes = None,
                           nsweeps: int = 1,
                           out_path: str = None,
                           underlay_map: bool = True,
                           use_flat_vehicle_coordinates: bool = True,
                           show_lidarseg: bool = False,
                           show_lidarseg_legend: bool = False,
                           filter_lidarseg_labels: List = None,
                           lidarseg_preds_bin_path: str = None,
                           verbose: bool = True,
                           use16: bool = False,
                           pred_boxes: list = None,
                           pred_labels: np.array = None) -> None:



        # Get sensor modality.
        sd_record = self.nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']
        if not use16:
            colormap = self.nusc.colormap
            name2idx = self.nusc.lidarseg_name2idx_mapping
            idx2name = self.nusc.lidarseg_idx2name_mapping
        else:
            colormap = colormap16
            name2idx = name2idx16
            idx2name = idx2name16

        if sensor_modality == 'lidar':
            sample_rec = self.nusc.get('sample', sd_record['sample_token'])
            chan = sd_record['channel']
            ref_chan = 'LIDAR_TOP'
            ref_sd_token = sample_rec['data'][ref_chan]
            ref_sd_record = self.nusc.get('sample_data', ref_sd_token)

            if show_lidarseg:
                gt_from = 'lidarseg'
                assert hasattr(self.nusc, gt_from), f'Error: nuScenes-{gt_from} not installed!'

                # Ensure that lidar pointcloud is from a keyframe.
                assert sd_record['is_key_frame'], \
                    'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'

                assert nsweeps == 1, \
                    'Error: Only pointclouds which are keyframes have lidar segmentation labels; nsweeps should ' \
                    'be set to 1.'

                # Load a single lidar point cloud.
                pcl_path = osp.join(self.nusc.dataroot, ref_sd_record['filename'])
                pc = LidarPointCloud.from_file(pcl_path)
            else:
                # Get aggregated lidar point cloud in lidar frame.
                pc, times = LidarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan,
                                                                 nsweeps=nsweeps)


            # By default we render the sample_data top down in the sensor frame.
            # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
            # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
            if use_flat_vehicle_coordinates:
                # Retrieve transformation matrices for reference point cloud.
                cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
                pose_record = self.nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
                ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                              rotation=Quaternion(cs_record["rotation"]))

                # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
                ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                rotation_vehicle_flat_from_vehicle = np.dot(
                    Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                    Quaternion(pose_record['rotation']).inverse.rotation_matrix)
                vehicle_flat_from_vehicle = np.eye(4)
                vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
                viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
            else:
                viewpoint = np.eye(4)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 9))

            # Render map if requested.
            if underlay_map:
                assert use_flat_vehicle_coordinates, 'Error: underlay_map requires use_flat_vehicle_coordinates, as ' \
                                                     'otherwise the location does not correspond to the map!'
                self.render_ego_centric_map(sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax)

            # Show point cloud.
            points = view_points(pc.points[:3, :], viewpoint, normalize=False)
            colors = 'gray'
            if show_lidarseg:
                gt_from = 'lidarseg'


                # Load labels for pointcloud.
                lidarseg_labels_filename = osp.join(self.nusc.dataroot, self.nusc.get(gt_from, sample_data_token)['filename'])
                points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)  # [num_points]

                if use16:
                    points_label = np.vectorize(l32tol16.__getitem__)(points_label) # class 16

                if pred_labels is not None:
                    points_label=pred_labels

                # Given a colormap (class name -> RGB color) and a mapping from class name to class index,
                # get an array of RGB values where each color sits at the index in the array corresponding
                # to the class index.
                colors = []
                for i, (k, v) in enumerate(colormap.items()):
                    # Ensure that the indices from the colormap is same as the class indices.
                    assert i == name2idx[k], 'Error: {} is of index {}, ' \
                                             'but it is of index {} in the colormap.'.format(k, name2idx[k], i)
                    colors.append(v)

                colors = np.array(colors) / 255  # (32,3) Normalize RGB values to be between 0 and 1 for each channel.


                # Paint each label with its respective RGBA value.
                colors = colors[points_label]  # Shape: [num_points, 4]
                mask = points_label != 0

                colors = np.concatenate((colors, mask[...,np.newaxis]), axis=1)

                if show_lidarseg_legend:
                    if filter_lidarseg_labels is None:
                        filter_lidarseg_labels = list(range(1,17,1))
                    create_lidarseg_legend(filter_lidarseg_labels,
                                           idx2name,
                                           colormap,
                                           loc='upper left',
                                           ncol=1,
                                           bbox_to_anchor=(1.05, 1.0))


            point_scale = 0.2 if sensor_modality == 'lidar' else 3.0
            if show_lidarseg:
                point_scale = 5.0
            scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale, marker='s')

            # Show ego vehicle.
            ax.plot(0, 0, 'x', color='red')

            # Get boxes in lidar frame.
            _, boxes, _ = self.nusc.get_sample_data(ref_sd_token, box_vis_level=box_vis_level, use_flat_vehicle_coordinates=use_flat_vehicle_coordinates)
            # Show boxes.
            if with_anns:
                if pred_boxes:
                    boxes = pred_boxes
                    for box in boxes:
                        box_clr = colormap[box.name]
                        c = np.array(box_clr) / 255.0
                        box.render(ax, view=np.eye(4), colors=(c, c, c))
                else:

                    for box in boxes:
                        # print(box.corners()[:2, :])

                        if use16:
                            newname = idx2name[l32tol16[self.nusc.lidarseg_name2idx_mapping[box.name]]]
                            box_clr = colormap[newname]
                        else:
                            box_clr = colormap[box.name]
                        c = np.array(box_clr) / 255.0
                        box.render(ax, view=np.eye(4), colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(-axes_limit, axes_limit)
            ax.set_ylim(-axes_limit, axes_limit)
        elif sensor_modality == 'camera':
            # Load boxes and image.
            data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_data_token,
                                                                           box_vis_level=box_vis_level)
            data = Image.open(data_path)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 16))

            # Show image.
            ax.imshow(data)

            # Show boxes.
            if with_anns:
                for box in boxes:
                    if use16:
                        newname = idx2name[l32tol16[self.nusc.lidarseg_name2idx_mapping[box.name]]]
                        color = colormap[newname]
                    else:
                        color = colormap[box.name]
                    # c = np.array(self.get_color(box.name)) / 255.0
                    c = np.array(color) / 255.0
                    box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(0, data.size[0])
            ax.set_ylim(data.size[1], 0)

        else:
            raise ValueError("Error: Unknown sensor modality!")

        ax.axis('off')
        # ax.set_title('{} {labels_type}'.format(
        #     sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
        ax.set_aspect('equal')

        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)

        if verbose:
            plt.show()

    def render_ego_centric_map(self,
                               sample_data_token: str,
                               axes_limit: float = 40,
                               ax: Axes = None) -> None:
        """
        Render map centered around the associated ego pose.
        :param sample_data_token: Sample_data token.
        :param axes_limit: Axes limit measured in meters.
        :param ax: Axes onto which to render.
        """

        def crop_image(image: np.array,
                       x_px: int,
                       y_px: int,
                       axes_limit_px: int) -> np.array:
            x_min = int(x_px - axes_limit_px)
            x_max = int(x_px + axes_limit_px)
            y_min = int(y_px - axes_limit_px)
            y_max = int(y_px + axes_limit_px)

            cropped_image = image[y_min:y_max, x_min:x_max]

            return cropped_image

        # Get data.
        sd_record = self.nusc.get('sample_data', sample_data_token)
        sample = self.nusc.get('sample', sd_record['sample_token'])
        scene = self.nusc.get('scene', sample['scene_token'])
        log = self.nusc.get('log', scene['log_token'])
        map_ = self.nusc.get('map', log['map_token'])
        map_mask = map_['mask']
        pose = self.nusc.get('ego_pose', sd_record['ego_pose_token'])

        # Retrieve and crop mask.
        pixel_coords = map_mask.to_pixel_coords(pose['translation'][0], pose['translation'][1])
        scaled_limit_px = int(axes_limit * (1.0 / map_mask.resolution))
        mask_raster = map_mask.mask()
        cropped = crop_image(mask_raster, pixel_coords[0], pixel_coords[1], int(scaled_limit_px * math.sqrt(2)))

        # Rotate image.
        ypr_rad = Quaternion(pose['rotation']).yaw_pitch_roll
        yaw_deg = -math.degrees(ypr_rad[0])
        rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))

        # Crop image.
        ego_centric_map = crop_image(rotated_cropped,
                                     int(rotated_cropped.shape[1] / 2),
                                     int(rotated_cropped.shape[0] / 2),
                                     scaled_limit_px)

        # Init axes and show image.
        # Set background to white and foreground (semantic prior) to gray.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 9))
        ego_centric_map[ego_centric_map == map_mask.foreground] = 125
        ego_centric_map[ego_centric_map == map_mask.background] = 255
        ax.imshow(ego_centric_map, extent=[-axes_limit, axes_limit, -axes_limit, axes_limit],
                  alpha=0.1, cmap='gray', vmin=0, vmax=255)

import argparse
parser = argparse.ArgumentParser(description='visualization')

parser.add_argument(
    '--root-path',
    type=str,
    default='./data/nuscenes',
    help='specify the root path of dataset')
parser.add_argument(
    '--bbox-path',
    type=str,
    default='',
    help='specify the bbox path of result')
parser.add_argument(
    '--voxel-path',
    type=str,
    default='',
    help='specify the voxel path of result')
parser.add_argument(
    '--save-path',
    type=str,
    default='',
    help='specify the save path of visualization')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0-trainval',
    required=False,
    help='specify the dataset version')

args = parser.parse_args()

if __name__=="__main__":

    result_path_bbox=args.bbox_path
    result_path_voxel=args.voxel_path
    nuscenes_version = args.version
    dataroot = args.root_path
    save_path = args.save_path


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    nusc = NuScenes(nuscenes_version, dataroot)
    expl = NuScenesExplorerCustom(nusc)


    boxes_dict = dict()
    with open(os.path.join(result_path_bbox,'pts_bbox/results_nusc.json')) as f:
        data = json.load(f)

        for sample_token in data['results']:
            anno_list = data['results'][sample_token]
            boxes_list = []
            for anno in anno_list:
                box = Box(center=anno['translation'], size=anno['size'], orientation=Quaternion(anno['rotation']),
                    name=anno['detection_name'], token=anno['sample_token'])

                boxes_list.append(box)
            boxes_dict[sample_token] = boxes_list




    filenames = os.listdir(result_path_voxel)
    cnt = 0
    for name in filenames:
        if not name.startswith('pred'):
            continue

        sample_token = name.split('.')[0].split('_')[1]
        print('{}: sample_token:{}'.format(cnt,sample_token))
        cnt += 1
        sample = nusc.get('sample', sample_token)

        lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_sd_token = lidar_sd['token']
        pose_record = nusc.get('ego_pose', lidar_sd['ego_pose_token'])

        pred_boxes_list = boxes_dict[sample_token]
        pred_boxes = []
        for box in pred_boxes_list:
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
            pred_boxes.append(box)


        # calculate pred point label form voxel
        pc_gt_name = lidar_sd['filename']
        lidarseg_label_gt_name = nusc.get('lidarseg', lidar_sd_token)['filename']
        pcl_path = os.path.join(nusc.dataroot, pc_gt_name)
        pc_gt = LidarPointCloud.from_file(pcl_path)
        lidarseg_labels_filename = os.path.join(nusc.dataroot, lidarseg_label_gt_name)
        points_label_gt = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)  # [num_points]
        voxel_pred = np.load(os.path.join(result_path_voxel, name)).reshape((128, 128, 10))
        xyz_pol = np.transpose(pc_gt.points[:3]) #[n,3]
        max_bound = np.asarray([51.2, 51.2, 3])  # 51.2 51.2 3
        min_bound = np.asarray([ -51.2, -51.2, -5])  # -51.2 -51.2 -5
        # get grid index
        crop_range = max_bound - min_bound  # 102.4  102.4  8

        cur_grid_size = np.array([128, 128, 10])
        intervals = crop_range / (cur_grid_size - 1)
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int64) #[n,3]
        points_label_pred = np.zeros_like(points_label_gt)
        for idx in range(grid_ind.shape[0]):
            gx,gy,gz=grid_ind[idx]
            points_label_pred[idx] = voxel_pred[gy,gx,gz]







        expl.render_sample_data(lidar_sd_token, with_anns=True, verbose=False,
                                out_path=os.path.join(save_path,'{}_od_pred.png'.format(sample_token)), pred_boxes=pred_boxes, use16=True)  # OD bev pred
        expl.render_sample_data(lidar_sd_token, with_anns=True, verbose=False,
                                out_path=os.path.join(save_path, '{}_od_gt.png'.format(sample_token)), use16=True)  # OD bev gt
        expl.render_sample_data(lidar_sd_token, with_anns=True, verbose=False, show_lidarseg=True,
                                out_path=os.path.join(save_path, '{}_hybrid_pred.png'.format(sample_token)), pred_boxes=pred_boxes, pred_labels=points_label_pred, use16=True)  # OD OCC bev pred
        expl.render_sample_data(lidar_sd_token, with_anns=True, verbose=False, show_lidarseg=True,
                                out_path=os.path.join(save_path, '{}_hybrid_gt.png'.format(sample_token)), use16=True)  # OD OCC bev gt



