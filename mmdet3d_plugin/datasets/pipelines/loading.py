import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines import LoadPointsFromFile
from mmdet3d.core.points import BasePoints, get_points_type

import torch
import torchvision
from PIL import Image
import mmcv
import numpy as np
from pyquaternion import Quaternion



@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_BEVDet(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, data_config, is_train=False,
                 sequential=False, aligned=False, trans_only=True):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = torchvision.transforms.Compose((
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])))
        self.sequential = sequential
        self.aligned = aligned
        self.trans_only = trans_only

    def get_rot(self,h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(self.data_config['cams']):
            cams = np.random.choice(self.data_config['cams'], self.data_config['Ncams'],
                                    replace=False)
        else:
            cams = self.data_config['cams']
        return cams

    def sample_augmentation(self, H , W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW)/float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW)/float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_inputs(self,results, flip=None, scale=None):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        cams = self.choose_cams()
        for cam in cams:
            cam_data = results['img_info'][cam]
            filename = cam_data['data_path']
            img = Image.open(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data['cam_intrinsic'])
            rot = torch.Tensor(cam_data['sensor2lidar_rotation'])
            tran = torch.Tensor(cam_data['sensor2lidar_translation'])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(H=img.height,
                                                                               W=img.width,
                                                                               flip=flip,
                                                                               scale=scale)
            img, post_rot2, post_tran2 = self.img_transform(img, post_rot, post_tran,
                                                            resize=resize,
                                                            resize_dims=resize_dims,
                                                            crop=crop,
                                                            flip=flip,
                                                            rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                if not type(results['adjacent']) is list:
                    filename_adjacent = results['adjacent']['cams'][cam]['data_path']
                    img_adjacent = Image.open(filename_adjacent)
                    img_adjacent = self.img_transform_core(img_adjacent,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
                else:
                    for id in range(len(results['adjacent'])):
                        filename_adjacent = results['adjacent'][id]['cams'][cam]['data_path']
                        img_adjacent = Image.open(filename_adjacent)
                        img_adjacent = self.img_transform_core(img_adjacent,
                                                               resize_dims=resize_dims,
                                                               crop=crop,
                                                               flip=flip,
                                                               rotate=rotate)
                        imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            if self.trans_only:
                if not type(results['adjacent']) is list:
                    rots.extend(rots)
                    post_trans.extend(post_trans)
                    post_rots.extend(post_rots)
                    intrins.extend(intrins)
                    if self.aligned:
                        posi_curr = np.array(results['curr']['ego2global_translation'], dtype=np.float32)
                        posi_adj = np.array(results['adjacent']['ego2global_translation'], dtype=np.float32)
                        shift_global = posi_adj - posi_curr

                        l2e_r = results['curr']['lidar2ego_rotation']
                        e2g_r = results['curr']['ego2global_rotation']
                        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
                        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

                        # shift_global = np.array([*shift_global[:2], 0.0])
                        shift_lidar = shift_global @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                            l2e_r_mat).T
                        trans.extend([tran + shift_lidar for tran in trans])
                    else:
                        trans.extend(trans)
                else:
                    # assert False
                    curr_post_trans = post_trans
                    curr_post_rots = post_rots
                    curr_intrins = intrins
                    curr_trans = trans
                    curr_rots = rots
                    for id in range(len(results['adjacent'])):
                        rots.extend(curr_rots)
                        post_trans.extend(curr_post_trans)
                        post_rots.extend(curr_post_rots)
                        intrins.extend(curr_intrins)
                        if self.aligned:
                            posi_curr = np.array(results['curr']['ego2global_translation'], dtype=np.float32)
                            posi_adj = np.array(results['adjacent'][id]['ego2global_translation'], dtype=np.float32)
                            shift_global = posi_adj - posi_curr

                            l2e_r = results['curr']['lidar2ego_rotation']
                            e2g_r = results['curr']['ego2global_rotation']
                            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
                            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

                            # shift_global = np.array([*shift_global[:2], 0.0])
                            shift_lidar = shift_global @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                                l2e_r_mat).T
                            trans.extend([tran + shift_lidar for tran in curr_trans])
                        else:
                            trans.extend(curr_trans)
            else:
                if not type(results['adjacent']) is list:
                    post_trans.extend(post_trans)
                    post_rots.extend(post_rots)
                    intrins.extend(intrins)
                    if self.aligned:
                        egocurr2global = np.eye(4, dtype=np.float32)
                        egocurr2global[:3,:3] = Quaternion(results['curr']['ego2global_rotation']).rotation_matrix
                        egocurr2global[:3,3] = results['curr']['ego2global_translation']

                        egoadj2global = np.eye(4, dtype=np.float32)
                        egoadj2global[:3,:3] = Quaternion(results['adjacent']['ego2global_rotation']).rotation_matrix
                        egoadj2global[:3,3] = results['adjacent']['ego2global_translation']

                        lidar2ego = np.eye(4, dtype=np.float32)
                        lidar2ego[:3, :3] = Quaternion(results['curr']['lidar2ego_rotation']).rotation_matrix
                        lidar2ego[:3, 3] = results['curr']['lidar2ego_translation']

                        lidaradj2lidarcurr = np.linalg.inv(lidar2ego) @ np.linalg.inv(egocurr2global) \
                                             @ egoadj2global @ lidar2ego
                        trans_new = []
                        rots_new =[]
                        for tran,rot in zip(trans, rots):
                            mat = np.eye(4, dtype=np.float32)
                            mat[:3,:3] = rot
                            mat[:3,3] = tran
                            mat = lidaradj2lidarcurr @ mat
                            rots_new.append(torch.from_numpy(mat[:3,:3]))
                            trans_new.append(torch.from_numpy(mat[:3,3]))
                        rots.extend(rots_new)
                        trans.extend(trans_new)
                    else:
                        rots.extend(rots)
                        trans.extend(trans)
                else:
                    # assert False
                    curr_post_trans = post_trans.copy()
                    curr_post_rots = post_rots.copy()
                    curr_intrins = intrins.copy()
                    curr_trans = trans.copy()
                    curr_rots = rots.copy()
                    for id in range(len(results['adjacent'])):
                        post_trans.extend(curr_post_trans)
                        post_rots.extend(curr_post_rots)
                        intrins.extend(curr_intrins)
                        if self.aligned:
                            egocurr2global = np.eye(4, dtype=np.float32)
                            egocurr2global[:3,:3] = Quaternion(results['curr']['ego2global_rotation']).rotation_matrix
                            egocurr2global[:3,3] = results['curr']['ego2global_translation']

                            egoadj2global = np.eye(4, dtype=np.float32)
                            egoadj2global[:3,:3] = Quaternion(results['adjacent'][id]['ego2global_rotation']).rotation_matrix
                            egoadj2global[:3,3] = results['adjacent'][id]['ego2global_translation']

                            lidar2ego = np.eye(4, dtype=np.float32)
                            lidar2ego[:3, :3] = Quaternion(results['curr']['lidar2ego_rotation']).rotation_matrix
                            lidar2ego[:3, 3] = results['curr']['lidar2ego_translation']

                            lidaradj2lidarcurr = np.linalg.inv(lidar2ego) @ np.linalg.inv(egocurr2global) \
                                                @ egoadj2global @ lidar2ego
                            trans_new = []
                            rots_new =[]
                            for tran,rot in zip(curr_trans, curr_rots):
                                mat = np.eye(4, dtype=np.float32)
                                mat[:3,:3] = rot
                                mat[:3,3] = tran
                                mat = lidaradj2lidarcurr @ mat
                                rots_new.append(torch.from_numpy(mat[:3,:3]))
                                trans_new.append(torch.from_numpy(mat[:3,3]))
                            rots.extend(rots_new)
                            trans.extend(trans_new)
                        else:
                            rots.extend(curr_rots)
                            trans.extend(curr_trans)

        imgs, rots, trans, intrins, post_rots, post_trans = (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                                                             torch.stack(intrins), torch.stack(post_rots),
                                                             torch.stack(post_trans))
        return imgs, rots, trans, intrins, post_rots, post_trans

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results
    
@PIPELINES.register_module()
class PointToMultiViewDepth(object):
    def __init__(self, grid_config, downsample=16):
        self.downsample = downsample
        self.grid_config=grid_config

    def points2depthmap(self, points, height, width, canvas=None):
        height, width = height//self.downsample, width//self.downsample
        depth_map = torch.zeros((height,width), dtype=torch.float32)
        coor = torch.round(points[:,:2]/self.downsample)
        depth = points[:,2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) \
               & (coor[:, 1] >= 0) & (coor[:, 1] < height) \
                & (depth < self.grid_config['dbound'][1]) \
                & (depth >= self.grid_config['dbound'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks+depth/100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:,1],coor[:,0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs, rots, trans, intrins, post_rots, post_trans = results['img_inputs']
        depth_map_list = []
        for cid in range(rots.shape[0]):
            combine = rots[cid].matmul(torch.inverse(intrins[cid]))
            combine_inv = torch.inverse(combine)
            points_img = (points_lidar.tensor[:,:3] - trans[cid:cid+1,:]).matmul(combine_inv.T)
            points_img = torch.cat([points_img[:,:2]/points_img[:,2:3],
                                   points_img[:,2:3]], 1)
            points_img = points_img.matmul(post_rots[cid].T)+post_trans[cid:cid+1,:]
            depth_map = self.points2depthmap(points_img, imgs.shape[2], imgs.shape[3])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, depth_map)
        return results

@PIPELINES.register_module()
class LoadPointsFromFileOccupancy(LoadPointsFromFile):
    def __init__(self, occupancy_root='data/nuscenes/occupancy/', learning_map=None, label_from='lidarseg',**kwargs):
        super(LoadPointsFromFileOccupancy, self).__init__(**kwargs)
        self.occupancy_root = occupancy_root
        self.learning_map = learning_map
        self.label_from = label_from

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.
        Args:
            pts_filename (str): Filename of point clouds data.
        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float16)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float16)

        return points

    def __call__(self, results):
        pts_filename = results['pts_filename']
        pts_filename = self.occupancy_root + pts_filename.split("/")[-1]
        points = self._load_points(pts_filename)
        if self.label_from == 'panoptic':
            points = points.reshape(self.load_dim, -1).transpose(1,0)
        else:
            points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        results['learning_map'] = self.learning_map

        return results