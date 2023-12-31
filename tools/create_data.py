# Copyright (c) OpenMMLab. All rights reserved.
# import argparse
from os import path as osp

# from tools.data_converter import indoor_converter as indoor
# from tools.data_converter import kitti_converter as kitti
# from tools.data_converter import lyft_converter as lyft_converter
from tools.data_converter import nuscenes_converter as nuscenes_converter
# from tools.data_converter.create_gt_database import create_groundtruth_database


def nuscenes_data_prep(root_path,
                       info_prefix,
                       version,
                    #    dataset_name,
                    #    out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    # if version == 'v1.0-test':
    #     info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
    #     nuscenes_converter.export_2d_annotation(
    #         root_path, info_test_path, version=version)
    #     return

    # info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
    # info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
    # nuscenes_converter.export_2d_annotation(
    #     root_path, info_train_path, version=version)
    # nuscenes_converter.export_2d_annotation(
    #     root_path, info_val_path, version=version)
    # create_groundtruth_database(dataset_name, root_path, info_prefix,
    #                             f'{out_dir}/{info_prefix}_infos_train.pkl')



if __name__ == '__main__':
    train_version = 'v1.0-trainval'
    nuscenes_data_prep(
        root_path='./data/nuscenes',
        info_prefix='nuscenes',
        version=train_version,
        # dataset_name='NuScenesDataset',
        # out_dir='./data/nuscenes',
        max_sweeps=10)
    test_version = 'v1.0-test'
    nuscenes_data_prep(
        root_path='./data/nuscenes',
        info_prefix='nuscenes',
        version=test_version,
        # dataset_name='NuScenesDataset',
        # out_dir='./data/nuscenes',
        max_sweeps=10)