You can download nuScenes 3D detection data [HERE](https://www.nuscenes.org/download) and unzip all zip files.
You can only download `v1.0-trainval` version conveniently.
Like the general way to prepare dataset, it is recommended to symlink the dataset root to `$SOGDet/data`.

We typically need to organize the useful data information with a .pkl.
To prepare these files for nuScenes, run the following command:

```bash
python tools/create_data.py  
python tools/data_converter/prepare_nuscenes_for_bevdet4d.py  
```

Then download nuscenes panoptic dataset [HERE](https://www.nuscenes.org/download).
Symlink the dataset root to `$SOGDet/data/nuscenes` and replace .json file following [HERE](https://www.nuscenes.org/nuscenes?tutorial=lidarseg_panoptic).
Run the following command to generate point cloud with panoptic labels.
```bash
python tools/data_converter/prepare_panoptic.py
```

Here, the total data structure is shown like this:
```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── panoptic
│   │   ├── pc_panoptic
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_train_4d_interval3_max60.pkl
│   │   ├── nuscenes_infos_val_4d_interval3_max60.pkl
```

