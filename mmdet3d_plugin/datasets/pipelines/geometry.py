import numpy as np

from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class GenVoxelLabel(object):
    def __init__(self, grid_size, ignore_label=0,
                 fixed_volume_space=False, max_volume_space=[50, np.pi, 3],
                 min_volume_space=[0, -np.pi, -5], is_binary=False):
        self.grid_size = np.asarray(grid_size)
        self.ignore_label = ignore_label
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.is_binary = is_binary

    def __call__(self, results):
        points = np.array(results['points'].tensor)
        # print(points.shape)
        learning_map = results['learning_map']
        points_label = points[:, 4].reshape([-1, 1]).astype(np.int8)
        labels = np.vectorize(learning_map.__getitem__)(points_label)
        xyz_pol = points[:, :3]
        assert self.fixed_volume_space
        max_bound = np.asarray(self.max_volume_space)  # 51.2 51.2 3
        min_bound = np.asarray(self.min_volume_space)  # -51.2 -51.2 -5
        # get grid index
        crop_range = max_bound - min_bound # 102.4  102.4  8
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)


        if (intervals == 0).any():
            print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        # process labels

        if self.is_binary:
            processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label  # zeros (n,4)
            processed_label[grid_ind[:,1], grid_ind[:,0], grid_ind[:,2]] = 1  # y x z
        else:
            processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label  # zeros (n,4)
            label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
            label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]  # >>> sort
            processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)  # x y z
            processed_label = np.transpose(processed_label, (1,0,2)) # y x z



        results['gt_voxel_bev'] = processed_label

        return results

def nb_process_label(processed_label, sorted_label_voxel_pair): # (gx, gy, gz), (n,4)
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label