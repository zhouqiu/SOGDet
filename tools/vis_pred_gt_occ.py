
import numpy as np
import os
from nuscenes import NuScenes
from PIL import Image
from pyquaternion import Quaternion
import math
from mayavi import mlab
import argparse
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pylab as pl

point_cloud_range = [-51.2, -51.2, -5, 51.2, 51.2, 3]
voxel_size=0.2
voxel_shape=(int((point_cloud_range[3]-point_cloud_range[0])/voxel_size),
             int((point_cloud_range[4]-point_cloud_range[1])/voxel_size),
             int((point_cloud_range[5]-point_cloud_range[2])/voxel_size))

def remove_far(points, point_cloud_range):
    mask = (points[:, 0]>point_cloud_range[0]) & (points[:, 0]<point_cloud_range[3]) & (points[:, 1]>point_cloud_range[1]) & (points[:, 1]<point_cloud_range[4]) \
            & (points[:, 2]>point_cloud_range[2]) & (points[:, 2]<point_cloud_range[5])
    return points[mask, :]

def voxelize(voxel: np.array, label_count: np.array):

    for x in range(voxel.shape[0]):
        for y in range(voxel.shape[1]):
            for z in range(voxel.shape[2]):
                if label_count[x, y, z] == 0:
                    continue
                labels = voxel[x, y, z]
                try:
                    label_count[x, y, z] = np.argmax(np.bincount(labels[labels!=0]))
                except:
                    label_count[x, y, z] = 0
    return label_count

def points2voxel(points, voxel_shape, voxel_size, max_points=5, specific_category=None):
    voxel = np.zeros((*voxel_shape, max_points), dtype=np.int64)
    label_count = np.zeros((voxel_shape), dtype=np.int64)
    index = points[:, 4].argsort()
    points = points[index]
    for point in points:
      
        x, y, z = point[0], point[1], point[2]
        x = round((x - point_cloud_range[0]) / voxel_size)
        y = round((y - point_cloud_range[1]) / voxel_size)
        z = round((z - point_cloud_range[2]) / voxel_size)

        try:
            voxel[x, y, z, label_count[x, y, z]] = int(point[4])  # map_label[int(point[4])]
            label_count[x, y, z] += 1
        except:
            continue

    voxel = voxelize(voxel, label_count)
    voxel = voxel.astype(np.float64)
    return voxel


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)
    g_zz = np.arange(0, dims[2] + 1)

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)

    coords_grid = (coords_grid * resolution) + resolution / 2

    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    coords_grid = np.copy(temp)

    return coords_grid


def mlab_imshowColor(im, alpha=255, **kwargs):
    """
    Plot a color image with mayavi.mlab.imshow.
    im is a ndarray with dim (n, m, 3) and scale (0->255]
    alpha is a single number or a ndarray with dim (n*m) and scale (0->255]
    **kwargs is passed onto mayavi.mlab.imshow(..., **kwargs)
    """
    try:
        alpha[0]
    except:
        alpha = pl.ones(im.shape[0] * im.shape[1]) * alpha
    if len(alpha.shape) != 1:
        alpha = alpha.flatten()

    # The lut is a Nx4 array, with the columns representing RGBA
    # (red, green, blue, alpha) coded with integers going from 0 to 255,
    # we create it by stacking all the pixles (r,g,b,alpha) as rows.
    myLut = pl.c_[im.reshape(-1, 3), alpha]
    myLutLookupArray = pl.arange(im.shape[0] * im.shape[1]).reshape(im.shape[0], im.shape[1])

    #We can display an color image by using mlab.imshow, a lut color list and a lut lookup table.
    theImshow = mlab.imshow(myLutLookupArray, colormap='binary', **kwargs) #temporary colormap
    theImshow.module_manager.scalar_lut_manager.lut.table = myLut
    mlab.draw()
    return theImshow


def draw(
    voxels,
    voxel_size=0.2,
    is16classes=False,
isbinary=False,
        img=None,
        show=True,
        out_path=None,
):


    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    )

    # Attach the predicted class to every voxel
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T


    grid_voxels = grid_coords[
        (grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 255)
    ]

    # figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1))
    figure = mlab.figure(size=(2000, 2000), bgcolor=(1, 1, 1))



    if img is not None:
        # mlab_imshowColor(np.zeros((128,128,3)))
        mlab_imshowColor(img)
    
    plt_plot = mlab.points3d(
        grid_voxels[:, 0],
        grid_voxels[:, 1],
        grid_voxels[:, 2],
        grid_voxels[:, 3] * grid_voxels[:, 2] if isbinary else grid_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19,
    )

    classname_to_color = {  # RGB.
        "noise": (0, 0, 0),  # Black.
        "animal": (70, 130, 180),  # Steelblue
        "human.pedestrian.adult": (0, 0, 230),  # Blue
        "human.pedestrian.child":(0, 0, 230),  # Skyblue,
        "human.pedestrian.construction_worker":(0, 0, 230),  # Cornflowerblue
        "human.pedestrian.personal_mobility": (0, 0, 230),  # Palevioletred
        "human.pedestrian.police_officer":(0, 0, 230),  # Navy,
        "human.pedestrian.stroller": (0, 0, 230),  # Lightcoral
        "human.pedestrian.wheelchair": (0, 0, 230),  # Blueviolet
        "movable_object.barrier": (112, 128, 144),  # Slategrey
        "movable_object.debris": (112, 128, 144),  # Chocolate
        "movable_object.pushable_pullable":(112, 128, 144),  # Dimgrey
        "movable_object.trafficcone":(112, 128, 144),  # Darkslategrey
        "static_object.bicycle_rack": (188, 143, 143),  # Rosybrown
        "vehicle.bicycle": (220, 20, 60),  # Crimson
        "vehicle.bus.bendy":(255, 158, 0),  # Coral
        "vehicle.bus.rigid": (255, 158, 0),  # Orangered
        "vehicle.car": (255, 158, 0),  # Orange
        "vehicle.construction":(255, 158, 0),  # Darksalmon
        "vehicle.emergency.ambulance":(255, 158, 0),
        "vehicle.emergency.police": (255, 158, 0),  # Gold
        "vehicle.motorcycle": (255, 158, 0),  # Red
        "vehicle.trailer":(255, 158, 0),  # Darkorange
        "vehicle.truck": (255, 158, 0),  # Tomato
        "flat.driveable_surface": (0, 207, 191),  # nuTonomy green
        "flat.other":(0, 207, 191),
        "flat.sidewalk": (75, 0, 75),
        "flat.terrain": (0, 207, 191),
        "static.manmade": (222, 184, 135),  # Burlywood
        "static.other": (0, 207, 191),  # Bisque
        "static.vegetation": (0, 175, 0),  # Green
        "vehicle.ego": (255, 240, 245)
    }
    colormap16 = {
        'noise': (0, 0, 0),  # Black.
        "barrier": (112, 128, 144),  # Slategrey
        "bicycle": (220, 20, 60),  # Crimson
        "bus": (255, 127, 80),  # Coral
        "car": (255, 158, 0),  # Orange
        "const.veh.": (233, 150, 70),  # Darksalmon
        "motorcycle": (255, 61, 99),  # Red
        "pedestrian.adult": (0, 0, 230),  # Blue
        "trafficcone": (47, 79, 79),  # Darkslategrey
        "trailer": (255, 140, 0),  # Darkorange
        "truck": (255, 99, 71),  # Tomato
        "drive.surf.": (0, 207, 191),  # nuTonomy green
        "other flat": (175, 0, 75),
        "sidewalk": (75, 0, 75),
        "terrain": (112, 180, 60),
        "manmade": (222, 184, 135),  # Burlywood
        "vegetation": (0, 175, 0),  # Green
    }
    colormap2 = {
        '0': (80, 80, 80),  # Black.
        "1": (95, 95, 95),  # Slategrey
        '2': (110, 110, 110),  # Black.
        "3": (125, 125, 125),  # Slategrey
        '4': (140, 140, 140),  # Black.
        "5": (155, 155, 155),  # Slategrey
        '6': (170, 170, 170),  # Black.
        "7": (185, 185, 185),  # Slategrey
        "8": (200, 200, 200),  # Slategrey
        '9': (215, 215, 215),  # Black.

    }

    if is16classes:
        colors = np.array(list(colormap16.values())).astype(np.uint8)
        alpha = np.ones((colors.shape[0], 1), dtype=np.uint8) * 255
        colors = np.hstack([colors, alpha])
    elif isbinary:
       
        colors = np.array(list(colormap2.values())).astype(np.uint8)
        alpha = np.ones((colors.shape[0], 1), dtype=np.uint8) * 255
        colors = np.hstack([colors, alpha])
    else:
        colors = np.array(list(classname_to_color.values())).astype(np.uint8)
        alpha = np.ones((colors.shape[0], 1), dtype=np.uint8) * 255
        colors = np.hstack([colors, alpha])



    plt_plot.glyph.scale_mode = "scale_by_vector"

    plt_plot.module_manager.scalar_lut_manager.lut.table = colors
    if is16classes:
        plt_plot.module_manager.scalar_lut_manager.data_range = [0, 16]
    elif isbinary:
        plt_plot.module_manager.scalar_lut_manager.data_range = [0, 10]
    else:
        plt_plot.module_manager.scalar_lut_manager.data_range = [0, 31]
    if show:
        mlab.show()
    else:
        mlab.savefig(out_path,size=(100,100))
    mlab.close()


nuscenes_version = 'v1.0-trainval'
dataroot = 'D:\Jenny\Code\\v1.0-val'

nusc = NuScenes(nuscenes_version, dataroot)


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

def get_map_image(sample_token: str,axes_limit: float = 40,):
    # Get data.
    sample = nusc.get('sample', sample_token)
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    sample_data_token = lidar_sd['token']
    sd_record = nusc.get('sample_data', sample_data_token)
    sample = nusc.get('sample', sd_record['sample_token'])
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    map_ = nusc.get('map', log['map_token'])
    map_mask = map_['mask']
    pose = nusc.get('ego_pose', sd_record['ego_pose_token'])

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

    ego_centric_map[ego_centric_map == map_mask.foreground] = 125
    ego_centric_map[ego_centric_map == map_mask.background] = 255
    # ax.imshow(ego_centric_map, extent=[-axes_limit, axes_limit, -axes_limit, axes_limit],
    #           alpha=0.1, cmap='gray', vmin=0, vmax=255)
    return ego_centric_map


parser = argparse.ArgumentParser(description='visualization')
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


args = parser.parse_args()
if __name__=='__main__':
    # coarse
    result_path=args.voxel_path
    savepath = args.save_path

    filenames = os.listdir(result_path)
    cnt = 0
    for name in filenames:
        if not name.startswith('pred'):
            continue

        sample_token = name.split('.')[0].split('_')[1]
        print('{}: sample_token:{}'.format(cnt,sample_token))
        cnt+=1
        gt = np.load(os.path.join(result_path, name.replace('pred', 'gt')))
        pred = np.load(os.path.join(result_path, name))

        gt = gt.reshape((128, 128, 10))
        pred = pred.reshape((128, 128, 10)).astype(np.uint8)

        draw(gt[::-1], voxel_size=0.8, isbinary=True, show=False, out_path=os.path.join(savepath, '{}_gt_3D01.png'.format(sample_token)))
        draw(pred[::-1], voxel_size=0.8, isbinary=True, show=False, out_path=os.path.join(savepath, '{}_pred_3D01.png'.format(sample_token)))

