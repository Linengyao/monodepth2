# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
# from path import Path  # 如
from pathlib import Path
import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from PIL import Image
from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class CustomDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 假设CustomDataset的根目录结构如下：
        # - root_dir
        #   - images
        #     - train
        #       - img1.jpg
        #       - img2.jpg
        #     - test
        #       - ...
        #   - depths (可选，如果没有则忽略深度图读取)
        #     - train
        #       - depth1.npy
        #       - depth2.npy
        #     - test
        #       - ...

        # self.K = np.array([[6.673912084180287e+03,0,0],[0,6.669658167025367e+03,0],[7.257882442653760e+02,3.281547992822711e+02,1]], dtype=np.float32).T
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.root_dir = Path(args[0])  # 数据集根目录
        self.split = kwargs.get('split', 'train')  # 默认训练集
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}  # 根据需求调整

        # 初始化图像和深度图的路径列表
        # self.split=kwargs['is_train']
        # if self.split:
        #     self.split='train'
        # else:
        #     self.split = 'test'
        self.image_paths = sorted((self.root_dir / "images" / self.split).rglob("*.jpg"))  # 假定是.jpg格式

        if (self.root_dir / "depths" / self.split).exists():
            self.depth_paths = sorted((self.root_dir / "depths" / self.split).rglob("*.npy"))  # 假定深度图为.npy格式

        else:
            self.depth_paths = None  # 若没有深度图，则设置为None

        # 读取相机内参，根据实际情况调整读取方式
        
    def get_color(self, folder, frame_index, side, do_flip):
        # color = self.loader(self.get_image_path(folder, frame_index, side))
        color = self.loader(folder)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


    def get_image_path(self, folder, frame_index, side):
        """
        返回给定帧和视角的图像路径
        """
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = self.image_paths[frame_index]
        return str(image_path)

    def get_depth(self, folder, frame_index, side, w):
        """
        获取深度图数据，如果存在的话
        """
        if self.depth_paths is not None:
            depth_path = self.depth_paths[frame_index]

            depth_data = np.load(str(depth_path)).squeeze()

            # 重塑数组（前提是总元素数量允许）

            # 假设 target_size 是目标尺寸，比如 (new_height, new_width)
            target_size = (1242,375)

            # 将浮点数值数据转换为 0-255 的整数数据
            depth_data_scaled = (depth_data * 2.55).astype(np.uint8)

            # 将数据转换为 PIL Image 对象
            depth_image = Image.fromarray(depth_data_scaled)

            # 使用 PIL Image 对象进行 resize 操作
            resized_depth_image = depth_image.resize(target_size, Image.BILINEAR)

            # 将 resize 后的图像数据转换回 numpy 数组
            resized_depth_data = np.array(resized_depth_image) / 2.55
            return resized_depth_data
        else:
            return None
class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        # color = self.loader(self.get_image_path(folder, frame_index, side))
        color = self.loader(folder)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
