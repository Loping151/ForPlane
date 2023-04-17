import glob
import json
import logging as log
import math
import os
import time
from collections import defaultdict
from typing import Optional, List, Tuple, Any, Dict, Union

import numpy as np
import torch

from .base_dataset import BaseDataset
from .data_loading import parallel_load_images, parallel_load_endo_depth, parallel_load_endo_mask
from .intrinsics import Intrinsics
from .llff_dataset import load_llff_poses_helper, load_endo_pose_helper
from .ray_utils import (
    generate_spherical_poses, create_meshgrid, stack_camera_dirs, get_rays, generate_spiral_path
)
from .synthetic_nerf_dataset import (
    load_360_images, load_360_intrinsics,
)
import imageio


class VideoEndoDataset(BaseDataset):
    len_time: int
    max_cameras: Optional[int]
    max_tsteps: Optional[int]
    timestamps: Optional[torch.Tensor]

    def __init__(self,
                 datadir: str,
                 split: str,
                 batch_size: Optional[int] = None,
                 downsample: float = 1.0,
                 keyframes: bool = False,
                 max_cameras: Optional[int] = None,
                 max_tsteps: Optional[int] = None,
                 isg: bool = False,
                 contraction: bool = False,
                 ndc: bool = False,
                 scene_bbox: Optional[List] = None,
                 near_scaling: float = 0.9,
                 ndc_far: float = 2.6,
                 **kwargs):
        self.keyframes = keyframes
        self.max_cameras = max_cameras
        self.max_tsteps = max_tsteps
        self.downsample = downsample
        self.isg = isg
        self.ist = False
        self.maskIS = kwargs.get('maskIS', False)
        self.p_ratio = kwargs.get('p_ratio', 1)
        self.frequency_ratio = kwargs.get('frequency_ratio', None)
        if self.maskIS:
            assert self.frequency_ratio is not None
        self.sample_from_masks = kwargs.get('sample_from_masks', False)
        assert not(self.maskIS and self.sample_from_masks)
        # self.lookup_time = False
        self.per_cam_near_fars = None
        self.global_translation = torch.tensor([0, 0, 0])
        self.global_scale = torch.tensor([1, 1, 1])
        self.near_scaling = near_scaling
        self.ndc_far = ndc_far
        self.median_imgs = None
        self.mask_weights = None
        self.p = None  # p is the additional weight
        self.bg_color = kwargs.get('bg_color', 1)
        if contraction and ndc:
            raise ValueError("Options 'contraction' and 'ndc' are exclusive.")
        
        intrinsics = load_endo_pose_helper(
            datadir, self.downsample, self.near_scaling)

        if split == 'test':
            keyframes = False

        # load images
        paths_img, paths_mask, paths_depth = [], [], []
        png_cnt = 0
        str_cnt = '/000000.png'

        while os.path.exists(datadir + "/images" + str_cnt):
            paths_img.append(datadir + "/images" + str_cnt)
            paths_mask.append(datadir + "/gt_masks" + str_cnt)
            paths_depth.append(datadir + "/depth" + str_cnt)
            png_cnt += 1
            str_cnt = '/' + '0' * \
                (6-len(str(png_cnt))) + str(png_cnt) + '.png'

        assert len(paths_img) == len(paths_mask) == len(
            paths_depth), "The lists must have the same length."

        imgs = parallel_load_images(
            data_dir=datadir,
            dset_type='llff',
            tqdm_title=f"Loading {split} data",
            num_images=len(paths_img),  # Need to be modifieds
            paths=paths_img,
            out_h=intrinsics.height,
            out_w=intrinsics.width,
        )

        # for endonerf dataset, we need load mask and depth
        masks = parallel_load_endo_mask(
            data_dir=datadir,
            tqdm_title=f"Loading {split} data",
            num_images=len(paths_mask),  # Need to be modifieds
            paths=paths_mask,
            out_h=intrinsics.height,
            out_w=intrinsics.width,
        )
        depths = parallel_load_endo_depth(
            data_dir=datadir,
            tqdm_title=f"Loading {split} data",
            num_images=len(paths_depth),  # Need to be modifieds
            paths=paths_depth,
            out_h=intrinsics.height,
            out_w=intrinsics.width,
        )

        timestamps = torch.linspace(0, 299, len(paths_img))

        imgs, self.masks, self.depths = [torch.cat(lst, dim=0) for lst in [
            imgs, masks, depths]]
        # we use near and far to normalize depth
        self.close_depth = percentile_torch(self.depths, 3)
        self.inf_depth = percentile_torch(self.depths, 99.9)
        # values larger than inf_depth should be regarded as unreliable, use 0 to replace them
        self.depths[self.depths > self.inf_depth] = 0
        # pre norm depth to be in [0, 1]
        self.depths = (self.depths - self.close_depth) / (self.inf_depth -
                                                            self.close_depth + torch.finfo(self.depths.dtype).eps)

        # generate dummy pose
        self.poses = torch.stack(
            [torch.eye(4)] * len(paths_img)).float()

        # bds, torch.Size([1, 2])
        self.per_cam_near_fars = torch.Tensor([[1e-6, 1.]])

        self.median_imgs, _ = torch.median(imgs.reshape(
            len(paths_img), intrinsics.height, intrinsics.width, 3), dim=0)
        self.median_imgs = self.median_imgs.reshape(
            1, *self.median_imgs.shape)

        self.mask_weights = self.masks.clone()
        self.mask_weights = torch.Tensor(
            1.0 - self.mask_weights).to(torch.device("cpu"))

        self.mask_weights = self.mask_weights.reshape(
            len(paths_img), intrinsics.height, intrinsics.width)
        freq = (1 - self.mask_weights).sum(0)
        self.p = freq / torch.sqrt((torch.pow(freq, 2)).sum())
        self.mask_weights = self.mask_weights * \
            (1.0 + self.p * self.p_ratio)
        self.mask_weights = self.mask_weights.reshape(
            -1) / torch.sum(self.mask_weights)
        # self.global_translation = torch.tensor([0, 0, 2.])
        # self.global_scale = torch.tensor([0.5, 0.6, 1])
        # Normalize timestamps between -1, 1

        self.timestamps = (timestamps.float() / max(timestamps)) * 2 - 1

        if split == 'train':
            self.timestamps = self.timestamps[:, None, None].repeat(
                1, intrinsics.height, intrinsics.width).reshape(-1)  # [n_frames * h * w]
        assert self.timestamps.min() >= - \
            1.0 and self.timestamps.max() <= 1.0, "timestamps out of range."
        if imgs is not None and imgs.dtype != torch.uint8:
            imgs = (imgs * 255).to(torch.uint8)
        if self.median_imgs is not None and self.median_imgs.dtype != torch.uint8:
            self.median_imgs = (self.median_imgs * 255).to(torch.uint8)
        if split == 'train':
            imgs = imgs.view(-1, imgs.shape[-1])
            self.masks = self.masks.view(-1, self.masks.shape[-1])
            self.depths = self.depths.view(-1, self.depths.shape[-1])
        elif imgs is not None:
            imgs = imgs.view(-1, intrinsics.height *
                             intrinsics.width, imgs.shape[-1])

        # ISG/IST weights are computed on original data.
        weights_subsampled = 1
        if scene_bbox is not None:
            scene_bbox = torch.tensor(scene_bbox)
        else:
            scene_bbox = get_bbox(
                datadir, is_contracted=contraction, dset_type='llff')
        super().__init__(
            datadir=datadir,
            split=split,
            batch_size=batch_size,
            is_ndc=ndc,
            is_contracted=contraction,
            scene_bbox=scene_bbox,
            rays_o=None,
            rays_d=None,
            intrinsics=intrinsics,
            imgs=imgs,
            sampling_weights=None,  # Start without importance sampling, by default
            weights_subsampled=weights_subsampled,
        )

        self.isg_weights = None
        self.ist_weights = None

        if split == "train":  # Only use importance sampling with DyNeRF videos
            if self.maskIS:
                if os.path.exists(os.path.join(datadir, f"isg_weights_masked.pt")):
                    self.isg_weights = torch.load(
                        os.path.join(datadir, f"isg_weights_masked.pt"))
                    log.info(
                        f"Reloaded {self.isg_weights.shape[0]} ISG weights masked from file.")
                else:
                    # Precompute ISG weights
                    t_s = time.time()
                    gamma = 1e-3 if self.keyframes else 2e-2
                    self.isg_weights = dynerf_isg_weight(
                        imgs.view(-1, intrinsics.height,
                                  intrinsics.width, imgs.shape[-1]),
                        median_imgs=self.median_imgs, gamma=gamma, masks=self.masks.view(-1, intrinsics.height, intrinsics.width))
                    # Normalize into a probability distribution, to speed up sampling
                    self.isg_weights = (
                        self.isg_weights.reshape(-1) / torch.sum(self.isg_weights))
                    torch.save(self.isg_weights, os.path.join(
                        datadir, f"isg_weights_masked.pt"))
                    t_e = time.time()
                    log.info(
                        f"Computed {self.isg_weights.shape[0]} ISG weights masked in {t_e - t_s:.2f}s.")
            else:
                if os.path.exists(os.path.join(datadir, f"isg_weights.pt")):
                    self.isg_weights = torch.load(
                        os.path.join(datadir, f"isg_weights.pt"))
                    log.info(
                        f"Reloaded {self.isg_weights.shape[0]} ISG weights from file.")
                else:
                    # Precompute ISG weights
                    t_s = time.time()
                    gamma = 1e-3 if self.keyframes else 2e-2
                    self.isg_weights = dynerf_isg_weight(
                        imgs.view(-1, intrinsics.height,
                                  intrinsics.width, imgs.shape[-1]),
                        median_imgs=self.median_imgs, gamma=gamma)
                    # Normalize into a probability distribution, to speed up sampling
                    self.isg_weights = (
                        self.isg_weights.reshape(-1) / torch.sum(self.isg_weights))
                    torch.save(self.isg_weights, os.path.join(
                        datadir, f"isg_weights.pt"))
                    t_e = time.time()
                    log.info(
                        f"Computed {self.isg_weights.shape[0]} ISG weights in {t_e - t_s:.2f}s.")

            if self.maskIS:
                if os.path.exists(os.path.join(datadir, f"ist_weights_masked.pt")):
                    self.ist_weights = torch.load(
                        os.path.join(datadir, f"ist_weights_masked.pt"))
                    log.info(
                        f"Reloaded {self.ist_weights.shape[0]} IST weights nasked from file.")
                else:
                    # Precompute IST weights
                    t_s = time.time()
                    self.ist_weights = dynerf_ist_weight(
                        imgs.view(-1, self.img_h, self.img_w, imgs.shape[-1]),
                        num_cameras=self.median_imgs.shape[0], masks=self.masks.view(-1, intrinsics.height, intrinsics.width), p=self.p, ratio=self.frequency_ratio)
                    # Normalize into a probability distribution, to speed up sampling
                    self.ist_weights = (
                        self.ist_weights.reshape(-1) / torch.sum(self.ist_weights))
                    torch.save(self.ist_weights, os.path.join(
                        datadir, f"ist_weights_masked.pt"))
                    t_e = time.time()
                    log.info(
                        f"Computed {self.ist_weights.shape[0]} IST weights masked in {t_e - t_s:.2f}s.")
            else:
                if os.path.exists(os.path.join(datadir, f"ist_weights.pt")):
                    self.ist_weights = torch.load(
                        os.path.join(datadir, f"ist_weights.pt"))
                    log.info(
                        f"Reloaded {self.ist_weights.shape[0]} IST weights from file.")
                else:
                    # Precompute IST weights
                    t_s = time.time()
                    self.ist_weights = dynerf_ist_weight(
                        imgs.view(-1, self.img_h, self.img_w, imgs.shape[-1]),
                        num_cameras=self.median_imgs.shape[0])
                    # Normalize into a probability distribution, to speed up sampling
                    self.ist_weights = (
                        self.ist_weights.reshape(-1) / torch.sum(self.ist_weights))
                    torch.save(self.ist_weights, os.path.join(
                        datadir, f"ist_weights.pt"))
                    t_e = time.time()
                    log.info(
                        f"Computed {self.ist_weights.shape[0]} IST weights in {t_e - t_s:.2f}s.")

        if self.isg:
            self.enable_isg()
        if self.sample_from_masks:
            self.enable_mask()
        log.info(f"VideoDataset contracted={self.is_contracted}, ndc={self.is_ndc}. "
                 f"Loaded {self.split} set from {self.datadir}: "
                 f"{len(self.poses)} images of size {self.img_h}x{self.img_w}. "
                 f"Images loaded: {self.imgs is not None}. "
                 f"{len(torch.unique(timestamps))} timestamps. Near-far: {self.per_cam_near_fars}. "
                 f"ISG={self.isg}, IST={self.ist}, weights_subsampled={self.weights_subsampled}. "
                 f"Sampling without replacement={self.use_permutation}. {intrinsics}")

    def enable_isg(self):
        self.isg = True
        self.ist = False
        self.sampling_weights = self.isg_weights
        log.info(f"Enabled ISG weights.")

    def enable_mask(self):
        self.sampling_weights = self.mask_weights
        log.info(f"Using masks as weights in endonerf's method.")

    def switch_isg2ist(self):
        self.isg = False
        self.ist = True
        self.sampling_weights = self.ist_weights
        log.info(f"Switched from ISG to IST weights.")

    def __getitem__(self, index):
        h = self.intrinsics.height
        w = self.intrinsics.width
        dev = "cpu"
        assert self.weights_subsampled == 1  # make sure the weights are not subsampled
        if self.split == 'train':
            # [batch_size // (weights_subsampled**2)]
            index = self.get_rand_ids(index)
            if self.weights_subsampled == 1 or self.sampling_weights is None:
                # Nothing special to do, either weights_subsampled = 1, or not using weights.
                image_id = torch.div(index, h * w, rounding_mode='floor')
                y = torch.remainder(index, h * w).div(w, rounding_mode='floor')
                x = torch.remainder(index, h * w).remainder(w)
            else:
                # We must deal with the fact that ISG/IST weights are computed on a dataset with
                # different 'downsampling' factor. E.g. if the weights were computed on 4x
                # downsampled data and the current dataset is 2x downsampled, `weights_subsampled`
                # will be 4 / 2 = 2.
                # Split each subsampled index into its 16 components in 2D.
                hsub, wsub = h // self.weights_subsampled, w // self.weights_subsampled
                image_id = torch.div(index, hsub * wsub, rounding_mode='floor')
                ysub = torch.remainder(
                    index, hsub * wsub).div(wsub, rounding_mode='floor')
                xsub = torch.remainder(index, hsub * wsub).remainder(wsub)
                # xsub, ysub is the first point in the 4x4 square of finely sampled points
                x, y = [], []
                for ah in range(self.weights_subsampled):
                    for aw in range(self.weights_subsampled):
                        x.append(xsub * self.weights_subsampled + aw)
                        y.append(ysub * self.weights_subsampled + ah)
                x = torch.cat(x)
                y = torch.cat(y)
                image_id = image_id.repeat(self.weights_subsampled ** 2)
                # Inverse of the process to get x, y from index. image_id stays the same.
                index = x + y * w + image_id * h * w
            x, y = x + 0.5, y + 0.5
        else:
            image_id = [index]
            x, y = create_meshgrid(
                height=h, width=w, dev=dev, add_half=True, flat=True)

        out = {
            "timestamps": self.timestamps[index],      # (num_rays or 1, )
            "imgs": None,
        }

        # if self.split == 'train': # can be simplified since we only have one camera
        #     num_frames_per_camera = len(self.imgs) // (len(self.per_cam_near_fars) * h * w)
        #     camera_id = torch.div(image_id, num_frames_per_camera, rounding_mode='floor')  # (num_rays)
        #     out['near_fars'] = self.per_cam_near_fars[camera_id, :]
        # else:
        #     out['near_fars'] = self.per_cam_near_fars  # Only one test camera

        out['near_fars'] = self.per_cam_near_fars  # we only have one camera

        if self.imgs is not None:
            out['imgs'] = (self.imgs[index] / 255.0).view(-1,
                                                          self.imgs.shape[-1])

        # [num_rays or 1, 3, 4]
        c2w = self.poses[image_id]
        camera_dirs = stack_camera_dirs(
            x, y, self.intrinsics, True)  # [num_rays, 3]
        out['rays_o'], out['rays_d'] = get_rays(
            camera_dirs, c2w, ndc=self.is_ndc, ndc_near=1.0, intrinsics=self.intrinsics,
            normalize_rd=True)                                        # [num_rays, 3]

        imgs = out['imgs']

        # Decide BG color
        bg_color = torch.ones((1, 3), dtype=torch.float32, device=dev)
        if self.split == 'train' and imgs.shape[-1] == 4:
            bg_color = torch.rand((1, 3), dtype=torch.float32, device=dev)
        if self.bg_color == 0:
            bg_color = None
        out['bg_color'] = bg_color
        if self.bg_color == 0:
            bg_color = torch.zeros((1, 3), dtype=torch.float32, device=dev)

        # Alpha compositing
        if imgs is not None and imgs.shape[-1] == 4:
            imgs = imgs[:, :3] * imgs[:, 3:] + bg_color * (1.0 - imgs[:, 3:])
        out['imgs'] = imgs  # [num_sample, 3]

        # sample depth
        if hasattr(self, 'depths'):
            out['depths'] = self.depths[index].view(-1, self.depths.shape[-1])
        return out

def get_bbox(datadir: str, dset_type: str, is_contracted=False) -> torch.Tensor:
    """Returns a default bounding box based on the dataset type, and contraction state.

    Args:
        datadir (str): Directory where data is stored
        dset_type (str): A string defining dataset type (e.g. synthetic, llff)
        is_contracted (bool): Whether the dataset will use contraction

    Returns:
        Tensor: 3x2 bounding box tensor
    """
    if is_contracted:
        radius = 2
    elif dset_type == 'synthetic':
        radius = 1.5
    elif dset_type == 'llff':
        return torch.tensor([[-3.0, -1.67, -1.2], [3.0, 1.67, 1.2]])
    else:
        radius = 1.3
    return torch.tensor([[-radius, -radius, -radius], [radius, radius, radius]])


def fetch_360vid_info(frame: Dict[str, Any]):
    timestamp = None
    fp = frame['file_path']
    if '_r' in fp:
        timestamp = int(fp.split('t')[-1].split('_')[0])
    if 'r_' in fp:
        pose_id = int(fp.split('r_')[-1])
    else:
        pose_id = int(fp.split('r')[-1])
    if timestamp is None:  # will be None for dnerf
        timestamp = frame['time']
    return timestamp, pose_id


def load_360video_frames(datadir, split, max_cameras: int, max_tsteps: Optional[int]) -> Tuple[Any, Any]:
    with open(os.path.join(datadir, f"transforms_{split}.json"), 'r') as fp:
        meta = json.load(fp)
    frames = meta['frames']

    timestamps = set()
    pose_ids = set()
    fpath2poseid = defaultdict(list)
    for frame in frames:
        timestamp, pose_id = fetch_360vid_info(frame)
        timestamps.add(timestamp)
        pose_ids.add(pose_id)
        fpath2poseid[frame['file_path']].append(pose_id)
    timestamps = sorted(timestamps)
    pose_ids = sorted(pose_ids)

    if max_cameras is not None:
        num_poses = min(len(pose_ids), max_cameras or len(pose_ids))
        subsample_poses = int(round(len(pose_ids) / num_poses))
        pose_ids = set(pose_ids[::subsample_poses])
        log.info(
            f"Selected subset of {len(pose_ids)} camera poses: {pose_ids}.")

    if max_tsteps is not None:
        num_timestamps = min(len(timestamps), max_tsteps or len(timestamps))
        subsample_time = int(math.floor(
            len(timestamps) / (num_timestamps - 1)))
        timestamps = set(timestamps[::subsample_time])
        log.info(
            f"Selected subset of timestamps: {sorted(timestamps)} of length {len(timestamps)}")

    sub_frames = []
    for frame in frames:
        timestamp, pose_id = fetch_360vid_info(frame)
        if timestamp in timestamps and pose_id in pose_ids:
            sub_frames.append(frame)
    # We need frames to be sorted by pose_id
    sub_frames = sorted(sub_frames, key=lambda f: fpath2poseid[f['file_path']])
    return sub_frames, meta


def load_llffvideo_poses(datadir: str,
                         downsample: float,
                         split: str,
                         near_scaling: float) -> Tuple[
        torch.Tensor, torch.Tensor, Intrinsics, List[str]]:
    """Load poses and metadata for LLFF video.

    Args:
        datadir (str): Directory containing the videos and pose information
        downsample (float): How much to downsample videos. The default for LLFF videos is 2.0
        split (str): 'train' or 'test'.
        near_scaling (float): How much to scale the near bound of poses.

    Returns:
        Tensor: A tensor of size [N, 4, 4] containing c2w poses for each camera.
        Tensor: A tensor of size [N, 2] containing near, far bounds for each camera.
        Intrinsics: The camera intrinsics. These are the same for every camera.
        List[str]: List of length N containing the path to each camera's data.
    """
    poses, near_fars, intrinsics = load_llff_poses_helper(
        datadir, downsample, near_scaling)

    videopaths = np.array(
        glob.glob(os.path.join(datadir, '*.mp4')))  # [n_cameras]
    assert poses.shape[0] == len(videopaths), \
        'Mismatch between number of cameras and number of poses!'
    videopaths.sort()
    # The first camera is reserved for testing, following https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0
    if split == 'train':
        split_ids = np.arange(1, poses.shape[0])
    elif split == 'test':
        split_ids = np.array([0])
    else:
        split_ids = np.arange(poses.shape[0])
    if 'coffee_martini' in datadir:
        # https://github.com/fengres/mixvoxels/blob/0013e4ad63c80e5f14eb70383e2b073052d07fba/dataLoader/llff_video.py#L323
        log.info(f"Deleting unsynchronized camera from coffee-martini video.")
        split_ids = np.setdiff1d(split_ids, 12)
    poses = torch.from_numpy(poses[split_ids])
    near_fars = torch.from_numpy(near_fars[split_ids])
    videopaths = videopaths[split_ids].tolist()
    return poses, near_fars, intrinsics, videopaths


def load_llffvideo_data(videopaths: List[str],
                        cam_poses: torch.Tensor,
                        intrinsics: Intrinsics,
                        split: str,
                        keyframes: bool,
                        keyframes_take_each: Optional[int] = None,
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if keyframes and (keyframes_take_each is None or keyframes_take_each < 1):
        raise ValueError(f"'keyframes_take_each' must be a positive number, "
                         f"but is {keyframes_take_each}.")

    loaded = parallel_load_images(
        dset_type="video",
        tqdm_title=f"Loading {split} data",
        num_images=len(videopaths),
        paths=videopaths,
        poses=cam_poses,
        out_h=intrinsics.height,
        out_w=intrinsics.width,
        load_every=keyframes_take_each if keyframes else 1,
    )
    imgs, poses, median_imgs, timestamps = zip(*loaded)
    # Stack everything together
    timestamps = torch.cat(timestamps, 0)  # [N]
    poses = torch.cat(poses, 0)            # [N, 3, 4]
    imgs = torch.cat(imgs, 0)              # [N, h, w, 3]
    median_imgs = torch.stack(median_imgs, 0)  # [num_cameras, h, w, 3]

    return poses, imgs, timestamps, median_imgs


@torch.no_grad()
def dynerf_isg_weight(imgs, median_imgs, gamma, masks=None):
    # imgs is [num_cameras * num_frames, h, w, 3]
    # median_imgs is [num_cameras, h, w, 3]
    assert imgs.dtype == torch.uint8
    assert median_imgs.dtype == torch.uint8
    num_cameras, h, w, c = median_imgs.shape
    if masks is not None:
        for i in range(len(masks)):
            for j in range(3):
                imgs[i, :, :, j] = imgs[i, :, :, j]*(1-masks[i])
        median_imgs, _ = torch.median(imgs.reshape(-1, h, w, 3), dim=0)
        median_imgs = median_imgs.reshape(1, *median_imgs.shape)
    squarediff = (
        imgs.view(num_cameras, -1, h, w, c)
            .float()  # creates new tensor, so later operations can be in-place
            .div_(255.0)
            .sub_(
                median_imgs[:, None, ...].float().div_(255.0)
            )
            .square_()  # noqa
    )  # [num_cameras, num_frames, h, w, 3]
    # differences = median_imgs[:, None, ...] - imgs.view(num_cameras, -1, h, w, c)  # [num_cameras, num_frames, h, w, 3]
    # squarediff = torch.square_(differences)
    psidiff = squarediff.div_(squarediff + gamma**2)
    # [num_cameras, num_frames, h, w]
    psidiff = (1./3) * torch.sum(psidiff, dim=-1)
    if masks is not None:
        assert len(masks) == psidiff.shape[1]
        masks = np.stack(masks, axis=0).astype(np.float64)
        for i in range(len(masks)):
            psidiff[0][i] *= (1-masks[i])
    return psidiff  # valid probabilities, each in [0, 1]


@torch.no_grad()
# DyNerf uses alpha=0.1
def dynerf_ist_weight(imgs, num_cameras, alpha=0.1, frame_shift=25, masks=None, p=None, ratio=None):
    assert imgs.dtype == torch.uint8
    N, h, w, c = imgs.shape
    if masks is not None:
        for i in range(len(masks)):
            for j in range(3):
                imgs[i, :, :, j] = imgs[i, :, :, j]*(1-masks[i])
    # [num_cameras, num_timesteps, h, w, 3]
    frames = imgs.view(num_cameras, -1, h, w, c).float()
    max_diff = None
    shifts = list(range(frame_shift + 1))[1:]
    for shift in shifts:
        shift_left = torch.cat(
            [frames[:, shift:, ...], torch.zeros(num_cameras, shift, h, w, c)], dim=1)
        shift_right = torch.cat(
            [torch.zeros(num_cameras, shift, h, w, c), frames[:, :-shift, ...]], dim=1)
        mymax = torch.maximum(torch.abs_(shift_left - frames),
                              torch.abs_(shift_right - frames))
        if max_diff is None:
            max_diff = mymax
        else:
            # [num_timesteps, h, w, 3]
            max_diff = torch.maximum(max_diff, mymax)
    max_diff = torch.mean(max_diff, dim=-1)  # [num_timesteps, h, w]
    max_diff = max_diff.clamp_(min=alpha)
    # print(max_diff, max_diff.shape)
    if masks is not None:
        assert len(masks) == max_diff.shape[1]
        masks = np.stack(masks, axis=0).astype(np.float64)
        for i in range(len(masks)):
            max_diff[0][i] *= (1-masks[i])
            max_diff[0][i] *= (1+p*ratio)
    return max_diff


@torch.jit.script
def percentile_torch(t: torch.Tensor, q: float) -> float:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + int(round(.01 * float(q) * (t.numel() - 1)))
    result = t.view(-1).kthvalue(k).values.item()
    return result
