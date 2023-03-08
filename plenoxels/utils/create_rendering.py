"""Entry point for simple renderings, given a trainer and some poses."""
import os
import logging as log
from typing import Union
import open3d as o3d
import cv2
import torch
import numpy as np

from plenoxels.models.lowrank_model import LowrankModel
from plenoxels.utils.my_tqdm import tqdm
from plenoxels.ops.image.io import write_video_to_file
from plenoxels.runners.static_trainer import StaticTrainer
from plenoxels.runners.video_trainer import VideoTrainer


@torch.no_grad()
def render_to_path(trainer: Union[VideoTrainer, StaticTrainer], extra_name: str = "") -> None:
    """Render all poses in the `test_dataset`, saving them to file
    Args:
        trainer: The trainer object which is used for rendering
        extra_name: String to append to the saved file-name
    """
    dataset = trainer.test_dataset

    pb = tqdm(total=len(dataset), desc=f"Rendering scene")
    frames = []
    for img_idx, data in enumerate(dataset):
        ts_render = trainer.eval_step(data)

        if isinstance(dataset.img_h, int):
            img_h, img_w = dataset.img_h, dataset.img_w
        else:
            img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]
        preds_rgb = (
            ts_render["rgb"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
            .mul(255.0)
            .byte()
            .numpy()
        )
        frames.append(preds_rgb)
        pb.update(1)
    pb.close()

    out_fname = os.path.join(
        trainer.log_dir, f"rendering_path_{extra_name}.mp4")
    write_video_to_file(out_fname, frames)
    log.info(f"Saved rendering path with {len(frames)} frames to {out_fname}")


def generate_pointclouds(rgb_np, depth_np, crop_left_size=0, depth_filter=None, vis_rgbd=False):
    # use o3d to form rgbd image and convert to point cloud
    # set render params for DaVinci endoscopic
    hwf = [512, 640, 569.46820041]
    def check_image_type(img): return (
        img * 255).astype(np.uint8) if img.dtype == np.float32 else img
    if crop_left_size > 0:
        rgb_np = rgb_np[:, crop_left_size:, :]  # crop left side, HWC
        depth_np = depth_np[:, crop_left_size:]  # crop left side, HW

    if depth_filter is not None:
        depth_np = cv2.bilateralFilter(
            depth_np, depth_filter[0], depth_filter[1], depth_filter[2])

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(check_image_type(rgb_np)),
        o3d.geometry.Image(depth_np),
        convert_rgb_to_intensity=False)
    if vis_rgbd:
        import matplotlib.pyplot as plt
        plt.subplot(1, 2, 1)
        plt.title('RGB image')
        plt.imshow(rgbd_image.color)
        plt.subplot(1, 2, 2)
        plt.title('Depth image')
        plt.imshow(rgbd_image.depth)
        plt.colorbar()
        plt.show()

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            hwf[1], hwf[0], hwf[2], hwf[2], hwf[1] / 2, hwf[0] / 2)
    )
    return pcd


@torch.no_grad()
def render_to_path_with_pointcloud(trainer: Union[VideoTrainer, StaticTrainer], extra_name: str = "", save_pc_num: int = -1) -> None:
    """Render all poses in the `test_dataset`, saving them to file
    Args:
        trainer: The trainer object which is used for rendering
        extra_name: String to append to the saved file-name
        save_pc_num: Whether to save the pointclouds, -2 for not save, -1 for save_all, other number for save specific number
    """
    dataset = trainer.test_dataset

    pb = tqdm(total=len(dataset), desc=f"Rendering scene")
    frames, depths = [], []
    for img_idx, data in enumerate(dataset):
        ts_render = trainer.eval_step(data)

        if isinstance(dataset.img_h, int):
            img_h, img_w = dataset.img_h, dataset.img_w
        else:
            img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]
        preds_rgb = (
            ts_render["rgb"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
            .mul(255.0)
            .byte()
            .numpy()
        )
        frames.append(preds_rgb)
        if "depth" in ts_render:
            depths.append(ts_render["depth"].cpu().reshape(
                img_h, img_w)[..., None])
            np.save(os.path.join(trainer.log_dir, 'estm',
                    'depth'+str(len(depths)-1)+'.npy'), depths[-1])
        # if "accumulation" in ts_render:
            # acc = ts_render['accumulation'].cpu().reshape(img_h, img_w)
            # cv2.imwrite( "acc.png",(ts_render['accumulation'].cpu().reshape(img_h, img_w).numpy()*255.).astype(np.uint8))
        pb.update(1)
    pb.close()

    # out_fname = os.path.join(
    #     trainer.log_dir, f"rendering_path_{extra_name}.mp4")
    # write_video_to_file(out_fname, frames)
    # log.info(f"Saved rendering path with {len(frames)} frames to {out_fname}")

    # if save_pc_num == -1:
    #     save_pc_num = range(len(frames))
    # elif save_pc_num == -2:
    #     return
    # else:
    #     save_pc_num = [min(save_pc_num, len(frames))]
    # for idx in save_pc_num:
    #     pcd = generate_pointclouds(frames[idx], depths[idx])
    #     o3d.io.write_point_cloud(os.path.join(
    #         trainer.log_dir, f"pcd_{idx}.ply"), pcd)


def normalize_for_disp(img):
    img = img - torch.min(img)
    img = img / torch.max(img)
    return img


@torch.no_grad()
def decompose_space_time_dynerf(trainer: StaticTrainer, extra_name: str = "") -> None:
    """Render space-time decomposition videos for poses in the `test_dataset`.

    The space-only part of the decomposition is obtained by setting the time-planes to 1.
    The time-only part is obtained by simple subtraction of the space-only part from the full
    rendering.

    Args:
        trainer: The trainer object which is used for rendering
        extra_name: String to append to the saved file-name
    """
    chosen_cam_idx = 15
    model: LowrankModel = trainer.model
    dataset = trainer.test_dataset

    # Store original parameters from main field and proposal-network field
    parameters = []
    for multires_grids in model.field.grids:
        parameters.append([grid.data for grid in multires_grids])
    pn_parameters = []
    for pn in model.proposal_networks:
        pn_parameters.append([grid_plane.data for grid_plane in pn.grids])

    camdata = None
    for img_idx, data in enumerate(dataset):
        if img_idx == chosen_cam_idx:
            camdata = data
    if camdata is None:
        raise ValueError(f"Cam idx {chosen_cam_idx} invalid.")

    num_frames = img_idx + 1
    frames = []
    for img_idx in tqdm(range(num_frames), desc="Rendering scene with separate space and time components"):
        # Linearly interpolated timestamp, normalized between -1, 1
        camdata["timestamps"] = torch.Tensor([img_idx / num_frames]) * 2 - 1

        if isinstance(dataset.img_h, int):
            img_h, img_w = dataset.img_h, dataset.img_w
        else:
            img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]

        # Full model: turn on time-planes
        for i in range(len(model.field.grids)):
            for plane_idx in [2, 4, 5]:
                model.field.grids[i][plane_idx].data = parameters[i][plane_idx]
        for i in range(len(model.proposal_networks)):
            for plane_idx in [2, 4, 5]:
                model.proposal_networks[i].grids[plane_idx].data = pn_parameters[i][plane_idx]
        preds = trainer.eval_step(camdata)
        full_out = preds["rgb"].reshape(img_h, img_w, 3).cpu()

        # Space-only model: turn off time-planes
        for i in range(len(model.field.grids)):
            for plane_idx in [2, 4, 5]:  # time-grids off
                model.field.grids[i][plane_idx].data = torch.ones_like(
                    parameters[i][plane_idx])
        for i in range(len(model.proposal_networks)):
            for plane_idx in [2, 4, 5]:
                model.proposal_networks[i].grids[plane_idx].data = torch.ones_like(
                    pn_parameters[i][plane_idx])
        preds = trainer.eval_step(camdata)
        spatial_out = preds["rgb"].reshape(img_h, img_w, 3).cpu()

        # Temporal model: full - space
        temporal_out = normalize_for_disp(full_out - spatial_out)

        frames.append(
            torch.cat([full_out, spatial_out, temporal_out], dim=1)
                 .clamp(0, 1)
                 .mul(255.0)
                 .byte()
                 .numpy()
        )

    out_fname = os.path.join(trainer.log_dir, f"spacetime_{extra_name}.mp4")
    write_video_to_file(out_fname, frames)
    log.info(f"Saved rendering path with {len(frames)} frames to {out_fname}")


@torch.no_grad()
def decompose_space_time(trainer: StaticTrainer, extra_name: str = "") -> None:
    """Render space-time decomposition videos for poses in the `test_dataset`.
    code for EndoNeRF dataset, only have one camera
    The space-only part of the decomposition is obtained by setting the time-planes to 1.
    The time-only part is obtained by simple subtraction of the space-only part from the full
    rendering.

    Args:
        trainer: The trainer object which is used for rendering
        extra_name: String to append to the saved file-name
    """
    chosen_cam_idx = 15  # the frame idx among dataset
    model: LowrankModel = trainer.model
    dataset = trainer.test_dataset

    # Store original parameters from main field and proposal-network field
    parameters = []
    for multires_grids in model.field.grids:
        parameters.append([grid.data for grid in multires_grids])
    pn_parameters = []
    for pn in model.proposal_networks:
        pn_parameters.append([grid_plane.data for grid_plane in pn.grids])

    camdata = None
    for img_idx, data in enumerate(dataset):
        if img_idx == chosen_cam_idx:
            camdata = data
    if camdata is None:
        raise ValueError(f"Cam idx {chosen_cam_idx} invalid.")

    num_frames = img_idx + 1
    frames = []
    for img_idx in tqdm(range(num_frames), desc="Rendering scene with separate space and time components"):
        # Linearly interpolated timestamp, normalized between -1, 1
        camdata["timestamps"] = torch.Tensor([img_idx / num_frames]) * 2 - 1

        if isinstance(dataset.img_h, int):
            img_h, img_w = dataset.img_h, dataset.img_w
        else:
            img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]

        # Full model: turn on time-planes
        for i in range(len(model.field.grids)):
            for plane_idx in [2, 4, 5]:
                model.field.grids[i][plane_idx].data = parameters[i][plane_idx]
        for i in range(len(model.proposal_networks)):
            for plane_idx in [2, 4, 5]:
                model.proposal_networks[i].grids[plane_idx].data = pn_parameters[i][plane_idx]
        preds = trainer.eval_step(camdata)
        full_out = preds["rgb"].reshape(img_h, img_w, 3).cpu()

        # Space-only model: turn off time-planes
        for i in range(len(model.field.grids)):
            for plane_idx in [2, 4, 5]:  # time-grids off
                model.field.grids[i][plane_idx].data = torch.ones_like(
                    parameters[i][plane_idx])
        for i in range(len(model.proposal_networks)):
            for plane_idx in [2, 4, 5]:
                model.proposal_networks[i].grids[plane_idx].data = torch.ones_like(
                    pn_parameters[i][plane_idx])
        preds = trainer.eval_step(camdata)
        spatial_out = preds["rgb"].reshape(img_h, img_w, 3).cpu()

        # Temporal model: full - space
        temporal_out = normalize_for_disp(full_out - spatial_out)

        frames.append(
            torch.cat([full_out, spatial_out, temporal_out], dim=1)
                 .clamp(0, 1)
                 .mul(255.0)
                 .byte()
                 .numpy()
        )

    out_fname = os.path.join(trainer.log_dir, f"spacetime_{extra_name}.mp4")
    write_video_to_file(out_fname, frames)
    log.info(f"Saved rendering path with {len(frames)} frames to {out_fname}")
