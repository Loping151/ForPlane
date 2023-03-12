import logging as log
import math
import os
from collections import defaultdict
from typing import Dict, MutableMapping, Union, Any, List

import pandas as pd
import torch
import torch.utils.data

from lerplanes.datasets.video_datasets import Video360Dataset, VideoEndoDataset
from lerplanes.utils.ema import EMA
from lerplanes.utils.my_tqdm import tqdm
from lerplanes.ops.image import metrics
from lerplanes.ops.image.io import write_video_to_file
from lerplanes.models.lowrank_model import LowrankModel
from .base_trainer import BaseTrainer, init_dloader_random, initialize_model
from .regularization import (
    PlaneTV, TimeSmoothness, HistogramLoss, L1TimePlanes, DistortionLoss, DepthLossHuber
)
import imageio
import numpy as np
from utils.eval_rgb import img2mse, mse2psnr, ssim, lpips

class VideoTrainer(BaseTrainer):
    def __init__(self,
                 tr_loader: torch.utils.data.DataLoader,
                 tr_dset: torch.utils.data.TensorDataset,
                 ts_dset: torch.utils.data.TensorDataset,
                 num_steps: int,
                 logdir: str,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
                 save_outputs: bool,
                 isg_step: int,
                 ist_step: int,
                 device: Union[str, torch.device],
                 **kwargs
                 ):
        self.train_dataset = tr_dset
        self.test_dataset = ts_dset
        self.ist_step = ist_step
        self.isg_step = isg_step
        self.save_video = save_outputs
        self.endo = False
        if kwargs.get('endo', None):
            self.endo = True
        # Switch to compute extra video metrics (FLIP, JOD)
        self.compute_video_metrics = True # may switch to True
        super().__init__(
            train_data_loader=tr_loader,
            num_steps=num_steps,
            logdir=logdir,
            expname=expname,
            train_fp16=train_fp16,
            save_every=save_every,
            valid_every=valid_every,
            save_outputs=False,  # False since we're saving video
            device=device,
            **kwargs)

    def eval_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().eval_step(data, **kwargs)
        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            timestamp = data["timestamps"]
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]
            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            preds = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).to(self.device)
                outputs = self.model(
                    rays_o_b, rays_d_b, timestamps=timestamps_d_b, bg_color=bg_color,
                    near_far=near_far)
                for k, v in outputs.items():
                    if "rgb" in k or "depth" in k or "accumulation":
                        preds[k].append(v.cpu())
        return {k: torch.cat(v, 0) for k, v in preds.items()}

    def train_step(self, data: Dict[str, Union[int, torch.Tensor]], **kwargs):
        scale_ok = super().train_step(data, **kwargs)

        if self.global_step == self.isg_step:
            self.train_dataset.enable_isg()
            raise StopIteration  # Whenever we change the dataset
        if self.global_step == self.ist_step:
            self.train_dataset.switch_isg2ist()
            raise StopIteration  # Whenever we change the dataset

        return scale_ok

    def post_step(self, progress_bar):
        super().post_step(progress_bar)

    def pre_epoch(self):
        super().pre_epoch()
        # Reset randomness in train-dataset
        self.train_dataset.reset_iter()

    # @torch.no_grad()
    # def validate(self):
    #     dataset = self.test_dataset
    #     per_scene_metrics: Dict[str, Union[float, List]] = defaultdict(list)
    #     pred_frames, out_depths = [], []
    #     pb = tqdm(total=len(dataset), desc=f"Test scene ({dataset.name})")
    #     for img_idx, data in enumerate(dataset):
    #         preds = self.eval_step(data)
    #         out_metrics, out_img, out_depth = self.evaluate_metrics(
    #             data["imgs"], preds, dset=dataset, img_idx=img_idx, name=None,
    #             save_outputs=self.save_outputs)
    #         pred_frames.append(out_img)
    #         if out_depth is not None:
    #             out_depths.append(out_depth)
    #         for k, v in out_metrics.items():
    #             per_scene_metrics[k].append(v)
    #         pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
    #         pb.update(1)

    #     pb.close()
    #     if self.save_video:
    #         write_video_to_file(
    #             os.path.join(self.log_dir, f"step{self.global_step}.mp4"),
    #             pred_frames
    #         )
    #         if len(out_depths) > 0:
    #             write_video_to_file(
    #                 os.path.join(self.log_dir, f"step{self.global_step}-depth.mp4"),
    #                 out_depths
    #             )
    #     # Calculate JOD (on whole video)
    #     if self.compute_video_metrics:
    #         # per_scene_metrics["JOD"] = metrics.jod(
    #         #     [f[:dataset.img_h, :, :] for f in pred_frames],
    #         #     [f[dataset.img_h: 2*dataset.img_h, :, :] for f in pred_frames],
    #         # )
    #         per_scene_metrics["FLIP"] = metrics.flip(
    #             [f[:, :dataset.img_w, :] for f in pred_frames],
    #             [f[:, dataset.img_w: 2*dataset.img_w, :] for f in pred_frames],
    #         )

    #     val_metrics = [
    #         self.report_test_metrics(per_scene_metrics, extra_name=f'step_{self.global_step}'),
    #     ]
    #     df = pd.DataFrame.from_records(val_metrics)
    #     df.to_csv(os.path.join(self.log_dir, f"test_metrics_step{self.global_step}.csv"))
    #     # here we save all metrics to a csv file for further analysis
    #     save_all_metrics(per_scene_metrics, os.path.join(self.log_dir, f"metrics_all_step{self.global_step}.csv"))

    # @torch.no_grad()
    # def validate(self):
    #     # 夺舍
    #     self.validate_endo(self)

    @torch.no_grad()
    def validate(self): # Todo: @kailing
        dataset = self.test_dataset
        pred_frames, out_depths = [], []
        per_scene_metrics: Dict[str, Union[float, List]] = defaultdict(list)

        device = torch.device("cpu")
        stdshape = (len(dataset), dataset.img_h, dataset.img_w, 3)
        masks = (dataset.masks).reshape(stdshape[:-1])
        flip_mask = np.array(masks.clone())
        masks = torch.Tensor(1.0 - masks).to(device).unsqueeze(-1)
        gts = (dataset.imgs/255).reshape(stdshape)
        # masks = np.stack(mask_list, axis=0).astype(np.float32) / 255.0
        gts = np.stack(gts, axis=0).astype(np.float64)

        # gt_dir = os.path.join(data_dirs, 'images')
        # mask_dir = os.path.join(data_dirs, 'gt_masks')
        # img_dir = os.path.join(logdir, 'estm')
        # gt_all = [imageio.imread(os.path.join(gt_dir, fn)) for fn in sorted(os.listdir(gt_dir)) if fn.endswith('.png')]
        # mask_all = [imageio.imread(os.path.join(mask_dir, fn)) for fn in sorted(os.listdir(mask_dir)) if fn.endswith('.png')]
        # gt_list = []
        # mask_list = []
        img_list = []
        # indexex = []
        logdir = self.log_dir
        if not os.path.exists(os.path.join(logdir, 'estm')):
            os.mkdir(os.path.join(logdir, 'estm'))
        for img_idx, data in tqdm(enumerate(dataset)):
            preds = self.eval_step(data)
            out_metrics, out_img, out_depth, out_pred= self.evaluate_metrics(
                data["imgs"], preds, dset=dataset, img_idx=img_idx, name=None,
                save_outputs=self.save_outputs, out_pred=True)
            pred_frames.append(out_img)
            if out_depth is not None:
                out_depths.append(out_depth)
            for k, v in out_metrics.items():
                per_scene_metrics[k].append(v)
            img_list.append(out_pred.clone())
            imageio.imwrite(os.path.join(logdir, 'estm', str(img_idx)+'.png'), torch.round(out_pred*255).to(torch.uint8))

        # for img_idx, data in tqdm(enumerate(dataset)):
        #     preds = self.eval_step(data)
        #     if isinstance(dataset.img_h, int):
        #         img_h, img_w = dataset.img_h, dataset.img_w
        #     else:
        #         img_h, img_w = dataset.img_h[img_idx], dataset.img_w[img_idx]
        #     preds_rgb = (
        #         preds["rgb"]
        #         .reshape(img_h, img_w, 3)
        #         .cpu()
        #         .clamp(0, 1)
        #     )
        #     if not torch.isfinite(preds_rgb).all():
        #         log.warning(f"Predictions have {torch.isnan(preds_rgb).sum()} NaNs, "
        #                     f"{torch.isinf(preds_rgb).sum()} infs.")
        #         preds_rgb = torch.nan_to_num(preds_rgb, nan=0.0)

        if self.save_video:
            write_video_to_file(
                os.path.join(self.log_dir, f"step{self.global_step}.mp4"),
                pred_frames
            )
            if len(out_depths) > 0:
                write_video_to_file(
                    os.path.join(self.log_dir, f"step{self.global_step}-depth.mp4"),
                    out_depths
                )
        # Calculate JOD (on whole video)
        if self.compute_video_metrics:
            # per_scene_metrics["JOD"] = metrics.jod(
            #     [f[:dataset.img_h, :, :] for f in pred_frames],
            #     [f[dataset.img_h: 2*dataset.img_h, :, :] for f in pred_frames],
            # )
            for i in range(len(pred_frames)):
                pred_frames[i][:,:dataset.img_w] = (1-flip_mask[i])[:,:,np.newaxis] * pred_frames[i][:,:dataset.img_w]
                pred_frames[i][:,dataset.img_w: 2*dataset.img_w] = (1-flip_mask[i])[:,:,np.newaxis] * pred_frames[i][:,dataset.img_w: 2*dataset.img_w]

            per_scene_metrics["FLIP"] = metrics.flip(
                [f[:, :dataset.img_w, :] for f in pred_frames],
                [f[:, dataset.img_w: 2*dataset.img_w, :] for f in pred_frames],
            )

        val_metrics = [
            self.report_test_metrics(per_scene_metrics, extra_name=f'step_{self.global_step}'),
        ]
        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_step{self.global_step}.csv"))
        # here we save all metrics to a csv file for further analysis
        save_all_metrics(per_scene_metrics, os.path.join(self.log_dir, f"metrics_all_step{self.global_step}.csv")) 

            # if not os.path.exists(os.path.join(logdir, 'gt_img')):
            #     os.mkdir(os.path.join(logdir, 'gt_img'))
            # if not os.path.exists(os.path.join(logdir, 'gt_mask')):
            #     os.mkdir(os.path.join(logdir, 'gt_mask'))
            # if debug:
        
                # imageio.imwrite(os.path.join(logdir, 'gt_img', str(img_idx)+'.png'), gt_all[img_idx])
                # imageio.imwrite(os.path.join(logdir, 'gt_mask', str(img_idx)+'.png'), mask_all[img_idx])
            # indexex.append(img_idx)
            # gt_list.append(gt_all[img_idx])
            # mask_list.append(mask_all[img_idx])
        # img_list = [imageio.imread(os.path.join(logdir, 'estm', fn)) for fn in sorted(os.listdir(os.path.join(logdir, 'estm'))) if fn.endswith('.png')]
        # gt_list = [imageio.imread(os.path.join(gt_dir, fn)) for fn in sorted(os.listdir(gt_dir)) if fn.endswith('.png')]
        # mask_list = [imageio.imread(os.path.join(mask_dir, fn)) for fn in sorted(os.listdir(mask_dir)) if fn.endswith('.png')]
        # img_list = [imageio.imread(os.path.join(img_dir, fn)) for fn in sorted(os.listdir(img_dir)) if fn.endswith('.png')]
        # gt_list = [gt_list[i] for i in indexex]
        # mask_list = [mask_list[i] for i in indexex]

        imgs = np.stack(img_list, axis=0).astype(np.float64)
        gts = torch.Tensor(gts).to(device) * masks
        imgs = torch.Tensor(imgs).to(device) * masks
        print('Shapes (gt, imgs, masks):', gts.shape, imgs.shape, masks.shape)
        print('running endo eval')
        mse = img2mse(imgs, gts)
        psnr = mse2psnr(mse)
        ssim_ = ssim(imgs, gts, format='NHWC')
        lpips_ = lpips(imgs, gts, format='NHWC')
        print('PSNR:', psnr.item())
        print('SSIM:', ssim_.item())
        print('LPIPS:', torch.mean(lpips_).item())
        with open(os.path.join(self.log_dir, 'endo_log.txt'), 'w') as file:
            file.writelines(('PSNR:', str(psnr.item()), '\n', 'SSIM:', str(ssim_.item()), '\n', 'LPIPS:', str(torch.mean(lpips_).item()), '\n'))
        print('logging endo eval successful.')


    def get_save_dict(self):
        base_save_dict = super().get_save_dict()
        return base_save_dict

    def load_model(self, checkpoint_data, training_needed: bool = True):
        super().load_model(checkpoint_data, training_needed)
        if self.train_dataset is not None:
            if -1 < self.isg_step < self.global_step < self.ist_step:
                self.train_dataset.enable_isg()
            elif -1 < self.ist_step < self.global_step:
                self.train_dataset.switch_isg2ist()

    def init_epoch_info(self):
        ema_weight = 0.9
        loss_info = defaultdict(lambda: EMA(ema_weight))
        return loss_info

    def init_model(self, **kwargs) -> LowrankModel:
        return initialize_model(self, **kwargs)

    def get_regularizers(self, **kwargs):
        return [
            HistogramLoss(kwargs.get('histogram_loss_weight', 0.0)),
            DistortionLoss(kwargs.get('distortion_loss_weight', 0.0)),
            DepthLossHuber(kwargs.get('depth_huber_weight', 0.0), what='field', step_iter=kwargs.get('step_iter', -1)), # temp use 0.2 for huber
            DepthLossHuber(kwargs.get('depth_huber_weight_proposal_net', 0.0), what='proposal_network', step_iter=kwargs.get('step_iter', -1)), # temp use 0.2 for huber
            PlaneTV(kwargs.get('plane_tv_weight', 0.0), what='field'),
            PlaneTV(kwargs.get('plane_tv_weight_proposal_net', 0.0), what='proposal_network'),
            L1TimePlanes(kwargs.get('l1_time_planes', 0.0), what='field'),
            L1TimePlanes(kwargs.get('l1_time_planes_proposal_net', 0.0), what='proposal_network'),
            TimeSmoothness(kwargs.get('time_smoothness_weight', 0.0), what='field'),
            TimeSmoothness(kwargs.get('time_smoothness_weight_proposal_net', 0.0), what='proposal_network'),
        ]

    @property
    def calc_metrics_every(self):
        return 5


def init_tr_data(data_downsample, data_dir, **kwargs):
    isg = kwargs.get('isg', False)
    ist = kwargs.get('ist', False)
    keyframes = kwargs.get('keyframes', False)
    batch_size = kwargs['batch_size']
    if kwargs.get('endo', None):
        log.info(f"Loading VideoEndoDataset with downsample={data_downsample}")
        tr_dset = VideoEndoDataset(
        data_dir, split='train', downsample=data_downsample,
        batch_size=batch_size,
        max_cameras=kwargs.get('max_train_cameras', None),
        max_tsteps=kwargs['max_train_tsteps'] if keyframes else None,
        isg=isg, keyframes=keyframes, contraction=kwargs['contract'], ndc=kwargs['ndc'],
        near_scaling=float(kwargs.get('near_scaling', 0)), ndc_far=float(kwargs.get('ndc_far', 0)),
        scene_bbox=kwargs['scene_bbox'],
        maskIS = kwargs.get('maskIS', False),
        sample_from_masks = kwargs.get('sample_from_masks', False),
        p_ratio = kwargs.get('p_ratio', 1),
        frequency_ratio = kwargs.get('frequency_ratio', None),
        bg_color = kwargs.get('bg_color', 1),
    )
    else:
        log.info(f"Loading Video360Dataset with downsample={data_downsample}")
        tr_dset = Video360Dataset(
            data_dir, split='train', downsample=data_downsample,
            batch_size=batch_size,
            max_cameras=kwargs.get('max_train_cameras', None),
            max_tsteps=kwargs['max_train_tsteps'] if keyframes else None,
            isg=isg, keyframes=keyframes, contraction=kwargs['contract'], ndc=kwargs['ndc'],
            near_scaling=float(kwargs.get('near_scaling', 0)), ndc_far=float(kwargs.get('ndc_far', 0)),
            scene_bbox=kwargs['scene_bbox'],
        )
    if ist:
        tr_dset.switch_isg2ist()  # this should only happen in case we're reloading

    g = torch.Generator()
    g.manual_seed(0)
    tr_loader = torch.utils.data.DataLoader(
        tr_dset, batch_size=None, num_workers=4,  prefetch_factor=4, pin_memory=True,
        worker_init_fn=init_dloader_random, generator=g)
    return {"tr_loader": tr_loader, "tr_dset": tr_dset}


def init_ts_data(data_dir, split, **kwargs):
    if 'dnerf' or 'endo' in data_dir:
        downsample = 1.0
    else:
        downsample = 2.0
    if kwargs.get('endo', None):
        ts_dset = VideoEndoDataset(
            data_dir, split=split, downsample=downsample,
            max_cameras=kwargs.get('max_test_cameras', None), max_tsteps=kwargs.get('max_test_tsteps', None),
            contraction=kwargs['contract'], ndc=kwargs['ndc'],
            near_scaling=float(kwargs.get('near_scaling', 0)), ndc_far=float(kwargs.get('ndc_far', 0)),
            scene_bbox=kwargs['scene_bbox'],
            maskIS = kwargs.get('maskIS', False),
            sample_from_masks = kwargs.get('sample_from_masks', False),
            p_ratio = kwargs.get('p_ratio', 1),
            frequency_ratio = kwargs.get('frequency_ratio', None),
            bg_color = kwargs.get('bg_color', 1),
        )
    else:
        ts_dset = Video360Dataset(
            data_dir, split=split, downsample=downsample,
            max_cameras=kwargs.get('max_test_cameras', None), max_tsteps=kwargs.get('max_test_tsteps', None),
            contraction=kwargs['contract'], ndc=kwargs['ndc'],
            near_scaling=float(kwargs.get('near_scaling', 0)), ndc_far=float(kwargs.get('ndc_far', 0)),
            scene_bbox=kwargs['scene_bbox'],
        )
    return {"ts_dset": ts_dset}


def load_data(data_downsample, data_dirs, validate_only, render_only, **kwargs):
    assert len(data_dirs) == 1
    od: Dict[str, Any] = {}
    if not validate_only and not render_only:
        od.update(init_tr_data(data_downsample, data_dirs[0], **kwargs))
    else:
        od.update(tr_loader=None, tr_dset=None)
    test_split = 'render' if render_only else 'test'
    od.update(init_ts_data(data_dirs[0], split=test_split, **kwargs))
    return od


def save_all_metrics(per_scene_metrics, data_path):

    tensor_to_float = lambda x: x.item()
    # pre-convert the elements to float
    if 'mse' in per_scene_metrics:
        per_scene_metrics['mse'] = list(map(tensor_to_float, per_scene_metrics['mse']))
    # import json
    # with open(data_path, 'w') as f:
    # # Convert the dictionary to a JSON string and write it to disk.
    #     json.dump(per_scene_metrics, f)

    df = pd.DataFrame.from_dict(per_scene_metrics)
    df.to_csv(data_path, index=True)
