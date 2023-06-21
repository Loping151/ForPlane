from typing import Dict, List, Optional, Sequence, Tuple, Union

import nerfacc
import numpy as np
import torch
import torch.nn as nn

from lerplanes.models.density_fields import KPlaneDensityField
from lerplanes.models.kplane_field import KPlaneField
from lerplanes.ops.activations import init_density_activation
from lerplanes.raymarching.ray_samplers import (ProposalNetworkSampler,
                                                RayBundle, RaySamples,
                                                UniformLinDispPiecewiseSampler,
                                                UniformSampler)
from lerplanes.raymarching.spatial_distortions import (SceneContraction,
                                                       SpatialDistortion)
from lerplanes.utils.timer import CudaTimer


class LowrankModel(nn.Module):
    def __init__(self,
                 grid_config: Union[str, List[Dict]],
                 # boolean flags
                 is_ndc: bool,
                 is_contracted: bool,
                 aabb: torch.Tensor,
                 # Model arguments
                 multiscale_res: Sequence[int],
                 density_activation: Optional[str] = 'trunc_exp',
                 concat_features_across_scales: bool = False,
                 linear_decoder: bool = True,
                 linear_decoder_layers: Optional[int] = 1,
                 # Spatial distortion
                 global_translation: Optional[torch.Tensor] = None,
                 global_scale: Optional[torch.Tensor] = None,
                 # occ-sampling arguments
                 occ_grid_reso: int = -1,  # -1 to disable [64, 128, 256]
                 occ_step_size: float = 1e-2, # [4e-3, 1e-3, 1e-4, 1e-2]
                 occ_level: int = 1, # [1, 2]
                 occ_alpha_thres: float = 0.0, # [1e-2, 1e-3, 1e-4]
                 # proposal-sampling arguments
                 num_proposal_iterations: int = 1,
                 use_same_proposal_network: bool = False,
                 proposal_net_args_list: List[Dict] = None,
                 num_proposal_samples: Optional[Tuple[int]] = None,
                 num_samples: Optional[int] = None,
                 single_jitter: bool = False,
                 proposal_warmup: int = 5000,
                 proposal_update_every: int = 5,
                 use_proposal_weight_anneal: bool = True,
                 proposal_weights_anneal_max_num_iters: int = 1000,
                 proposal_weights_anneal_slope: float = 10.0,
                 # appearance embedding (phototourism)
                 use_appearance_embedding: bool = False,
                 appearance_embedding_dim: int = 0,
                 num_images: Optional[int] = None,
                 **kwargs,
                 ):
        super().__init__()
        if isinstance(grid_config, str):
            self.config: List[Dict] = eval(grid_config)
        else:
            self.config: List[Dict] = grid_config
        self.multiscale_res = multiscale_res
        self.is_ndc = is_ndc
        self.is_contracted = is_contracted
        self.concat_features_across_scales = concat_features_across_scales
        self.linear_decoder = linear_decoder
        self.linear_decoder_layers = linear_decoder_layers
        self.density_act = init_density_activation(density_activation)
        self.timer = CudaTimer(enabled=False)

        self.spatial_distortion: Optional[SpatialDistortion] = None
        if self.is_contracted:
            self.spatial_distortion = SceneContraction(
                order=float('inf'), global_scale=global_scale,
                global_translation=global_translation)

        self.field = KPlaneField(
            aabb,
            grid_config=self.config,
            concat_features_across_scales=self.concat_features_across_scales,
            multiscale_res=self.multiscale_res,
            use_appearance_embedding=use_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            spatial_distortion=self.spatial_distortion,
            density_activation=self.density_act,
            linear_decoder=self.linear_decoder,
            linear_decoder_layers=self.linear_decoder_layers,
            num_images=num_images,
        )

        self.occ_grid_reso = int(occ_grid_reso)
        self.use_occ_grid = self.occ_grid_reso > 0
        self.occ_grid = None
        self.occ_step_size = float(occ_step_size)
        self.occ_alpha_thres = float(occ_alpha_thres)
        # we use aabb nerfacc for this task. the resolution may differ
        if self.use_occ_grid > 0: 
            self.occupancy_grid = nerfacc.OccGridEstimator(
                roi_aabb=aabb.reshape(-1), resolution=self.occ_grid_reso, 
                levels=occ_level
            )

        # Initialize proposal-sampling nets
        self.density_fns = []
        self.num_proposal_iterations = num_proposal_iterations
        self.proposal_net_args_list = proposal_net_args_list
        self.proposal_warmup = proposal_warmup
        self.proposal_update_every = proposal_update_every
        self.use_proposal_weight_anneal = use_proposal_weight_anneal
        self.proposal_weights_anneal_max_num_iters = proposal_weights_anneal_max_num_iters
        self.proposal_weights_anneal_slope = proposal_weights_anneal_slope
        self.proposal_networks = torch.nn.ModuleList()
        if use_same_proposal_network:
            assert len(
                self.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.proposal_net_args_list[0]
            network = KPlaneDensityField(
                aabb, spatial_distortion=self.spatial_distortion,
                density_activation=self.density_act, linear_decoder=self.linear_decoder, **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend(
                [network.get_density for _ in range(self.num_proposal_iterations)])
        else:
            for i in range(self.num_proposal_iterations):
                prop_net_args = self.proposal_net_args_list[min(
                    i, len(self.proposal_net_args_list) - 1)]
                network = KPlaneDensityField(
                    aabb, spatial_distortion=self.spatial_distortion,
                    density_activation=self.density_act, linear_decoder=self.linear_decoder, **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend(
                [network.get_density for network in self.proposal_networks])

        def update_schedule(step): return np.clip(
            np.interp(step, [0, self.proposal_warmup],
                      [0, self.proposal_update_every]),
            1,
            self.proposal_update_every,
        )
        if self.is_contracted or self.is_ndc:
            initial_sampler = UniformLinDispPiecewiseSampler(
                single_jitter=single_jitter)
        else:
            initial_sampler = UniformSampler(single_jitter=single_jitter)
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=num_samples,
            num_proposal_samples_per_ray=num_proposal_samples,
            num_proposal_network_iterations=self.num_proposal_iterations,
            single_jitter=single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler
        )

    def step_before_iter(self, step):
        if self.use_occ_grid and self.training:
            def occ_eval_fn(x):
                requires_timestamps = len(self.field.grids[0]) == 6
                density = self.field.get_density(
                    x[:, None],
                    timestamps=torch.rand_like(x[:, 0]) * 2 - 1
                    if requires_timestamps
                    else None,
                )[0][:, 0]
                return density * self.occ_step_size

            self.occupancy_grid.update_every_n_steps(
                step=step, occ_eval_fn=occ_eval_fn, ema_decay=0.99
            )
        elif self.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.proposal_weights_anneal_max_num_iters
            # https://arxiv.org/pdf/2111.12077.pdf eq. 18, mipnerf360
            train_frac = np.clip(step / N, 0, 1)
            def bias(x, b): return (b * x) / ((b - 1) * x + 1)
            anneal = bias(train_frac, self.proposal_weights_anneal_slope)
            self.proposal_sampler.set_anneal(anneal)

    def step_after_iter(self, step):
        if not self.use_occ_grid and self.use_proposal_weight_anneal:
            self.proposal_sampler.step_cb(step)

    @staticmethod
    def render_rgb(rgb: torch.Tensor, weights: torch.Tensor, bg_color: Optional[torch.Tensor]):
        comp_rgb = torch.sum(weights * rgb, dim=-2)
        accumulated_weight = torch.sum(weights, dim=-2)
        if bg_color is None:
            pass
        else:
            comp_rgb = comp_rgb + (1.0 - accumulated_weight) * bg_color
        return comp_rgb

    @staticmethod
    def render_depth(weights: torch.Tensor, ray_samples: RaySamples, rays_d: torch.Tensor):
        steps = (ray_samples.starts + ray_samples.ends) / 2
        one_minus_transmittance = torch.sum(weights, dim=-2)
        depth = torch.sum(weights * steps, dim=-2) + \
            one_minus_transmittance * rays_d[..., -1:]
        return depth

    @staticmethod
    def render_accumulation(weights: torch.Tensor):
        accumulation = torch.sum(weights, dim=-2)
        return accumulation

    def forward(self, rays_o, rays_d, bg_color, near_far: torch.Tensor, timestamps=None):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        timestamps : [batch]
        near_far : [batch, 2]
        """
        # Fix shape for near-far
        nears, fars = torch.split(near_far, [1, 1], dim=-1)
        if nears.shape[0] != rays_o.shape[0]:
            ones = torch.ones_like(rays_o[..., 0:1])
            nears = ones * nears
            fars = ones * fars

        rgb = accumulation = depth = None
        if self.use_occ_grid:

            # def sigma_fn(t_starts, t_ends, ray_indices):
            #     t_origins = rays_o[ray_indices]
            #     if t_origins.shape[0] == 0:
            #         return torch.zeros((0,), device=t_origins.device)
            #     t_dirs = rays_d[ray_indices]
            #     t_times = (
            #         timestamps[ray_indices] if timestamps is not None else None
            #     )
            #     positions = t_origins + t_dirs * \
            #         (t_starts + t_ends)[:, None] / 2.0
            #     return self.field.get_density(positions[:, None], t_times)[0][
            #         :, 0
            #     ].squeeze(-1)

            # def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            #     t_origins = rays_o[ray_indices]
            #     if t_origins.shape[0] == 0:
            #         return torch.zeros(
            #             (0, 3), device=t_origins.device
            #         ), torch.zeros((0,), device=t_origins.device)
            #     t_dirs = rays_d[ray_indices]
            #     t_times = (
            #         timestamps[ray_indices] if timestamps is not None else None
            #     )
            #     positions = t_origins + t_dirs * \
            #         (t_starts + t_ends)[:, None] / 2.0
            #     field_out = self.field(
            #         positions[:, None], t_dirs[:, None], t_times
            #     )
            #     return field_out["rgb"][:, 0], field_out["density"][:, 0].squeeze(-1)
            
            def sigma_fn(t_starts, t_ends, ray_indices):
                t_origins = rays_o[ray_indices]
                if t_origins.shape[0] == 0:
                    return torch.zeros((0,), device=t_origins.device)
                t_dirs = rays_d[ray_indices]
                t_times = (
                    timestamps[ray_indices] if timestamps is not None else None
                )
                positions = t_origins + t_dirs * \
                    (t_starts + t_ends)[:, None] / 2.0
                return self.field.get_density(positions[:, None], t_times)[0][
                    :, 0
                ].squeeze(-1)

            def rgb_sigma_fn(t_starts, t_ends, ray_indices):
                t_origins = rays_o[ray_indices]
                if t_origins.shape[0] == 0:
                    return torch.zeros(
                        (0, 3), device=t_origins.device
                    ), torch.zeros((0,), device=t_origins.device)
                t_dirs = rays_d[ray_indices]
                t_times = (
                    timestamps[ray_indices] if timestamps is not None else None
                )
                positions = t_origins + t_dirs * \
                    (t_starts + t_ends)[:, None] / 2.0
                field_out = self.field(
                    positions[:, None], t_dirs[:, None], t_times
                )
                return field_out["rgb"][:, 0], field_out["density"][:, 0].squeeze(-1)
            
            ray_indices, t_starts, t_ends = self.occupancy_grid.sampling(
                rays_o,
                rays_d,
                sigma_fn=sigma_fn,
                near_plane=nears[0, 0],
                far_plane=fars[0, 0],
                render_step_size=self.occ_step_size,
                stratified=self.training,
                alpha_thre=self.occ_alpha_thres,
                early_stop_eps=1e-4,
            )
            rgb, accumulation, depth, _ = nerfacc.rendering(
                t_starts,
                t_ends,
                ray_indices,
                n_rays=rays_o.shape[0],
                rgb_sigma_fn=rgb_sigma_fn,
                render_bkgd=bg_color[0] if bg_color is not None else None,
            )
        else:
            ray_bundle = RayBundle(
                origins=rays_o, directions=rays_d, nears=nears, fars=fars
            )
            # Note: proposal sampler mustn't use timestamps (=camera-IDs) with appearance-embedding,
            #       since the appearance embedding should not affect density. We still pass them in the
            #       call below, but they will not be used as long as density-field resolutions
            #       are be 3D.
            (
                ray_samples,
                weights_list,
                ray_samples_list,
            ) = self.proposal_sampler.generate_ray_samples(
                ray_bundle, timestamps=timestamps, density_fns=self.density_fns
            )

            field_out = self.field(
                ray_samples.get_positions(), ray_bundle.directions, timestamps
            )
            rgb, density = field_out["rgb"], field_out["density"]

            weights = ray_samples.get_weights(density)
            weights_list.append(weights)
            ray_samples_list.append(ray_samples)

            rgb = self.render_rgb(rgb=rgb, weights=weights, bg_color=bg_color)
            depth = self.render_depth(
                weights=weights,
                ray_samples=ray_samples,
                rays_d=ray_bundle.directions,
            )
            accumulation = self.render_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if not self.use_occ_grid:
            # These use a lot of GPU memory, so we avoid storing them for eval.
            if self.training:
                outputs["weights_list"] = weights_list  # type: ignore
                outputs["ray_samples_list"] = ray_samples_list  # type: ignore
            for i in range(self.num_proposal_iterations):
                outputs[f"prop_depth_{i}"] = self.render_depth(
                    weights=weights_list[i],  # type: ignore
                    ray_samples=ray_samples_list[i],  # type: ignore
                    rays_d=ray_bundle.directions,  # type: ignore
                )
        return outputs

    def get_params(self, lr: float):
        model_params = self.field.get_params()
        pn_params = [pn.get_params() for pn in self.proposal_networks]
        field_params = model_params["field"] + \
            [p for pnp in pn_params for p in pnp["field"]]
        nn_params = model_params["nn"] + \
            [p for pnp in pn_params for p in pnp["nn"]]
        other_params = model_params["other"] + \
            [p for pnp in pn_params for p in pnp["other"]]
        return [
            {"params": field_params, "lr": lr},
            {"params": nn_params, "lr": lr},
            {"params": other_params, "lr": lr},
        ]
