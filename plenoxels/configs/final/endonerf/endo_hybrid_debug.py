config = {
 'expname': 'debug_endo',
 'logdir': './logs/endo_example_x',
 'device': 'cuda:0',

 'data_downsample': 1.0,
    'data_dirs': ['data/endonerf_full_datasets/pulling_soft_tissues'], 'contract': False,
 'ndc': True,
 'ndc_far': 2.6,
 'isg': False,
 'isg_step': -1,
 'ist_step': 50000,
 'keyframes': False,
 'scene_bbox': [[-3.0, -1.8, -1.2], [3.0, 1.8, 1.2]],
 'keyframes': False,
 'endo': True,
 'near_scaling': 0.95,

 # Optimization settings
 'num_steps': 10000,
 'batch_size': 16384,
 'scheduler_type': 'warmup_cosine',
 'optim_type': 'adam',
 'lr': 0.01,

 # Regularization
 'distortion_loss_weight': 0.001,
 'histogram_loss_weight': 1.0,
 'l1_time_planes': 0.0001,
 'l1_time_planes_proposal_net': 0.0001,
 'plane_tv_weight': 0.0001,
 'plane_tv_weight_proposal_net': 0.0001,
 'time_smoothness_weight': 0.01,
 'time_smoothness_weight_proposal_net': 0.0001,

 # Training settings
 'valid_every': 1000,
 'save_every': 1000,
 'save_outputs': True,
 'train_fp16': True,

 # Raymarching settings
 'single_jitter': False,
 'num_samples': 48,
 'num_proposal_iterations': 2,
 'num_proposal_samples': [256, 128],
 'use_same_proposal_network': False,
 'use_proposal_weight_anneal': True,
 'proposal_net_args_list': [
  {'num_input_coords': 4, 'num_output_coords': 8, 'resolution': [128, 128, 128, 150]},
  {'num_input_coords': 4, 'num_output_coords': 8, 'resolution': [256, 256, 256, 150]}
 ],
#  'max_train_tsteps': 100000, 

 # Model settings
 'concat_features_across_scales': True,
 'density_activation': 'trunc_exp',
 'linear_decoder': False,
 'multiscale_res': [1, 2, 4, 8],
 'grid_config': [{
  'grid_dimensions': 2,
  'input_coordinate_dim': 4,
  'output_coordinate_dim': 16,
  'resolution': [64, 64, 64, 78]
 }],
}
