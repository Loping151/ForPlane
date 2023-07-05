config = {
    'description': 'hamlyn debug',

    'expname': 'endo',
    'logdir': './logs/debug',
    'device': 'cuda:0',

    'data_downsample': 1.0,
    'data_dirs': ['data/endonerf_full_datasets/pulling_soft_tissues'],
    'contract': False,
    'ndc': True,
    'ndc_far': 2.0,
    'isg': True,
    'isg_step': 0,
    'ist_step': 55,
    'keyframes': False,
    'scene_bbox': [[-1.0, -1.0, -1.0], [1.0, 1.0, 0.1]],
    'maskIS': True,
    'frequency_ratio': 1,
    'near_scaling': 0.95,
    'bg_color': 0,
    'depth_type': 'mono_depth',
    # Optimization settings
    'num_steps': 550,
    'batch_size': 32768,
    'scheduler_type': 'warmup_cosine',
    'optim_type': 'adam',
    'lr': 0.01,
    "eval_batch_size": 65536,

    # Regularization
    # 'distortion_loss_weight': 0.001, [yc: 2.20 remove dist loss for better scene recon]
    'distortion_loss_weight': 0.0,
    'histogram_loss_weight': 1.0,
    'mono_depth_weight': 0.1, 
    'mono_depth_weight_proposal_net': 0.1, 
    'l1_time_planes': 0.0001,
    'l1_time_planes_proposal_net': 0.0001,
    'plane_tv_weight': 0.0001,
    'plane_tv_weight_proposal_net': 0.0001,
    'time_smoothness_weight': 0.03,
    'time_smoothness_weight_proposal_net': 0.0001,
    'depth_huber_weight': 0, # 0.05
    'depth_huber_weight_proposal_net': 0, # 0.05
    'step_iter': 225,

    # Training settings
    'valid_every': 550,
    'save_every': 550,
    'save_outputs': True,
    'train_fp16': True,

    # Raymarching settings, used for assist sampling
    'single_jitter': False,
    'num_samples': 64,
    'num_proposal_iterations': 2,
    'num_proposal_samples': [256, 128],
    'use_same_proposal_network': True,
    'use_proposal_weight_anneal': True,
    'proposal_net_args_list': [
        {'num_input_coords': 4, 'num_output_coords': 8,
            'resolution': [128, 128, 128, 300]}
        #     ,
        # {'num_input_coords': 4, 'num_output_coords': 8,
        #     'resolution': [256, 256, 256, 156]}
    ],
    #  'max_train_tsteps': 100000,

    # Model settings, main model settings
    'concat_features_across_scales': True,
    'density_activation': 'trunc_exp',
    'linear_decoder': False,
    'multiscale_res': [1, 2, 4, 8],
    'grid_config': [
        {
            'grid_dimensions': 2,
            'input_coordinate_dim': 4,
            'output_coordinate_dim': 16,
            'disable_view_encoder': True,
            'resolution': [64, 64, 64, 300]
        },
        {
            'encoder_type': 'OneBlob',
            'encode_items': 'xyzt',
            'n_bins': 4
        }
    ],
}

# exit()
# PYTHONPATH=. python lerplanes/main.py --config-path /home/yangchen/projects/kplanes-endo/lerplanes/configs/hamlyn/debug_hamlyn.py
