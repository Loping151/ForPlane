conda activate jaxnerf
# export LD_LIBRARY_PATH=/home/yangchen/.conda/envs/ngp/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=3

# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/enable_view.py
# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/no_mask.py
# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/current_2000.py
# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/current_1500.py
# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/compare_endo_pushing.py
# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/compare_endo_pulling.py
# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/compare_endo_cutting.py
# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/compare_endo_thin.py
# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/compare_endo_tearing.py
# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/compare_endo_traction.py
# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/encode_xy_freq_8.py
# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/encode_xy_blob_8.py
# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/encode_xy_freq_12.py
# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/encode_xyz_freq_12.py
# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/encode_xyz_blob_12.py
# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/encode_xyt_freq_12.py
# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/encode_xyt_blob_12.py
# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/encode_xyzt_freq_8.py
# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/encode_xyzt_blob_8.py
# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/encode_xyzt_freq_16.py
# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/encode_xyzt_blob_16.py
# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/no_isg.py
# PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/step_iter10.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/step_iter20.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/step_iter30.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/step_iter40.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/bicubic.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/nearest.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/loss_trial1.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/loss_trial2.py

python plenoxels/collector.py --path logs/finals