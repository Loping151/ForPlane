conda activate ngp
export LD_LIBRARY_PATH=/home/yangchen/.conda/envs/ngp/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/enable_view.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/current_2000.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/current_1500.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/compare_endo.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/encode_xy_freq_8.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/encode_xy_freq_12.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/encode_xyz_freq_12.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/encode_xyt_freq_12.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/encode_xyzt_freq_8.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/encode_xyzt_freq_16.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/final_exps/no_isg.py
