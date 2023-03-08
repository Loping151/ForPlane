export CUDA_VISIBLE_DEVICES=0

PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/paper_exps/full_2000_cutting.py --render-only
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/paper_exps/full_2000_pulling.py --render-only
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/paper_exps/full_2000_pushing.py --render-only
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/paper_exps/full_2000_tearing.py --render-only
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/paper_exps/full_2000_thin.py --render-only
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/paper_exps/full_2000_traction.py --render-only