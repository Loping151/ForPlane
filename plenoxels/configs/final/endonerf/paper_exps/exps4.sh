export CUDA_VISIBLE_DEVICES=1

PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/paper_exps/full_2000_cutting.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/paper_exps/full_2000_pulling.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/paper_exps/full_2000_pushing.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/paper_exps/full_2000_tearing.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/paper_exps/full_2000_thin.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/paper_exps/full_2000_traction.py
