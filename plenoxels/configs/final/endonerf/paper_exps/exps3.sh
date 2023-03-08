export CUDA_VISIBLE_DEVICES=0

PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/paper_exps/full_550_cutting.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/paper_exps/full_550_pulling.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/paper_exps/full_550_pushing.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/paper_exps/full_550_tearing.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/paper_exps/full_550_thin.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/paper_exps/full_550_traction.py