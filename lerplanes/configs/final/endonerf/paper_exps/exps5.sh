export CUDA_VISIBLE_DEVICES=2

PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/full_5000_cutting.py --render-only
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/full_5000_pulling.py --render-only
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/full_5000_pushing.py --render-only
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/full_5000_tearing.py --render-only
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/full_5000_thin.py --render-only
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/full_5000_traction.py --render-only