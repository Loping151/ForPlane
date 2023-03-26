export CUDA_VISIBLE_DEVICES=1

PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/full_2000_cutting.py
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/full_2000_pulling.py
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/full_2000_pushing.py
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/full_2000_tearing.py
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/full_2000_thin.py
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/full_2000_traction.py
