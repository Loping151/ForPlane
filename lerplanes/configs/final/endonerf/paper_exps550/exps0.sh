export CUDA_VISIBLE_DEVICES=0

# PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/full_175.py --render-only
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/full_550.py --render-only
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/full_1500.py --render-only
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/full_2000.py --render-only


PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/full_1500_cutting.py --render-only
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/full_1500_pulling.py --render-only
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/full_1500_pushing.py --render-only
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/full_1500_tearing.py --render-only
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/full_1500_thin.py --render-only
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/full_1500_traction.py --render-only

PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/no_ist.py --render-only
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/naive_mask.py --render-only
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/enable_view.py --render-only
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/no_warm_up.py --render-only