export CUDA_VISIBLE_DEVICES=0

PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/enable_view_pushing.py
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/enable_view_pulling.py
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/enable_view_tr.py
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/enable_view_te.py
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/enable_view_th.py

# PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/naive_mask_th.py
# PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/naive_mask_pushing.py
# PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/naive_mask_pulling.py
# PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/naive_mask_tr.py
# PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/naive_mask_te.py

# PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/no_isg_tr.py
# PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/no_isg_th.py
# PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/no_isg_te.py
# PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/no_isg_pushing.py
# PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/no_isg_pulling.py

# PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/no_warm_up_tr.py
# PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/no_warm_up_th.py
# PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/no_warm_up_te.py
# PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/no_warm_up_pushing.py
# PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/configs/final/endonerf/paper_exps/no_warm_up_pulling.py

