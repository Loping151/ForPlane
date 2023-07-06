export CUDA_VISIBLE_DEVICES=2

# train the endonerf dataset
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/endonerf/paper/iter9k/acc_cutt.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/endonerf/paper/iter9k/acc_pull.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/endonerf/paper/iter9k/acc_push.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/endonerf/paper/iter9k/acc_tear.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/endonerf/paper/iter9k/acc_thin.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/endonerf/paper/iter9k/acc_trac.py

