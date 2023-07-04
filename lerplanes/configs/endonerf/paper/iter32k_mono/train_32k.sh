export CUDA_VISIBLE_DEVICES=0

# train the endonerf dataset
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/endonerf/paper/iter32k/acc_cutt.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/endonerf/paper/iter32k/acc_pull.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/endonerf/paper/iter32k/acc_push.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/endonerf/paper/iter32k/acc_tear.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/endonerf/paper/iter32k/acc_thin.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/endonerf/paper/iter32k/acc_trac.py

