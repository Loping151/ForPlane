export CUDA_VISIBLE_DEVICES=0

# train the endonerf dataset
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/hamlyn/iter9k/hamlyn1.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/hamlyn/iter9k/hamlyn2.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/hamlyn/iter9k/hamlyn3.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/hamlyn/iter9k/hamlyn4.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/hamlyn/iter9k/hamlyn5.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/hamlyn/iter9k/hamlyn6.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/hamlyn/iter9k/hamlyn7.py
