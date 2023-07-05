export CUDA_VISIBLE_DEVICES=0

# train the endonerf dataset
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/hamlyn/iter32k/hamlyn1.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/hamlyn/iter32k/hamlyn2.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/hamlyn/iter32k/hamlyn3.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/hamlyn/iter32k/hamlyn4.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/hamlyn/iter32k/hamlyn5.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/hamlyn/iter32k/hamlyn6.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/hamlyn/iter32k/hamlyn7.py
