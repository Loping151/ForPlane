conda activate lpl

PYTHONPATH=. python lerplanes/main.py --config-path /home/yangchen/projects/kplanes-endo/lerplanes/configs/std/std_cutt.py

PYTHONPATH=. python lerplanes/main.py --config-path /home/yangchen/projects/kplanes-endo/lerplanes/configs/std/std_push.py

PYTHONPATH=. python lerplanes/main.py --config-path /home/yangchen/projects/kplanes-endo/lerplanes/configs/std/std_pull.py

PYTHONPATH=. python lerplanes/main.py --config-path /home/yangchen/projects/kplanes-endo/lerplanes/configs/std/std_tear.py

PYTHONPATH=. python lerplanes/main.py --config-path /home/yangchen/projects/kplanes-endo/lerplanes/configs/std/std_thin.py

PYTHONPATH=. python lerplanes/main.py --config-path /home/yangchen/projects/kplanes-endo/lerplanes/configs/std/std_trac.py

export CUDA_VISIBLE_DEVICES=3

PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/endonerf/paper/iter32k/acc_push.py


PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/std/hamlyn/lerplane_32k_hamlyn1.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/std/hamlyn/lerplane_32k_hamlyn2.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/std/hamlyn/lerplane_32k_hamlyn3.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/std/hamlyn/lerplane_32k_hamlyn4.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/std/hamlyn/lerplane_32k_hamlyn5.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/std/hamlyn/lerplane_32k_hamlyn6.py
PYTHONPATH=. python lerplanes/main.py --config-path lerplanes/configs/std/hamlyn/lerplane_32k_hamlyn7.py