PYTHONPATH=. python lerplanes/main.py --config-path /home/yangchen/projects/kplanes-endo/lerplanes/configs/acc/acc_cutt.py 'expname'='0000' 'occ_grid_reso'='64' 'occ_step_size'='4e-3' 'occ_level'='1' 'occ_alpha_thres'='1e-2'

PYTHONPATH=. python lerplanes/main.py --config-path /home/yangchen/projects/kplanes-endo/lerplanes/configs/std/std_push.py

PYTHONPATH=. python lerplanes/main.py --config-path /home/yangchen/projects/kplanes-endo/lerplanes/configs/std/std_pull.py

PYTHONPATH=. python lerplanes/main.py --config-path /home/yangchen/projects/kplanes-endo/lerplanes/configs/std/std_tear.py

PYTHONPATH=. python lerplanes/main.py --config-path /home/yangchen/projects/kplanes-endo/lerplanes/configs/std/std_thin.py

PYTHONPATH=. python lerplanes/main.py --config-path /home/yangchen/projects/kplanes-endo/lerplanes/configs/std/std_trac.py

export CUDA_VISIBLE_DEVICES=0