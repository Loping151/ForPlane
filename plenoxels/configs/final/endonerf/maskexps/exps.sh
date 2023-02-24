# cutting_tissues_twice

PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_direct_mask.py
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS.py --expname endo_hybrid_maskIS_1_100 --isg_step 1 --ist_step 100
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS.py --expname endo_hybrid_maskIS_1_300 --isg_step 1 --ist_step 300
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS.py --expname endo_hybrid_maskIS_1_500 --isg_step 1 --ist_step 500
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS.py --expname endo_hybrid_maskIS_1_1000 --isg_step 1 --ist_step 1000
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS.py --expname endo_hybrid_maskIS_1_1500 --isg_step 1 --ist_step 1500
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS.py --expname endo_hybrid_maskIS_1_2000 --isg_step 1 --ist_step 2000

PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_direct_mask.py --data_dirs ['data/endonerf_full_datasets/pulling_soft_tissues']
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS.py --expname endo_hybrid_maskIS_1_100 --isg_step 1 --ist_step 100 --data_dirs ['data/endonerf_full_datasets/pulling_soft_tissues']
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS.py --expname endo_hybrid_maskIS_1_300 --isg_step 1 --ist_step 300 --data_dirs ['data/endonerf_full_datasets/pulling_soft_tissues']
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS.py --expname endo_hybrid_maskIS_1_500 --isg_step 1 --ist_step 500 --data_dirs ['data/endonerf_full_datasets/pulling_soft_tissues']
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS.py --expname endo_hybrid_maskIS_1_1000 --isg_step 1 --ist_step 1000 --data_dirs ['data/endonerf_full_datasets/pulling_soft_tissues']
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS.py --expname endo_hybrid_maskIS_1_1500 --isg_step 1 --ist_step 1500 --data_dirs ['data/endonerf_full_datasets/pulling_soft_tissues']
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS.py --expname endo_hybrid_maskIS_1_2000 --isg_step 1 --ist_step 2000 --data_dirs ['data/endonerf_full_datasets/pulling_soft_tissues']