BATCH_SIZE=12500

# endo's ray sampling baseline
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_direct_mask_cutting.py 'logdir'='./logs/endo_hybrid_direct_mask/cutting' 'batch_size'=$BATCH_SIZE
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_direct_mask_pulling.py 'logdir'='./logs/endo_hybrid_direct_mask/pulling' 'batch_size'=$BATCH_SIZE
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_direct_mask_pushing.py 'logdir'='./logs/endo_hybrid_direct_mask/pushing' 'batch_size'=$BATCH_SIZE
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_direct_mask_tearing.py 'logdir'='./logs/endo_hybrid_direct_mask/tearing' 'batch_size'=$BATCH_SIZE
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_direct_mask_thin.py 'logdir'='./logs/endo_hybrid_direct_mask/thin' 'batch_size'=$BATCH_SIZE
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_direct_mask_traction.py 'logdir'='./logs/endo_hybrid_direct_mask/traction' 'batch_size'=$BATCH_SIZE

# cutting_tissues_twice

PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS_cutting.py 'expname'='endo_hybrid_maskIS_1_100' 'ist_step'=100 'logdir'='./logs/endo_hybrid_maskIS/cutting' 'batch_size'=$BATCH_SIZE
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS_cutting.py 'expname'='endo_hybrid_maskIS_1_300' 'ist_step'=300 'logdir'='./logs/endo_hybrid_maskIS/cutting' 'batch_size'=$BATCH_SIZE
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS_cutting.py 'expname'='endo_hybrid_maskIS_1_500' 'ist_step'=500 'logdir'='./logs/endo_hybrid_maskIS/cutting' 'frequency_ratio'=1 'batch_size'=$BATCH_SIZE
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS_cutting.py 'expname'='endo_hybrid_maskIS_1_500_10' 'ist_step'=500 'logdir'='./logs/endo_hybrid_maskIS/cutting' 'frequency_ratio'=10 'batch_size'=$BATCH_SIZE
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS_cutting.py 'expname'='endo_hybrid_maskIS_1_500_50' 'ist_step'=500 'logdir'='./logs/endo_hybrid_maskIS/cutting' 'frequency_ratio'=50 'batch_size'=$BATCH_SIZE
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS_cutting.py 'expname'='endo_hybrid_maskIS_1_500_100' 'ist_step'=500 'logdir'='./logs/endo_hybrid_maskIS/cutting' 'frequency_ratio'=100 'batch_size'=$BATCH_SIZE
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS_cutting.py 'expname'='endo_hybrid_maskIS_1_500_1000' 'ist_step'=500 'logdir'='./logs/endo_hybrid_maskIS/cutting' 'frequency_ratio'=1000 'batch_size'=$BATCH_SIZE
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS_cutting.py 'expname'='endo_hybrid_maskIS_1_1000' 'ist_step'=1000 'logdir'='./logs/endo_hybrid_maskIS/cutting' 'batch_size'=$BATCH_SIZE
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS_cutting.py 'expname'='endo_hybrid_maskIS_1_1500' 'ist_step'=1500 'logdir'='./logs/endo_hybrid_maskIS/cutting' 'batch_size'=$BATCH_SIZE
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS_cutting.py 'expname'='endo_hybrid_maskIS_1_2000' 'ist_step'=2000 'logdir'='./logs/endo_hybrid_maskIS/cutting' 'batch_size'=$BATCH_SIZE

# pulling_soft_tissues

PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS_pulling.py 'expname'='endo_hybrid_maskIS_1_100' 'ist_step'=100 'logdir'='./logs/endo_hybrid_maskIS/pulling' 'batch_size'=$BATCH_SIZE
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS_pulling.py 'expname'='endo_hybrid_maskIS_1_300' 'ist_step'=300 'logdir'='./logs/endo_hybrid_maskIS/pulling' 'batch_size'=$BATCH_SIZE
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS_pulling.py 'expname'='endo_hybrid_maskIS_1_500' 'ist_step'=500 'logdir'='./logs/endo_hybrid_maskIS/pulling' 'batch_size'=$BATCH_SIZE
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS_pulling.py 'expname'='endo_hybrid_maskIS_1_1000' 'ist_step'=1000 'logdir'='./logs/endo_hybrid_maskIS/pulling' 'batch_size'=$BATCH_SIZE
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS_pulling.py 'expname'='endo_hybrid_maskIS_1_1500' 'ist_step'=1500 'logdir'='./logs/endo_hybrid_maskIS/pulling' 'batch_size'=$BATCH_SIZE
PYTHONPATH='.' python plenoxels/main.py --config-path plenoxels/configs/final/endonerf/maskexps/endo_hybrid_maskIS_pulling.py 'expname'='endo_hybrid_maskIS_1_2000' 'ist_step'=2000 'logdir'='./logs/endo_hybrid_maskIS/pulling' 'batch_size'=$BATCH_SIZE

