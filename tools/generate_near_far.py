# %%
import os
import imageio.v2 as imageio
import numpy as np

# %%
root_path = "data/endonerf_full_datasets"
dataset_names = [os.path.join(root_path, d) for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]

# %%
# we load all depth images to get the depth distribution and then we calculate the depth range
all_depth = {}
for dataset in dataset_names:
    depths = []
    for dp in os.listdir(os.path.join(dataset, "depth")):
        depth = imageio.imread(os.path.join(dataset, "depth", dp))
        depths.append(depth)
    all_depth[dataset] = np.stack(depths)

# %%
def get_close_depth_inf_depth(depths):
    close_depth, inf_depth = np.percentile(depths, 3.0), np.percentile(depths, 99.9)
    return close_depth, inf_depth

# %%
for k in all_depth.keys():
    close_depth, inf_depth = get_close_depth_inf_depth(all_depth[k])
    print(k, close_depth, inf_depth)
    # we save the close depth and inf depth to the dataset folder, using np.savetxt
    np.savetxt(os.path.join(k, "close_inf_depth.txt"), np.array([close_depth, inf_depth]))

# %%
close_depth, inf_depth = np.loadtxt("data/endonerf_full_datasets/cutting_tissues_twice/close_inf_depth.txt")

# %%
close_depth, inf_depth

# %%
# 


