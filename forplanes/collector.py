### not used
import os
import csv
import argparse
import glob

parser = argparse.ArgumentParser(description='Choose dataset name.')
parser.add_argument('--directory', type=str, help='path dir', default='logs/acc')
args = parser.parse_args()
directory = args.directory

file_list = []
for root, dirs, files in os.walk(directory):
    for file in files:
        if 'endo_log.txt' in file:
            file_list.append(os.path.join(root, file))

param_list = [[64, 128, 256],
              [4e-3, 0, 0, 1e-2], 
              [1, 2], 
              [1e-2, 1e-3, 1e-4 , 0]]

with open(os.path.join(directory, 'performance.txt'), mode='w') as file:
    for file_path in file_list:
        with open(file_path, 'r') as mertric:
            param_label = file_path[-17:-13]
            file.write('occ_grid_reso = '+str(param_list[0][int(param_label[0])])+'\tocc_step_size = '+str(param_list[1][int(param_label[1])])+'\tocc_level = '+str(param_list[2][int(param_label[2])])+'\tocc_alpha_thres = '+str(param_list[3][int(param_label[3])])+'\t\t'+mertric.read()+'\n')