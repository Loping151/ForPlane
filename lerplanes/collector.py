# import os
# import csv
# import argparse
# import glob

# parser = argparse.ArgumentParser(description='Choose dataset name.')
# parser.add_argument('--index', type=int, help='from 0 to 5', default=0)
# parser.add_argument('--dataname', type=str, help='data name', default='')
# parser.add_argument('--path', type=str, help='path dir', default='')

# args = parser.parse_args()

# filename = "endo_log.txt"
# datalist = ['cutting', 'pulling', 'pushing', 'tearing', 'thin', 'traction']

# if len(args.dataname) > 0:
#     dataname = args.dataname
# else:
#     dataname = ''

# if args.path == '':
#     if os.path.exists('logs'):
#         directory = "logs"
#     elif os.path.exists('../logs'):
#         directory = "../logs"
# else:
#     directory = args.path

# file_list = []
# for root, dirs, files in os.walk(directory):
#     for file in files:
#         if file == filename:
#             file_list.append(os.path.join(root, file))

# if not os.path.exists(os.path.join(directory, 'performance')):
#     os.mkdir(os.path.join(directory, 'performance'))

# with open(os.path.join(directory, 'performance', '2000all_results.csv'), mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['path:'+directory])
#     writer.writerow(['dataset:'+dataname])
#     label_items = ['expname', 'num_steps', 'step_iter', 'maskIS', 'isg', 'isg_step', 'bg_color', 
#                    'ist_step', 'grid_config', 'description', 'PSNR', 'SSIM', 'LPIPS', 'FLIP']
#     writer.writerow(label_items)
#     for file_path in file_list:
#         if dataname not in file_path:
#             continue
#         with open(os.path.join(os.path.dirname(file_path), 'config.csv'), 'r') as config, open(file_path, 'r') as endolog, open(glob.glob(os.path.join(os.path.dirname(file_path), 'test*.csv'))[0]) as FLIPS:
#             config = csv.reader(config, delimiter='\t')
#             config = {rows[0]: rows[1] for rows in config}
#             mertric = []
#             for line in endolog:
#                 key, value = line.strip().split(':')
#                 mertric.append(value)
#             flip = 0
#             csvreader = csv.reader(FLIPS, delimiter=',')
#             next(csvreader)  # 跳过第一行
#             second_row = next(csvreader)  # 获取第二行
#             flips = float(second_row[-1])  # 获取最后一个浮点数

#             writer.writerow([os.path.basename(os.path.dirname(
#                 file_path)), *[config.get(i) for i in label_items[1:-4]], *mertric, flips])

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