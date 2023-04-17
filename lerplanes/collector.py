import os
import csv
import argparse
import glob

parser = argparse.ArgumentParser(description='Choose dataset name.')
parser.add_argument('--index', type=int, help='from 0 to 5', default=0)
parser.add_argument('--dataname', type=str, help='data name', default='')
parser.add_argument('--path', type=str, help='path dir', default='')

args = parser.parse_args()

filename = "endo_log.txt"
datalist = ['cutting', 'pulling', 'pushing', 'tearing', 'thin', 'traction']

if len(args.dataname) > 0:
    dataname = args.dataname
else:
    dataname = ''
<<<<<<< HEAD
# elif args.index is not None and args.index < len(datalist):
#     dataname = datalist[args.index]
# else:
#     print('Invalid index')
#     exit()
=======
>>>>>>> b26eda0cef18828bb6d35a349459deb84f752fbb

if args.path == '':
    if os.path.exists('logs'):
        directory = "logs"
    elif os.path.exists('../logs'):
        directory = "../logs"
else:
    directory = args.path

file_list = []
for root, dirs, files in os.walk(directory):
    for file in files:
        if file == filename:
            file_list.append(os.path.join(root, file))

if not os.path.exists(os.path.join(directory, 'performance')):
    os.mkdir(os.path.join(directory, 'performance'))

with open(os.path.join(directory, 'performance', '2000all_results.csv'), mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['path:'+directory])
    writer.writerow(['dataset:'+dataname])
    label_items = ['expname', 'num_steps', 'step_iter', 'maskIS', 'isg', 'isg_step', 'bg_color', 
                   'ist_step', 'grid_config', 'description', 'PSNR', 'SSIM', 'LPIPS', 'FLIP']
    writer.writerow(label_items)
    for file_path in file_list:
        if dataname not in file_path:
            continue
        with open(os.path.join(os.path.dirname(file_path), 'config.csv'), 'r') as config, open(file_path, 'r') as endolog, open(glob.glob(os.path.join(os.path.dirname(file_path), 'test*.csv'))[0]) as FLIPS:
            config = csv.reader(config, delimiter='\t')
            config = {rows[0]: rows[1] for rows in config}
<<<<<<< HEAD
            # ratio = config.get('frequency_ratio')
            # isg = config.get('isg')
            # ist_step = config.get('ist_step')
=======
>>>>>>> b26eda0cef18828bb6d35a349459deb84f752fbb
            mertric = []
            for line in endolog:
                key, value = line.strip().split(':')
                mertric.append(value)
            flip = 0
            csvreader = csv.reader(FLIPS, delimiter=',')
            next(csvreader)  # 跳过第一行
            second_row = next(csvreader)  # 获取第二行
            flips = float(second_row[-1])  # 获取最后一个浮点数

            writer.writerow([os.path.basename(os.path.dirname(
                file_path)), *[config.get(i) for i in label_items[1:-4]], *mertric, flips])
