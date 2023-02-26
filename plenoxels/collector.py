import os
import csv
import argparse

parser = argparse.ArgumentParser(description='Choose dataset name.')
parser.add_argument('--index', type=str, help='from 0 to 5', default=0)

args = parser.parse_args()

filename = "endo_log.txt"
datalist = ['cutting', 'pulling', 'pushing', 'tearing', 'thin', 'traction']

if args.index is not None and args.index < len(datalist):
    dataname = datalist[args.index]
else:
    print('Invalid index')
    exit()

if os.path.exists('logs'):
    directory = "logs"
elif os.path.exists('../logs'):
    directory = "../logs"


file_list = []
for root, dirs, files in os.walk(directory):
    for file in files:
        if file == filename:
            file_list.append(os.path.join(root, file))

if not os.path.exists(os.path.join(directory, 'performance')):
    os.mkdir(os.path.join(directory, 'performance'))

with open(os.path.join(directory, 'performance','all_results.csv'), mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['dataset:'+dataname])
    writer.writerow(['expname', 'isg', 'freq_ratio', 'ist_step', 'PSNR', 'SSIM', 'LPIPS'])
    for file_path in file_list:
        if dataname not in file_path:
            continue
        with open(os.path.join(os.path.dirname(file_path), 'config.csv'), 'r') as config, open(file_path, 'r') as endolog:
            config = csv.reader(config, delimiter = '\t')
            config = {rows[0]: rows[1] for rows in config}
            ratio = config.get('frequency_ratio')
            isg = config.get('isg')
            ist_step = config.get('ist_step')
            mertric = []
            for line in endolog:
                key, value = line.strip().split(':')
                mertric.append(value)

            writer.writerow([os.path.basename(os.path.dirname(file_path)), isg, ratio, ist_step, *mertric])
