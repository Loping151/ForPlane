import os
import csv
import argparse
import glob
import torch.utils.data
import torch
import argparse
import importlib.util
import logging
import os
import pprint
import sys
from typing import List, Dict, Any
import tempfile
import numpy as np
import imageio.v2 as imageio
from plenoxels.ops.image import metrics
import lpips

@staticmethod
def common_sort(name_list):
    valid_extensions = ['.png', '.jpg', '.jpeg']
    image_names = [name for name in name_list if any(name.lower().endswith(ext) for ext in valid_extensions)]
    sorted_names = sorted(image_names, key=lambda x: int(x.split('.')[0]))
    return sorted_names

lpips_alex = lpips.LPIPS(net='alex') # best forward scores
lpips_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

def lpips(img1, img2, net='alex', format='NCHW'):
    if format == 'HWC':
        img1 = img1.permute([2, 0, 1])[None, ...]
        img2 = img2.permute([2, 0, 1])[None, ...]
    elif format == 'NHWC':
        img1 = img1.permute([0, 3, 1, 2])
        img2 = img2.permute([0, 3, 1, 2])

    if net == 'alex':
        model = lpips_alex.to(img1.device)
        return model(img1, img2)
    elif net == 'vgg':
        model = lpips_vgg.to(img1.device)
        return model(img1, img2)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='logs/paper/full_5000_traction/estm')
    parser.add_argument('--gt_dir', type=str, default='data/endonerf_full_datasets/traction')
    parser.add_argument('--out_dir', type=str, default='./lpips_scores.csv')
    args = parser.parse_args()

    pred_images = [os.path.join(args.data_dir, i) for i in common_sort(os.listdir(args.data_dir)) if i.endswith('.png')]
    gt_images = [os.path.join(args.gt_dir+"/images", i) for i in common_sort(os.listdir(args.gt_dir+"/images")) if i.endswith('.png')]
    masks = [os.path.join(args.gt_dir+"/gt_masks", i) for i in common_sort(os.listdir(args.gt_dir+"/gt_masks")) if i.endswith('.png')]

    pred_images = [imageio.imread(i) for i in pred_images]
    gt_images = [imageio.imread(i) for i in gt_images]
    masks = np.array([(1-imageio.imread(i).astype(np.float32)/255.) for i in masks])[...,None]

    dst = np.array(gt_images) * np.array(masks)
    src = np.array(pred_images) * np.array(masks)
    imageio.imwrite('test.png', src[0])
    lpips_ = lpips(torch.from_numpy(src)/255., torch.from_numpy(dst)/255., format='NHWC')
    print(lpips_, lpips_.mean(), lpips_.shape)
    # save lpips scores
    np.savetxt('lpips_scores_2.txt', lpips_.squeeze().detach().numpy())


if __name__=="__main__":
    main()

# filename = "endo_log.txt"

# directory = 'logs/paper'

# datalist = ['cutting', 'pulling', 'pushing', 'tearing', 'thin', 'traction']

# file_list = []
# for root, dirs, files in os.walk(directory):
#     for file in files:
#         if file == filename:
#             file_list.append(os.path.join(root, file))

# if not os.path.exists(os.path.join(directory, 'performance')):
#     os.mkdir(os.path.join(directory, 'performance'))
# flag = 1

# with open(os.path.join(directory, 'performance', 'flip.csv'), mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['path:'+directory])
#     label_items = ['expname', 'FLIP']
#     writer.writerow(label_items)
#     for file_path in file_list:
#     #    file_path = "logs/paper/full_5000_pulling/endo_log.txt"
#        dirname = os.path.dirname(file_path) 
#     #    if 'enable' not in dirname:
#     #        continue
#        gt_all = []
#        masks = []
#        estmdir = os.path.join(dirname, 'estm')
#        estm = [imageio.imread(os.path.join(estmdir, fn)) for fn in common_sort(os.listdir(estmdir)) if fn.endswith('.png')]
#        prefix = (len(estm), 512, 640, 3)
#        with open(os.path.join(dirname, 'config.csv'), 'r') as config:
#             config = csv.reader(config, delimiter='\t')
#             config = {rows[0]: rows[1] for rows in config}
#             datadir = config.get('data_dirs')[2:-2]
#             gt = os.path.join(datadir, 'images')
#             mask = os.path.join(datadir, 'gt_masks')
#             gt_all = [imageio.imread(os.path.join(gt, fn)) for fn in common_sort(os.listdir(gt)) if fn.endswith('.png')]
#             masks = [imageio.imread(os.path.join(mask, fn)) for fn in common_sort(os.listdir(mask)) if fn.endswith('.png')]

#             masks = 1-np.array(masks, dtype = np.float32)/255

#             # for i in range(len(estm)):
#             #     for j in range(gt_all[0].shape[-1]):
#             #         gt_all[i][:,:,j] = (1-masks[i]) * gt_all[i][:,:,j]
#             #         estm[i][:,:,j] = (1-masks[i]) * estm[i][:,:,j]
#             # if flag:
#             #     flag = 0
#             #     imageio.imwrite('1.png', gt_all[0])
#             #     imageio.imwrite('2.png', estm[0])
#             # info = [config.get('expname'), np.mean(metrics.flip([e.astype(np.uint8) for e in estm], [g.astype(np.uint8) for g in gt_all]))]
            
#             gt_all = np.array(gt_all) *  masks[..., np.newaxis]
#             estm = np.array(estm) *  masks[..., np.newaxis]

#             # Get the 'expname' value from the 'config' object
#             expname = config.get('expname')

#             # Convert the elements of the 'estm' and 'gt_all' lists to unsigned 8-bit integers,
#             # and calculate the mean of the flipped metrics using NumPy
#             flipped_metrics = metrics.flip([e.astype(np.uint8) for e in estm], [g.astype(np.uint8) for g in gt_all], interval=10)
#             # mean_flipped_metrics = np.mean(flipped_metrics)

#             # Create a list 'info' containing the 'expname' value and the mean flipped metrics value
#             info = [expname, flipped_metrics]

#             print(info)
#             writer.writerow(info)

