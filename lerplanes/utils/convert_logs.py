import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def convert_logs(logs_path, save=None):
    # load log file, exps/lerplane/hamlyn_32k_gt_dp/hamlyn_seq1/endo_log.txt
    # {'psnr': [33.371252644856774], 'ssim': [0.9291370717684427], 'lpips': [0.12446547845999399], 'masked_psnr': [32.76124434153239], 'FLIP': [0.09649633333333335]}
    # load this file to format a dict
    performance = {}
    with open(logs_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('{'):
                performance = eval(line)
                break
    # turn a dict into a pandas dataframe
    performance = pd.DataFrame(performance)
    if save is not None:
        with open(os.path.join(os.path.dirname(logs_path), 'metrics.txt'), 'w') as f:
            f.write('{}\n'.format(performance['psnr'].item()))
            # f.write('SSIM,{}\n'.format(ssim_.item()))
            # f.write('LPIPS,{}\n'.format(torch.mean(lpips_).item()))
            # f.write('FLIPS,{}\n'.format(flip_))
            # f.write('Masked PSNR,{}\n'.format(masked_psnr.item()))

            f.write('{}\n'.format(performance['lpips'].item()))
            f.write('{}\n'.format(performance['ssim'].item()))
            f.write('{}\n'.format(performance['FLIP'].item()))
            f.write('{}\n'.format(performance['masked_psnr'].item()))

    return performance

if __name__ == "__main__":
    tgt_path = ['exps/endonerf_iter32k_gtdp/cutt/endo_log.txt',
                'exps/endonerf_iter32k_gtdp/pull/endo_log.txt',
                'exps/endonerf_iter32k_gtdp/push/endo_log.txt',
                'exps/endonerf_iter32k_gtdp/tear/endo_log.txt',
                'exps/endonerf_iter32k_gtdp/thin/endo_log.txt',
                'exps/endonerf_iter32k_gtdp/trac/endo_log.txt',
                ]
    
    for i in tgt_path:
        convert_logs(i, save=True)
    # pf = convert_logs('./exps/lerplane/hamlyn_32k_gt_dp/hamlyn_seq1/endo_log.txt', save=True)
    # print(pf)
