# LerPlane

### [Paper](https://arxiv.org/pdf/2305.19906.pdf)

> [LerPlane: Neural Representations for Fast 4D Reconstruction of Deformable Tissues](https://arxiv.org/pdf/2305.19906.pdf) \
> Chen Yang, Kailing Wang, Yuehao Wang, Xiaokang Yang, Wei Shen \
> MICCAI2023, Oral, STAR Award

![](lerplanes/images/overview.png)

## Schedule
- [x] Initial Code Release.
- [ ] Further check of the reproducibility.
- [ ] The code release for the extended version.

## Introduction
Reconstructing deformable tissues from endoscopic stereo videos in robotic surgery is crucial for various clinical applications. However, existing methods relying only on implicit representations are computationally expensive and require dozens of hours, which limits further practical applications. To address this challenge, we introduce LerPlane, a novel method for fast and accurate reconstruction of surgical scenes under a single-viewpoint setting. LerPlane treats surgical procedures as 4D volumes and factorizes them into explicit 2D planes of static and dynamic fields, leading to a compact memory footprint and significantly accelerated optimization. The efficient factorization is accomplished by fusing features obtained through linear interpolation of each plane and enabling the use of lightweight neural networks to model surgical scenes. Besides, LerPlane shares static fields, significantly reducing the workload of dynamic tissue modeling. We also propose a novel sample scheme to boost optimization and improve performance in regions with tool occlusion and large motions. Experiments on DaVinci robotic surgery videos demonstrate that LerPlane accelerates optimization by over 100× while maintaining high quality across various non-rigid deformations, showing significant promise for future intraoperative surgery applications.

[http://loping151.top/images/trainging_speed_vs_endo.mp4](https://user-images.githubusercontent.com/97866915/274361556-3cdcd11b-3bb1-46a4-bd01-c5e9de160828.mp4)

## Installation

### Set up the Python environment
<details> <summary>Tested with an Ubuntu workstation i9-12900K, 3090GPU.</summary>

```
conda create -n lerplane python=3.9
conda activate lerplane
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch 
```
We notice tiny-cuda-nn is sometimes not compilable on some latest GPUs like RTX4090(tested 2023.1). If you found `OSError` while installing tiny-cuda-nn, you can refer to this [issue](https://github.com/NVlabs/tiny-cuda-nn/issues/245) or this [article](https://zhuanlan.zhihu.com/p/643834111). We've successfully built the same env on an Ubuntu 22.04 workstation i7-13700K, 4090GPU with the commands above.
</details>

### Set up datasets
<details> <summary>Download the datasets</summary> 

Please download the dataset from [EndoNeRF](https://github.com/med-air/EndoNeRF) 

To use the example config, organize your data like:
```
data
| - endonerf_full_datasets
|   | - cutting_tissues_twice
|   | - pushing_soft_tissues
| - hamlyn_lerplane
|   | - hamlyn1
|   | - hamlyn2
| - YourCustomDatasets
```

</details>

### training
<details> <summary>Using configs for training</summary> 

Lerplane uses configs to control the training process. The configs are stored in the `lerplane/configs` folder.
To train a model, run the following command:
```
export CUDA_VISIBLE_DEVICES=0
PYTHONPATH='.' python lerplanes/main.py --config-path lerplanes/config/example-9k.py
```
</details>

### Evaluation
We use the same evaluation protocol as [EndoNeRF](https://github.com/med-air/EndoNeRF). So please follow the instructions in EndoNeRF.

## Acknowledgements
We would like to acknowledge the following inspiring work:
- [EDSSR](https://arxiv.org/pdf/2107.00229) (Long et al.)
- [EndoNeRF](https://github.com/med-air/EndoNeRF) (Wang et al.)
- [K-Planes](https://sarafridov.github.io/K-Planes/) (Fridovich-Keil et al.)

Big thanks to [NeRFAcc](https://www.nerfacc.com/) (Li et al.) for their efficient implementation, which has significantly accelerated our rendering.

## Citation

If you find this code useful for your research, please use the following BibTeX entry

```
@article{yang2023neural,
  title={Neural LerPlane Representations for Fast 4D Reconstruction of Deformable Tissues},
  author={Yang, Chen and Wang, Kailing and Wang, Yuehao and Yang, Xiaokang and Shen, Wei},
  journal={MICCAI},
  year={2023}
}
```
