# LerPlane

### [Paper](https://arxiv.org/abs/2103.11547) | [Models](https://arxiv.org/abs/2103.11547)

> [LerPlane: Neural Representations for Fast 4D Reconstruction of Deformable Tissues](https://arxiv.org/abs/2103.11547) \
> Chen Yang, Kailing Wang, Yuehao Wang, Xiaokang Yang, Wei Shen \
> MICCAI2023, Oral, STAR Award

## Schedule
- [ ] Initial Code Release.
- [x] Further check of the reproducibility.
- [x] The code release for the extended version.

## Introduction
Reconstructing deformable tissues from endoscopic stereo videos in robotic surgery is crucial for various clinical applications. However, existing methods relying only on implicit representations are computationally expensive and require dozens of hours, which limits further practical applications. To address this challenge, we introduce LerPlane, a novel method for fast and accurate reconstruction of surgical scenes under a single-viewpoint setting. LerPlane treats surgical procedures as 4D volumes and factorizes them into explicit 2D planes of static and dynamic fields, leading to a compact memory footprint and significantly accelerated optimization. The efficient factorization is accomplished by fusing features obtained through linear interpolation of each plane and enable using lightweight neural networks to model surgical scenes. Besides, LerPlane shares static fields, significantly reducing the workload of dynamic tissue modeling. We also propose a novel sample scheme to boost optimization and improve performance in regions with tool occlusion and large motions. Experiments on DaVinci robotic surgery videos demonstrate that LerPlane accelerates optimization by over 100× while maintaining high quality across various non-rigid deformations, showing significant promise for future intraoperative surgery applications.

## Installation

### Set up the python environment
<details> <summary>Tested with an Ubuntu workstation i9-12900K, 3090GPU</summary>

```
conda create -n lerplane python=3.9
conda activate lerplane
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch 
pip install -r requirments.txt
```
</details>

### Set up datasets
<details> <summary>Download the datasets</summary> 

Please download the dataset from [EndoNeRF](https://github.com/med-air/EndoNeRF) 
</details>

### training
<details> <summary>Download the datasets</summary> 

Lerplane uses configs to control the training process. The configs are stored in the `lerplane/configs` folder.
To train a model, run the following command:
```
python train.py --config configs/lerplane.yaml
```
</details>

## Acknowledgements
We would like to acknowledge the following inspring work:
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
