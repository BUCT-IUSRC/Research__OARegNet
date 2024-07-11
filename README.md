## OARegNet: Online Adaptive Point Cloud Registration for Embodied Intelligent Driving

Jinying Zhang, Yadong Wang, Sihan Chen, Chao Xu, Tianyu Shen*, Kunfeng Wang*

(*Corresponding authors )

An official implementation of the OARegNet adversarial patch generation framework.

### Framework Overview

OARegNet is an innovative online point cloud registration network designed for V2X technology, addressing  gaps in vehicle-infrastructure cooperation. It introduces an adaptive density learning module to handle registration challenges due to point cloud density differences and occlusions, and an adaptive feature fusion module that utilizes a multi-head attention mechanism for effective fusion of complex keypoint features.

![](https://upload-images.jianshu.io/upload_images/29903503-ca811b6bc99c9f0c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



### Install

#### Environment

```
conda create -n OARegNet python=3.7
conda create OARegNet
pip install -r requirments.txt
```

Please refer to [PyTorch Docs](https://pytorch.org/get-started/previous-versions/) to install torch and torchvision for better compatibility.

#### Dataset:

DAIR-V2X

**Training device**: NVIDIA RTX 3090

### Train
The training of the whole network is divided into two steps: we firstly train the feature extraction module and then train the network based on the pretrain features.
#### Train feature extraction
- Train keypoints detector by running `sh scripts/train_V2X_det.sh` , please reminder to specify the `GPU`,`DATA_ROOT`,`CKPT_DIR`,`RUNNAME`,`WANDB_DIR` in the scripts.
- Train descriptor by running `sh scripts/train_V2X_desc.sh` , please reminder to specify the `GPU`,`DATA_ROOT`,`CKPT_DIR`,`RUNNAME`,`WANDB_DIR` and `PRETRAIN_DETECTOR` in the scripts.

#### Train the whole network
Train the network by running `sh scripts/train_V2X_reg.sh`, please reminder to specify the `GPU`,`DATA_ROOT`,`CKPT_DIR`,`RUNNAME`,`WANDB_DIR` and `PRETRAIN_FEATS` in the scripts.

**Update**: Pretrained weights for detector and descriptor are provided in `ckpt/pretrained`. If you want to train descriptor, you can set `PRETRAIN_DETECTOR` to `DATASET_keypoints.pth`. If you want to train the whole network, you can set `PRETRAIN_FEATS` to `DATASET_feats.pth`.

### Test
We provide pretrain models in `ckpt/pretrained`, please run `sh scripts/test_V2X.sh` , please reminder to specify `GPU`,`DATA_ROOT`,`SAVE_DIR` in the scripts. The test results will be saved in `SAVE_DIR`.

### Acknowledgments
We want to thank following open-source projects for the help of the implementation:

- [HRehNet](https://github.com/ispc-lab/HRegNet)
- [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X)


### Contact Us
If you have any problem about this work, please feel free to reach us out at 2022200811@buct.edu.cn
