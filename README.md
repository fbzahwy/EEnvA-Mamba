# EEnv-Mamba
![Python 3.11](https://img.shields.io/badge/python-3.11-g)
![pytorch 2.3.0](https://img.shields.io/badge/pytorch-2.3.0-blue.svg)

## Installation
``` shell
# pip install required packages
conda create -n eenvamamba -y python=3.11
conda activate eenvamamba
pip3 install torch===2.3.0 torchvision torchaudio
pip install seaborn thop timm einops
cd selective_scan && pip install . && cd ..
pip install -v -e .
```

## Training

```shell
python mbyolo_train.py --task train --data ultralytics/cfg/datasets/yourdataset.yaml \
 --config ultralytics/cfg/models/eenvamamba/EEnvA_Mamba.yaml \
--amp  --project ./output_dir/mscoco --name eenvamamba_n
```

## Acknowledgement

This repo is modified from open source real-time object detection codebase [Ultralytics](https://github.com/ultralytics/ultralytics). The selective-scan from [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO) and [VMamba](https://github.com/MzeroMiko/VMamba).
