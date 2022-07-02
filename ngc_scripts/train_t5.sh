#! /bin/bash
pip install wandb
pip install h5py
pip install clean-fid
wandb login 69a3bddb4146cf76113885de5af84c7f4c165753
python3 train_image.py --config configs/cifar/tunet-t5-ngc.yaml --num_gpus 8 --log