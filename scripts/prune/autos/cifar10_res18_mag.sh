#!/bin/bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=0

python main.py  --experiment 'prune'\
                --expid '5'\
                --result-dir 'experiment/prune_resutls'\
                --cfg 'cfgs/prune/autos/cifar10_res18_mag.yaml'\
                --lr 0.01 \
                --schedule 'pct' \
                --autos_model 'experiment/pretrain_results/cifar10_resnet18_mag_pct/resnet18_batch1024_lr0.01_seed0_0/0_20240419-11-00/epoch5_model.pth' \
