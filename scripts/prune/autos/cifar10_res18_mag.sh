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
                --autos_model '/home/cyf/Autosparse/experiment/pretrain_results/cifar10_resnet18_mag/batch4096_lr0.0001_pct_pepoch0_seed1_3/3_20240412-14-58/epoch4_model.pth' \
