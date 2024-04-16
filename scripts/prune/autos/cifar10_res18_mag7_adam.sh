#!/bin/bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=7

python main.py  --experiment 'prune'\
                --expid '4'\
                --result-dir 'experiment/prune_resutls'\
                --cfg 'cfgs/prune/autos/cifar10_res18_mag.yaml'\
                --autos_model '/home/cyf/Autosparse/experiment/pretrain_results/cifar10_resnet18_mag/batch1024_lr0.0001_pct_pepoch0_seed0_1_Adam/1_Adam_20240412-18-12/epoch8_model.pth' \
