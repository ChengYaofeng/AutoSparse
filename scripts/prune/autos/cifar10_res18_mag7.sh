#!/bin/bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=7

python main.py  --experiment 'prune'\
                --expid '2'\
                --result-dir 'experiment/prune_resutls'\
                --cfg 'cfgs/prune/autos/cifar10_res18_mag.yaml'\
                --autos_model 'experiment/pretrain_results/cifar10_resnet18_mag/batch4096_lr0.0001_pct_pepoch0_seed0_0_SGD/0_SGD_20240412-14-55/epoch0_model.pth' \
