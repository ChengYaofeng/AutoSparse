#!/bin/bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=0

python main.py  --experiment 'prune'\
                --expid '2'\
                --result-dir 'experiment/prune_resutls'\
                --cfg 'cfgs/prune/autos/cifar10_res18_mag.yaml'\
                --autos_model 'experiment/pretrain_results/cifar_pretrain_batch2048_lr0.001/cifar_pretrain_batch2048_lr0.001-20240408-12-24/model.pth' \
                # --singleshot_compression 0.8 0.7 0.5 0.3 0.1 0.05 0.01\
