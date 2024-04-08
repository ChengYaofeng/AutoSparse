#!/bin/bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=0

python main.py  --experiment 'prune'\
                --expid 'cifar10_mag'\
                --result-dir 'experiment/prune_resutls'\
                --cfg 'cfgs/prune/autos/autos_prune.yaml'\
                --autos_model '/home/cyf/sparsity/AutoSparse/experiment/pretrain_results/cifar_mag/cifar_mag-20240407-20-43/model.pth' \
                # --singleshot_compression 0.8 0.7 0.5 0.3 0.1 0.05 0.01\
