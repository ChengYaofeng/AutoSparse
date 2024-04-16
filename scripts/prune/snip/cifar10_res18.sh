#!/bin/bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=7

python main.py  --experiment 'prune'\
                --expid '0'\
                --lr 0.01 \
                --result-dir 'experiment/prune_resutls'\
                --cfg 'cfgs/prune/snip/cifar10_res18.yaml'\
                --schedule 'pct'\