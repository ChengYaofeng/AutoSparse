#!/bin/bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=0

python main.py  --experiment 'prune'\
                --expid '0'\
                --result-dir 'experiment/prune_resutls'\
                --cfg 'cfgs/prune/snip/cifar10_res18.yaml'\
                --schedule 'pct'\