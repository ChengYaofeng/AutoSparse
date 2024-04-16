#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=3


python main.py --experiment 'dataset'\
                --expid 'pct_1'\
                --lr 0.01\
                --batchsize 64\
                --seed 0\
                --schedule 'pct'\
                --cfg 'cfgs/dataset/cifar10_res18/snip_prune.yaml'

