#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=1


python main.py --experiment 'dataset'\
                --expid '1'\
                --lr 0.01\
                --batchsize 64\
                --seed 1\
                --schedule 'pct'\
                --cfg 'cfgs/dataset/cifar10_res18/mag_prune _40.yaml'
