#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=4


python main.py --experiment 'dataset'\
                --expid '0'\
                --lr 0.01\
                --batchsize 64\
                --seed 0\
                --schedule 'num'\
                --cfg 'cfgs/dataset/cifar10_res18/snip_prune.yaml'

