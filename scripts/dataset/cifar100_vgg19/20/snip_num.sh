#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=7


python main.py --experiment 'dataset'\
                --expid '0'\
                --lr 0.1\
                --seed 0\
                --schedule 'num'\
                --cfg 'cfgs/dataset/cifar100_vgg19/snip_prune.yaml' \
                --batchsize 128\

