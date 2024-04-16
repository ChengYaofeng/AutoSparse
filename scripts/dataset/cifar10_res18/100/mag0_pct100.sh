#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=2


python main.py --experiment 'dataset'\
                --expid '0'\
                --lr 0.01\
                --batchsize 64\
                --seed 0\
                --schedule 'pct'\
                --cfg 'cfgs/dataset/cifar10_res18/mag_prune_100.yaml'
