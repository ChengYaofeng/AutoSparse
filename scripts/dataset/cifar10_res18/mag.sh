#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0


python main.py --experiment 'dataset'\
                --expid 'cifar10_res18_mag_batch32_lr0.01'\
                --lr 0.01\
                --batchsize 64\
                --cfg 'cfgs/dataset/cifar10_res18/mag_prune.yaml'
