#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0


python main.py --experiment 'singleshot' \
                --model 'fc' \
                --model-class 'default' \
                --dataset 'cifar10' \
                --expid 'resnet18_64_(fc1_cifar10)' \
                --lr '1.2e-3' \
                --lr-drops 25 \
                --mask-scope 'global' \
                --optimizer 'adam' \
                --post-epochs '100' \
                --train-batch-size '60' \
                --prune-epochs '20' \
                --pruner "mag" \
                --weight-decay '0.0005' \
                --singleshot_compression 0.316 0 0 0\
                --seed 1 \
                --run_choice 'prediction_iterative'\
                --prediction_network '/home/liushengkai/Synaptic-Flow/prediction_model/resnet18/try.pth.tar'
