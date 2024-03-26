#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0


python main.py --experiment 'dataset'\
                --expid 'cifar100_vgg19_mag'\
                --result-dir 'experiment/dataset_results'\
                --gpu 0\
                --workers 4\
                --cfg 'cfgs/dataset/cifar100_vgg19/mag_prune.yaml'
