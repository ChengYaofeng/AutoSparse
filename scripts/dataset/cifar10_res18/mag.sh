#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0


python main.py --experiment 'dataset'\
                --expid 'cifar10_res18_mag'\
                --gpu 0\
                --workers 4\
                --cfg 'cfgs/dataset/cifar10_res18/mag_prune.yaml'
                # --result-dir 'experiment/dataset_results'\
