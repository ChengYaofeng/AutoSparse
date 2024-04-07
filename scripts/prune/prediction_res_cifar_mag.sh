#!/bin/bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=0

python main.py  --experiment 'pretrain'\
                --expid 'cifar10_mag'\
                --result-dir 'experiment/pretrain_resutls'\
                --gpu 0\
                --workers 4\
                --cfg 'cfgs/autos.yaml'
