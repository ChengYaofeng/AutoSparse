#!/bin/bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=3

python main.py  --experiment 'pretrain'\
                --expid '0'\
                --result-dir 'experiment/pretrain_resutls'\
                --cfg 'cfgs/pretrain/res50/mag_autos.yaml'\
                --batchsize 1024\
                --lr 0.01 \
                --seed 0