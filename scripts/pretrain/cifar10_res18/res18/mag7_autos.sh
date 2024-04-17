#!/bin/bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=7

python main.py  --experiment 'pretrain'\
                --expid '1_Adam'\
                --result-dir 'experiment/pretrain_resutls'\
                --cfg 'cfgs/pretrain/mag_autos.yaml'\
                --batchsize 1024\
                --lr 0.0001 \
                --seed 0