#!/bin/bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=5

python main.py  --experiment 'pretrain'\
                --expid '2'\
                --result-dir 'experiment/pretrain_resutls'\
                --cfg 'cfgs/pretrain/mag_autos.yaml'\
                --batchsize 4096\
                --lr 0.0001 \
                --seed 1