#!/bin/bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=5

python main.py  --experiment 'pretrain'\
                --expid '0'\
                --result-dir 'experiment/pretrain_resutls'\
                --cfg 'cfgs/pretrain/res50/snip_autos.yaml'\
                --batchsize 512\
                --lr 0.1 \
                --seed 0