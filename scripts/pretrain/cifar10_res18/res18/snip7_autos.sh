#!/bin/bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=4

python main.py  --experiment 'pretrain'\
                --expid '1'\
                --result-dir 'experiment/pretrain_resutls'\
                --cfg 'cfgs/pretrain/res18/snip_autos.yaml'\
                --batchsize 4096\
                --lr 0.01 \
                --seed 0