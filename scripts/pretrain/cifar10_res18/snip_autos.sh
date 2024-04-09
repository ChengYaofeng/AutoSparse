#!/bin/bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=0

python main.py  --experiment 'pretrain'\
                --expid 'cifar_mag'\
                --result-dir 'experiment/pretrain_resutls'\
                --cfg 'cfgs/pretrain/snip_autos.yaml'\
                --batchsize 4096\