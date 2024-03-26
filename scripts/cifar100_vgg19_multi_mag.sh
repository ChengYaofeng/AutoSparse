#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0


python main.py --experiment 'singleshot' \
                --model 'vgg19-bn' \
                --model-class 'lottery' \
                --dataset 'cifar100' \
                --expid 'cifar100_vgg19_40it_snip_important' \
                --result-dir 'results_new' \
                --lr '0.1' \
                --lr-drops 60 120 \
                --mask-scope 'global' \
                --optimizer 'momentum' \
                --post-epochs '160' \
                --train-batch-size '128' \
                --prune-epochs '40' \
                --pruner "snip" \
                --weight-decay '0.0001' \
                --singleshot_compression  1 \
                --run_choice 'prune_prediction'\
                --prediction_network '/home/cyf/sparsity/AutoSparse/prediction_model/resnet18/resnet18/cifar10/snip/cifar10_res18_40it_snip_important-2024-01-21-14-22resnet182024-02-18-09-53/model.pth.tar'




