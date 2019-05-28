#!/usr/bin/env bash
for i in 1 2 3 4 5
do
    python train.py --dataset cifar10 --optimizer trsgd --model ResNet18 --save-dir ./experiments/cifar10/trsgd/${i} \
        --lr 0.1 --batch-size 8 --epochs 200 --gpu 0 --random-seed ${i}
done