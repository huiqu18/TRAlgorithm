#!/usr/bin/env bash
for i in 3 4 5
do
    python train.py --dataset cifar10 --optimizer sgd --model ResNet18 --save-dir ./experiments/cifar10/sgd/${i} \
        --lr 0.1 --batch-size 8 --epochs 200 --gpu 2 --random-seed ${i}
done