#!/usr/bin/env bash
for i in 1 2 3 4 5
do
    python train.py --dataset cifar10 --optimizer tradam --model ResNet18 --save-dir ./experiments/cifar10/tradam/${i} \
        --lr 0.001 --batch-size 8 --epochs 200 --gpu 1 --random-seed ${i}
done