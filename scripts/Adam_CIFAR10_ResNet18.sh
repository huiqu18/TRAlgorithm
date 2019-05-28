#!/usr/bin/env bash
for i in 3 4 5
do
    python train.py --dataset cifar10 --optimizer adam --model ResNet18 --save-dir ./experiments/cifar10/adam/${i} \
        --lr 0.001 --batch-size 8 --epochs 200 --gpu 3 --random-seed ${i}
done