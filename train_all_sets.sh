#!/bin/bash
for ((i=1; i<=8; i++))
do
    echo $i
    python train.py --essay_set_id $i --num_samples 2
done
