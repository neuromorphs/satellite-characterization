#!/bin/bash
set -e

for seed in 1 2 3 4 5
do

note="Train model for seed $seed"

options="
--seed 42
--base-path ./results/artifical_motion_detection/seed_$seed

--lr 0.001
--epochs 100
--batch-size 100
--period-sim 1000
--period 100

--height 180
--width 240
--target-size 80 120

--model-name gauss_scnn
--input-avg-pooling 1
--reg-strength 1e-4
--tau-mem 0.8 0.8 0.9
--v-th 0.1 0.4 0.4
--train-tau-mem
--train-v-th
--train-gauss
"

singularity exec --app dls /containers/stable/latest python ./experiments/artificial_motion_detection.py $options --note="$note"

done