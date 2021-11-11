#!/bin/bash


# File=ideal_mini_batch_subgraph_train.py
# Data=ogbn-mag
# Aggre=lstm
# batch_size=(157393 78697 39349 19675 9838 4919 2400 1200)


# input =[108680, 54299, 27111, 13568, 6791, 3398, 1696]

# herer, batch size should be the size of ideal mini barch subgraph output nodes 
File=ideal_mini_batch_subgraph_train.py
Data=reddit
Aggre=mean
batch_size=(18900 6800 2950 1350 650 324 159)


lr=003
epochs=6

mkdir ideal_perf_${lr}
cd ideal_perf_${lr} 
mkdir ${Data}_${Aggre}_ideal_log/
cd ..


for i in ${batch_size[@]};do
  python $File \
  --dataset $Data \
  --aggre $Aggre \
  --batch-size $i \
  --num-epochs 6 \
  --eval-every 5 > ideal_perf_${lr}/${Data}_${Aggre}_ideal_log/bs_${i}_6_epoch.log
done
# for i in ${batch_size[@]};do
  # python $File \
  # --dataset $Data \
  # --aggre $Aggre \
  # --selection-method random \
  # --batch-size $i \
  # --num-epochs 1 \
  # --eval-every 5 > ideal_perf_random_${lr}/${Data}_${Aggre}_pseudo_log/bs_${i}_1_epoch.log
# done


