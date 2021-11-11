#!/bin/bash

####################### ogbn-mag dataset
File=pseudo.py
Data=ogbn-mag
Aggre=lstm
batch_size=(157393 78697 39349 19675 9838 4949 2400 1200)


####################### reddit dataset 
# File=pseudo.py
# Data=reddit
# Aggre=mean
# batch_size=(153431 76716 38358 19179 9590 4795 2400 1200)

lr=003
runs=2
epochs=6

mkdir range_${lr}
mkdir random_${lr}
cd range_${lr} 
mkdir ${Data}_${Aggre}_pseudo_log/
cd ../random_${lr}
mkdir ${Data}_${Aggre}_pseudo_log/
cd ..




for i in ${batch_size[@]};do
  python $File \
  --dataset $Data \
  --aggre $Aggre \
  --selection-method range \
  --batch-size $i \
  --num-runs $runs \
  --num-epochs $epochs \
  --eval-every 5 > range_${lr}/${Data}_${Aggre}_pseudo_log/bs_${i}_6_epoch.log
# done
# for i in ${batch_size[@]};do
  python $File \
  --dataset $Data \
  --aggre $Aggre \
  --selection-method random \
  --batch-size $i \
  --num-runs $runs \
  --num-epochs $epochs \
  --eval-every 5 > random_${lr}/${Data}_${Aggre}_pseudo_log/bs_${i}_6_epoch.log
done

