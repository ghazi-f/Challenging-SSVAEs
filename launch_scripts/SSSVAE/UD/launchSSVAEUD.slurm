#!/usr/bin/env bash
#SBATCH --job-name=SSVAE
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=./udvae.txt
#SBATCH --error=./udvae.err
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1

for ALPHA in 1.0 0.1 0.01 0.001 #0.0001
do
for DEV_INDEX in 1 2 3 4 5
do
for SUP_PROP in 0.01 0.03 0.1 0.3 1.0 #0.001 0.003
do
srun python sent_train.py --losses "SSVAE" --test_name "UD/SSVAE/alt$SUP_PROP.$ALPHA.$DEV_INDEX" --graph zy --batch_size 32 --grad_accu 2 --supervision_proportion $SUP_PROP --max_len 64 --dev_index $DEV_INDEX --generation_weight $ALPHA --dataset ud --result_csv ud.csv
done
done
done

