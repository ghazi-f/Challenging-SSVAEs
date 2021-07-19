#!/usr/bin/env bash
#SBATCH --job-name=Supervised
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t4
#SBATCH --output=./imdbsup.txt
#SBATCH --error=./imdbsup.err
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1

for ALPHA in 1.0
do
for DEV_INDEX in 1 2 3 4 5
do
for SUP_PROP in 1.0
do
srun python sent_train.py --losses "S" --test_name "IMDB4/Supervised/alt$SUP_PROP.$ALPHA.$DEV_INDEX" --graph y --anneal_kl0 -1 --batch_size 32 --grad_accu 1 --supervision_proportion $SUP_PROP --max_len 256 --dev_index $DEV_INDEX --generation_weight $ALPHA --dataset imdb --result_csv imdbtran.csv
done
done
done
