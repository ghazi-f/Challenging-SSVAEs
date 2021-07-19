#!/usr/bin/env bash
#SBATCH --job-name=VAENoKL
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=./dbpediavaenokl.txt
#SBATCH --error=./dbpediavaenokl.err
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1

for ALPHA in 1.0 0.1 0.01 0.001 #0.0001
do
for DEV_INDEX in 1 2 3 4 5
do
for SUP_PROP in 0.01 0.03 0.1 0.3 1.0 #0.001 0.003
do
srun python sent_train.py --losses "SSVAE" --test_name "DBPEDIA/SSVAENoKL/alt$SUP_PROP.$ALPHA.$DEV_INDEX" --graph zy --anneal_kl0 -1 --batch_size 32 --grad_accu 2 --supervision_proportion $SUP_PROP --max_len 80 --dev_index $DEV_INDEX --generation_weight $ALPHA --dataset dbpedia --result_csv dbpedia.csv
done
done
done

