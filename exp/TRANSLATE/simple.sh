#!/bin/bash
#SBATCH --job-name=simpletranslate
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --qos=high
#SBATCH --constrain=xeon-g6
#SBATCH --gres=gpu:volta:1
#SBATCH --array=0-4

types=${1}
i=0
lr=1.0
warmup_steps=4000
max_steps=80000
home="../.."
data="../../.."
mkdir -p $data/results/$types

python -u  $home/main.py \
--seed $i \
--n_batch 8 \
--n_layers 2 \
--noregularize \
--dim 512 \
--lr ${lr} \
--temp 1.0 \
--dropout 0.4 \
--beam_size 5 \
--gclip 5.0 \
--accum_count 4 \
--valid_steps 5000 \
--warmup_steps ${warmup_steps} \
--max_step ${max_steps} \
--tolarance 10 \
--copy \
--data_file ${types} \
--aligner $data/data_lex/alignments/${types}/simple.align.v3.json \
--save_model $data/results/$types/simple_model.${i}.pth \
--TRANSLATE > $data/results/$types/simple_eval.${i}.out
