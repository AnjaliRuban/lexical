#!/bin/bash
#SBATCH --job-name=pipelinetranslate
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --qos=high
#SBATCH --constrain=xeon-g6
#SBATCH --gres=gpu:volta:1
#SBATCH --array=0-4

types=${1}
name=${2}
unk=${3}
i=0
lr=1.0
warmup_steps=4000
max_steps=100000
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
--unk ${unk} \
--data_file ${types} \
--pipeline_align \
--aligner $data/data_lex/alignments/${types}/pipeline.align.pth \
--save_model $data/results/$types/${name}_pipeline_model.${i}.pth \
--TRANSLATE > $data/results/$types/${name}_pipeline_eval.${i}.out
# --one_shot \
# --TRANSLATE > $data/results/$types/pipeline_eval.test.out
# --load_model $data/results/$types/pipeline_model.${i}.pth \
