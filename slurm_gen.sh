#!/bin/bash

#SBATCH --partition=your_partition
#SBATCH --nodes=1
#SBATCH --array=0-15
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=80           # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 48:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%A_%a.out           # output file name
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --constraint='volta32gb'
#SBATCH --mem=480G
#SBATCH --open-mode=append

#SBATCH --mail-user=myemail@gmail.com
#SBATCH --mail-type=begin,end,fail # mail once the job finishes

export NCCL_DESYNC_DEBUG=1
export NCCL_ASYNC_ERROR_HANDLING=1

python llava/eval/model_vqa.py \
    --question-file /path/to/LLaVA/data/pose_dataset_concise_NE_DTW_pose_retrieval_v2_with_soccer_val_N5.json \
    --model-base /path/to/checkpoint-1400/ \
    --answers-file output/pose_gen_pose_mymethod_${SLURM_ARRAY_TASK_ID}_of_16.jsonl \
    --use-pose True \
    --use-egoexo False \
    --pose-generation True \
    --chunk-idx ${SLURM_ARRAY_TASK_ID} \
    --num-chunks 16



