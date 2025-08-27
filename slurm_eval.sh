#!/bin/bash

#SBATCH --partition=your_partition
#SBATCH --nodes=1
#SBATCH --array=0-31
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


export USE_POSE="True"
export USE_EGOEXO="False"
export FREEZE_LLAMA3="True"
export POSE_GENERATION="False"
export PRETRAIN_MM_MLP_ADAPTER="/path/to/checkpoints-10-2-2e-3-cosine/checkpoint-6384/mm_projector.bin"
if [ "$POSE_GENERATION" == "True" ]; then
  export POSE_MODULE=None
else
  export POSE_MODULE="pct_tokenizer"
fi

export NCCL_INFO=DEBUG

bash pretrain_eval.sh



