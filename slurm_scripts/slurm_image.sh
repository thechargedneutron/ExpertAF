#!/bin/bash

#SBATCH --partition=your_partition
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=80           # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 12:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.out           # output file name
#SBATCH --error=logs/%x-%j.err
#SBATCH --constraint='volta32gb'
#SBATCH --mem=480G
#SBATCH --open-mode=append

#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=begin,end,fail # mail once the job finishes


GPUS_PER_NODE=8
NNODES=$SLURM_NNODES
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
"

export CMD="llava/train/train_xformers.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --version plain \
    --data_path ../../multi_vid_retrieval_f23/LLaVA/chat.json \
    --image_folder ../../multi_vid_retrieval_f23/LLaVA/cc3m/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 False \
    --output_dir ./checkpoints/llava-v1.5-13b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy 'no' \
    --save_strategy 'steps' \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type 'cosine' \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
"

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --node_rank \$SLURM_PROCID $CMD" 2>&1