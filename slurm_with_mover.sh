#!/bin/bash

#SBATCH --partition=your_partition
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=80           # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 48:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/%x-%j.out           # output file name
#SBATCH --error=logs/%x-%j.err
#SBATCH --constraint='volta32gb'
#SBATCH --mem=480G
#SBATCH --open-mode=append

#SBATCH --mail-user=myemail@gmail.com
#SBATCH --mail-type=begin,end,fail # mail once the job finishes

JOB_NAME=$(date '+%Y-%m-%d_%H:%M:%S_')${SLURM_JOB_ID}_${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}
DESTINATION_DIR='/path/to/checkpoints/'

# Ensure that the job type is either 'localization' or 'retrieval'
# if [ "$JOB_TYPE" != "localization" ] && [ "$JOB_TYPE" != "retrieval" ]; then
#     echo "Invalid job type: $JOB_TYPE. Must be 'localization' or 'retrieval'."
#     exit 1
# fi

# Create the destination directory
mkdir -p $DESTINATION_DIR/$JOB_NAME/

cp -R llava/ $DESTINATION_DIR/$JOB_NAME/
cp -R scripts/ $DESTINATION_DIR/$JOB_NAME/

# Conditional file copying and job submission based on job type
cp pretrain_v15.sh $DESTINATION_DIR/$JOB_NAME/
cd "$DESTINATION_DIR/$JOB_NAME/"


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

########### HYPERPARAMETER SEARCH ####
LEARNING_RATE=2e-3
BATCH_SIZE=2
LR_SCHEDULER="cosine"
EPOCH=40

# Parse options
while getopts "l:b:e:s:" opt; do
  case $opt in
    l) LEARNING_RATE=$OPTARG;;
    b) BATCH_SIZE=$OPTARG;;
    e) EPOCH=$OPTARG;;
    s) LR_SCHEDULER=$OPTARG;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1;;
  esac
done
echo "Learning Rate: $LEARNING_RATE"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCH"
echo "Scheduler: $LR_SCHEDULER"
######################################
###### Other parameters -- change here so that we can put code without waiting #############
DATA_PATH="/path/to/LLaVA/data/pose_dataset_v1_finegrained_only_pose1_train.json"
DATA_PATH_VAL="/path/to/LLaVA/data/pose_dataset_v1_finegrained_only_pose1_val.json"
#######################################

source pretrain_v15.sh

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --node_rank \$SLURM_PROCID $CMD" 2>&1