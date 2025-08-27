#!/bin/bash


#SBATCH --partition=your_partition
#SBATCH --nodes=4
#SBATCH --array=0-2
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

JOB_NAME=$(date '+%Y-%m-%d_%H:%M:%S_')${SLURM_ARRAY_JOB_ID}_${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}
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

export NCCL_DESYNC_DEBUG=1
export NCCL_ASYNC_ERROR_HANDLING=1


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
# Define the arrays of parameter values
LEARNING_RATES=(2e-5 5e-6 7e-6)  # Example values
BATCH_SIZES=(2)            # Example values
LR_SCHEDULERS=("cosine")  # Example values
EPOCHS=(10)                  # Example values
USE_POSE="True"
USE_EGOEXO="False"
FREEZE_LLAMA3="False"
POSE_GENERATION="False"
if [ "$POSE_GENERATION" == "True" ]; then
  POSE_MODULE=None
else
  POSE_MODULE="pct_tokenizer"
fi

# Calculate total combinations and adjust SLURM array range
NUM_LEARNING_RATES=${#LEARNING_RATES[@]}
NUM_BATCH_SIZES=${#BATCH_SIZES[@]}
NUM_LR_SCHEDULERS=${#LR_SCHEDULERS[@]}
NUM_EPOCHS=${#EPOCHS[@]}
TOTAL_COMBINATIONS=$(($NUM_LEARNING_RATES * $NUM_BATCH_SIZES * $NUM_LR_SCHEDULERS * $NUM_EPOCHS))

# Dynamically set the SBATCH array range to accommodate all combinations
# Note: This line is for illustration and won't actually change the SBATCH option dynamically.
# You might need to set this manually or use a script to generate your submission script.
#SBATCH --array=0-$(($TOTAL_COMBINATIONS - 1))

# Calculate parameter index based on SLURM_ARRAY_TASK_ID
LR_INDEX=$(($SLURM_ARRAY_TASK_ID / ($NUM_BATCH_SIZES * $NUM_LR_SCHEDULERS * $NUM_EPOCHS) % $NUM_LEARNING_RATES))
BS_INDEX=$(($SLURM_ARRAY_TASK_ID / ($NUM_LR_SCHEDULERS * $NUM_EPOCHS) % $NUM_BATCH_SIZES))
LS_INDEX=$(($SLURM_ARRAY_TASK_ID / $NUM_EPOCHS % $NUM_LR_SCHEDULERS))
EP_INDEX=$(($SLURM_ARRAY_TASK_ID % $NUM_EPOCHS))

# Set the parameter values based on the current Slurm array index
LEARNING_RATE=${LEARNING_RATES[$LR_INDEX]}
BATCH_SIZE=${BATCH_SIZES[$BS_INDEX]}
LR_SCHEDULER=${LR_SCHEDULERS[$LS_INDEX]}
EPOCH=${EPOCHS[$EP_INDEX]}
echo "Learning Rate: $LEARNING_RATE"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCH"
echo "Scheduler: $LR_SCHEDULER"
######################################
###### Other parameters -- change here so that we can put code without waiting #############
DATA_PATH="/path/to/LLaVA/data/pose_dataset_concise_NE_DTW_pose_retrieval_v2_with_soccer_train_N5.json"
DATA_PATH_VAL="/path/to/LLaVA/data/pose_dataset_concise_NE_DTW_pose_retrieval_v2_with_soccer_val_N5.json"
#######################################

source pretrain_v15.sh

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --node_rank \$SLURM_PROCID $CMD" 2>&1