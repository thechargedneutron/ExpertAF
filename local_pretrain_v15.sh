#### Assumes you are already in a SLURM node and want to run locally for debugging/development

export NCCL_DESYNC_DEBUG=1
export NCCL_ASYNC_ERROR_HANDLING=1

########### HYPERPARAMETER SEARCH ####
LEARNING_RATE=2e-3
BATCH_SIZE=2
LR_SCHEDULER="cosine"
EPOCH=10

USE_POSE="True"
USE_EGOEXO="False"
FREEZE_LLAMA3="False"
POSE_GENERATION="True"
if [ "$POSE_GENERATION" == "True" ]; then
  POSE_MODULE="None"
else
  POSE_MODULE="pct_tokenizer"
fi

###### Other parameters -- change here so that we can put code without waiting #############
DATA_PATH="/path/to/LLaVA/data/pose_dataset_concise_NE_DTW_pose_retrieval_with_soccer_train_N5.json"
DATA_PATH_VAL="/path/to/LLaVA/data/pose_dataset_concise_NE_DTW_pose_retrieval_with_soccer_val_N5.json"
#######################################

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

source pretrain_v15.sh
bash -c "deepspeed $CMD"