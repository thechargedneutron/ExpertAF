current_hostname=$(hostname)

if [[ "$current_hostname" =~ host1* ]]; then
    DATA_PATH="/path/to/LLaVA/data/pose_dataset_v1_train.json"
    DATH_PATH_VAL="/path/to/LLaVA/data/pose_dataset_v1_val.json"
    POSE_PATH="/path/to/datasets/egoexo4d/poses_finalized_v1/"
    POSE_WEIGHTS="/path/to/LLaVA/models/best_avg_joint_loss_epoch_150.pth"
    MAIN_CODE="llava/train/train_mem.py"
elif [[ "$current_hostname" =~ host2* ]]; then
    DATA_PATH="/path/to/LLaVA/data/pose_dataset_v1_train.json"
    DATH_PATH_VAL="/path/to/LLaVA/data/pose_dataset_concise_NE_DTW_pose_retrieval_with_soccer_val_N5_retrieval_400.json"
    POSE_PATH="/path/to/DatasetCurate/poses_finalized_v1/"
    POSE_WEIGHTS="/path/to/LLaVA/models/best_avg_joint_loss_epoch_150.pth"
    MAIN_CODE="llava/train/train_xformers.py"
else
    echo "Error: Unknown hostname."
    exit 1
fi

deepspeed ${MAIN_CODE} \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --version llama_3 \
    --data_path ${DATA_PATH} \
    --data_path_val ${DATH_PATH_VAL} \
    --pose_path ${POSE_PATH} \
    --pose_token_path /path/to/PCT_Tokenizer/tokens/ \
    --egoexo_path /path/to/InternVideo/InternVideo2/multi_modality/demo/egoexo4d_feats_with_soccer/ \
    --pose_module ${POSE_MODULE} \
    --pose_weights ${POSE_WEIGHTS} \
    --num_pose_frames 10 \
    --tune_mm_mlp_adapter ${FREEZE_LLAMA3} \
    --mm_projector_type mlp2x_gelu \
    --use_pose ${USE_POSE} \
    --use_egoexo ${USE_EGOEXO} \
    --pose_generation ${POSE_GENERATION} \
    --pretrain_mm_mlp_adapter ${PRETRAIN_MM_MLP_ADAPTER} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 False \
    --output_dir ./checkpoints/ \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --eval_only True \
    --eval_in_parts True \
    --save_all_losses True \
    --eval_steps 0.05 \
    --save_steps 0.05 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard

# The above code saves the eval file under scores/eval_scores.pkl.
# Below code uses that to find the MCQ score
# python calculate_mcq_scores.py ${DATH_PATH_VAL}