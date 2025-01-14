#!/bin/bash

LLM_VERSION="/mnt/afs/hfmodel/deepseek-coder-6.7b-instruct" # Change to your own LLM path 
LLM_VERSION_CLEAN=$(basename $LLM_VERSION)
VISION_MODEL_VERSION="/mnt/afs/hfmodel/siglip-so400m-patch14-384" # Change to your own ViT path 
VISION_MODEL_VERSION_CLEAN=$(basename $VISION_MODEL_VERSION)

PROMPT_VERSION=plain
DATA="/data/home/zhaoxuanle/ChartCoder/pt_data.json" # Change to your own data path 

BASE_RUN_NAME="chartcoder-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
output_dir="./checkpoints/projectors"

NUM_GPUS_PER_WORKER=8 
MASTER_PORT=21231

OPTIONS_NCCL="NCCL_DEBUG=ERROR NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 CUDA_LAUNCH_BLOCKING=0 NCCL_IB_GID_INDEX=3"

run_cmd="${OPTIONS_NCCL} deepspeed --num_gpus ${NUM_GPUS_PER_WORKER} --master_port ${MASTER_PORT} \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA} \
    --image_folder /path/to/images \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts "mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/projectors/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy \"no\" \
    --save_strategy \"no\" \
    --save_steps 50000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type \"cosine\" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name $BASE_RUN_NAME \
    --attn_implementation sdpa"

eval $run_cmd 2>&1 | tee "$output_dir/train.log"
