LLM_VERSION="/mnt/afs/hfmodel/deepseek-coder-6.7b-instruct"
LLM_VERSION_CLEAN=$(basename $LLM_VERSION)
VISION_MODEL_VERSION="/mnt/afs/zxl/hfmodel/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN=$(basename $VISION_MODEL_VERSION)
cur_time=$(date "+%m%d-%H")
############### Pretrain ################

PROMPT_VERSION="llava_deepseekcoder"
DATA="chart2code160k.json"

BASE_RUN_NAME="chartcoder-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

MID_RUN_NAME="chartcoder-${LLM_VERSION_CLEAN}-${VISION_MODEL_VERSION_CLEAN}-mlp2x_gelu-finetune-${LLM_VERSION_CLEAN%%-*}-${cur_time}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"
########################################

NUM_WORKERS=4
NUM_GPUS_PER_WORKER=8
MASTER_PORT=21232
HOST_FILE_PATH="hostfile"

OPTIONS_NCCL="NCCL_DEBUG=EEROR NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 CUDA_LAUNCH_BLOCKING=0 NCCL_IB_GID_INDEX=3"
output_dir="./checkpoints"

run_cmd="${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --master_port ${MASTER_PORT} --hostfile ${HOST_FILE_PATH} \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA} \
    --image_folder your_image_folder \
    --pretrain_mm_mlp_adapter="./checkpoints/projectors/${BASE_RUN_NAME}/mm_projector.bin" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=1e-5 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints '[(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)]' \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "./checkpoints/${MID_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 30000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --attn_implementation sdpa"

eval $run_cmd 2>&1 | tee "$output_dir/train.log" 