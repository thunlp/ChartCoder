LLM_VERSION="/mnt/afs/hfmodel/deepseek-coder-6.7b-instruct"
LLM_VERSION_CLEAN=$(basename $LLM_VERSION)
VISION_MODEL_VERSION="/data/home/zhaoxuanle/hfmodel/openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN=$(basename $VISION_MODEL_VERSION)

############### Pretrain ################

PROMPT_VERSION=plain
DATA="/data/home/zhaoxuanle/LLaVA-NeXT/scicap_chart2text_blip_unichart_our123.json"

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"


########################################
NUM_WORKERS=3
NUM_GPUS_PER_WORKER=8
MASTER_PORT=21231
cur_time=$(date "+%Y%m%d-%H%M%S")
HOST_FILE_PATH="hostfile"

OPTIONS_NCCL="NCCL_DEBUG=EEROR NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 CUDA_LAUNCH_BLOCKING=0 NCCL_IB_GID_INDEX=3"
output_dir="./checkpoints/projectors"

run_cmd="${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --master_port ${MASTER_PORT} --hostfile ${HOST_FILE_PATH} \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA} \
    --image_folder /path/to/images \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/projectors/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
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