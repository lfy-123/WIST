#!/bin/bash
# Initialize conda environment
# Edit this path to your conda installation path
source "/path/to/your/anaconda3/etc/profile.d/conda.sh"
conda activate wist

# Edit this path to the WIST workspace
WORKSPACE_DIR="/path/to/your/WIST"
cd $WORKSPACE_DIR

# Automatic detection of available GPU count
AVAILABLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "$AVAILABLE_GPUS" -ne "$GPU_COUNT_EXPECTED" ]]; then
  GPU_COUNT_EXPECTED="$AVAILABLE_GPUS"
  echo "Adjusted GPU count to available GPUs: ${GPU_COUNT_EXPECTED}"
fi

# Set general training parameters
MODEL_NAME=${1:-"Qwen3-4B-Base"}
MAX_LEVELS=${2:-4}
GPU_COUNT_EXPECTED=${3:-4}
TRAIN_BATCH_SIZE=${4:-512}
MAX_RESAMPLE_ATTEMPTS=${5:-4}
TARGET_DOMAIN=${6:-"Mathematics"}
TREE_WINDOW_SIZE=${7:-5}

# Determine domain-specific parameters based on TARGET_DOMAIN
case "${TARGET_DOMAIN,,}" in
    "mathematics"|"math")
        TARGET_DOMAIN="Mathematics"
        PROMPT_VERSION="math"
        ;;
    "medicine"|"med")
        TARGET_DOMAIN="Medicine"
        PROMPT_VERSION="med"
        ;;
    "physics"|"phy")
        TARGET_DOMAIN="Physics"
        PROMPT_VERSION="phy"
        ;;
    *)
        echo "Unknown TARGET_DOMAIN: $TARGET_DOMAIN"
        exit 1
        ;;
esac

QUESTION_CANDIDATES_PER_PROMPT=16
EXPLORATION_STRATEGY="reward"

# Unified TASK_NAME string across domains
TASK_NAME="wist_${TARGET_DOMAIN,,}_web_tree_${EXPLORATION_STRATEGY}_window${TREE_WINDOW_SIZE}_${MODEL_NAME}_levels${MAX_LEVELS}_bs${TRAIN_BATCH_SIZE}_attempts${MAX_RESAMPLE_ATTEMPTS}"

# Set model path - Edit this to your base models path
MODEL_PATH="/path/to/your/Models/${MODEL_NAME}"

# Set storage paths
STORAGE_PATH="${WORKSPACE_DIR}/storage/${TASK_NAME}"
OUTPUT_PATH="${WORKSPACE_DIR}/outputs/${TASK_NAME}"
LOG_PATH="${WORKSPACE_DIR}/logs/${TASK_NAME}"
TENSORBOARD_PATH="${WORKSPACE_DIR}/logs_tensorboard/${TASK_NAME}"
WEB_CORPUS_PATH="${STORAGE_PATH}/${MODEL_NAME}_${TARGET_DOMAIN}_web_corpus.json"

mkdir -p "${STORAGE_PATH}" "${OUTPUT_PATH}" "${LOG_PATH}"

# start search engine
echo "============== Run the following two commands separately to start the search engine, or you can replace it with your own search engine. =============="
echo "bash ${WORKSPACE_DIR}/examples/scripts/web/worker_web_searxng.sh"
echo "bash ${WORKSPACE_DIR}/examples/scripts/web/worker_web.sh $MAX_LEVELS $MODEL_NAME $TASK_NAME"

# Train the model
python3 -m openrlhf.cli.train_ppo_ray \
  --specified_domain "${TARGET_DOMAIN}" \
  --model_name "${MODEL_NAME}" \
  --storage_path "${STORAGE_PATH}" \
  --web_corpus_path "${WEB_CORPUS_PATH}" \
  --tree_window_size "${TREE_WINDOW_SIZE}" \
  --explore_strategy "${EXPLORATION_STRATEGY}" \
  --tree_prompt_version "${PROMPT_VERSION}" \
  --title_selection_mode "sequence" \
  --question_type_judge_start_step -1 \
  --generate_qa_prompt "${PROMPT_VERSION}" \
  --challenger_n_candidates_per_prompt "${QUESTION_CANDIDATES_PER_PROMPT}" \
  --challenger_max_resample_attempts "${MAX_RESAMPLE_ATTEMPTS}" \
  --select_min_count 2 \
  --select_top_k_nodes 3 \
  --num_reasoner_samples 8 \
  --n_samples_per_prompt 8 \
  --format_invalid_penalty -0.1 \
  --challenger_rm_mode "spice" \
  --instruction_len 2400 \
  --cold_start_steps 10000 \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --micro_train_batch_size 1 \
  --micro_rollout_batch_size 1 \
  --rollout_batch_size 128 \
  --prompt_max_len 8192 \
  --generate_max_len 8192 \
  --max_len 16384 \
  --max_steps 50 \
  --save_steps 10 \
  --eval_steps -1 \
  --save_path "${OUTPUT_PATH}/final" \
  --ckpt_path "${OUTPUT_PATH}/ckpt" \
  --save_hf_ckpt \
  --max_ckpt_num 2 \
  --max_epochs 1 \
  --policy_loss_type "dr_grpo" \
  --advantage_estimator "dr_grpo" \
  --kl_estimator k1 \
  --init_kl_coef 0 \
  --gamma 1.0 \
  --zero_stage 3 \
  --bf16 \
  --actor_learning_rate 1e-6 \
  --critic_learning_rate 9e-6 \
  --pretrain "${MODEL_PATH}" \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node "${GPU_COUNT_EXPECTED}" \
  --reward_num_nodes 1 \
  --reward_num_gpus_per_node "${GPU_COUNT_EXPECTED}" \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node "${GPU_COUNT_EXPECTED}" \
  --vllm_num_engines "${GPU_COUNT_EXPECTED}" \
  --vllm_tensor_parallel_size 1 \
  --colocate_all_models \
  --vllm_gpu_memory_utilization 0.6 \
  --vllm_sync_backend nccl \
  --vllm_enable_sleep \
  --deepspeed_enable_sleep \
  --lr_scheduler "constant" \
  --input_key context_messages \
  --apply_chat_template \
  --gradient_checkpointing \
  --packing_samples \
  --enforce_eager \
  --load_checkpoint \
  --use_tensorboard "${TENSORBOARD_PATH}" \
  2>&1 | tee "${LOG_PATH}/train_${MODEL_NAME}.log"
