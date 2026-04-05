source "/path/to/your/anaconda3/etc/profile.d/conda.sh"
conda activate opencompass
cd $OPENCOMPASS_DIR

MODEL_PATH=${1:-""}
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
DATASETS="phybench_gen"

if [[ "$MODEL_PATH" == *"Qwen3-4B-Base"* || "$MODEL_PATH" == *"Qwen3-8B-Base"* || "$MODEL_PATH" == *"Qwen3-14B-Base"* ]]; then
  MAX_SEQ_LEN=32768
  MAX_OUT_LEN=24576
elif [[ "$MODEL_PATH" == *"Qwen3-4B" || "$MODEL_PATH" == *"Qwen3-8B"* || "$MODEL_PATH" == *"Qwen3-14B"* ]]; then
  MAX_SEQ_LEN=40960
  MAX_OUT_LEN=36864
else
  MAX_SEQ_LEN=32768
  MAX_OUT_LEN=24576
fi

NUM_RUNS=${NUM_RUNS:-10}
for run_id in $(seq 1 "${NUM_RUNS}"); do
  for ds in ${DATASETS//,/ }; do
    python run.py --datasets "$ds" --hf-path "$MODEL_PATH" --tokenizer-path "$MODEL_PATH" --hf-type "chat" \
      --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
      --model-kwargs trust_remote_code=True \
      --max-seq-len "$MAX_SEQ_LEN" --max-out-len "$MAX_OUT_LEN" --batch-size 32 \
      --max-num-workers "${NUM_GPUS}" --max-workers-per-gpu 1 -a vllm \
      --generation-kwargs do_sample=True temperature=0.6
  done
done
