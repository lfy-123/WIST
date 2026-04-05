source "/path/to/your/anaconda3/etc/profile.d/conda.sh"
conda activate opencompass
cd $OPENCOMPASS_DIR

# eval model path
MODEL_PATH=${1:-""}

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
DATASETS="medmcqa_gen MedQA_gen_3bf756 medbullets_gen"


NUM_RUNS=${NUM_RUNS:-10}
for run_id in $(seq 1 "${NUM_RUNS}"); do
  for ds in ${DATASETS//,/ }; do
    python run.py --datasets "$ds" --hf-path "$MODEL_PATH" --tokenizer-path "$MODEL_PATH" --hf-type "chat" \
      --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
      --model-kwargs trust_remote_code=True \
      --max-seq-len 32768 --max-out-len 24576 --batch-size 32 \
      --max-num-workers "${NUM_GPUS}" --max-workers-per-gpu 1 -a vllm \
      --generation-kwargs do_sample=True temperature=0.6
  done
done
