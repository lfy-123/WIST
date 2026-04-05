conda activate wist

TASK_NAME=${1:-"all"}
MODEL_PATH=${2:""}

case "$TASK_NAME" in
  all)
    echo "==> Evaluating Math"
    python -m openrlhf.evaluation.eval_math --model_path "$MODEL_PATH" --tensor_parallel_size 1

    echo "==> Evaluating MMLU-Pro"
    python -m openrlhf.evaluation.eval_mmlupro --model_path "$MODEL_PATH" --tensor_parallel_size 1

    echo "==> Evaluating BBEH"
    python -m openrlhf.evaluation.eval_bbeh --model_path "$MODEL_PATH" --tensor_parallel_size 1

    echo "==> Evaluating SuperGPQA"
    python -m openrlhf.evaluation.eval_supergpqa --model_path "$MODEL_PATH" --tensor_parallel_size 1

    echo "==> Evaluating GPQA-Diamond"
    python -m openrlhf.evaluation.eval_gpqa_diamond --model_path "$MODEL_PATH" --tensor_parallel_size 1
    ;;
  math)
    python -m openrlhf.evaluation.eval_math --model_path "$MODEL_PATH" --tensor_parallel_size 1
    ;;
  supergpqa)
    python -m openrlhf.evaluation.eval_supergpqa --model_path "$MODEL_PATH" --tensor_parallel_size 1
    ;;
  mmlu-pro)
    python -m openrlhf.evaluation.eval_mmlupro --model_path "$MODEL_PATH" --tensor_parallel_size 1
    ;;
  bbeh)
    python -m openrlhf.evaluation.eval_bbeh --model_path "$MODEL_PATH" --tensor_parallel_size 1
    ;;
  gpqa_diamond)
    python -m openrlhf.evaluation.eval_gpqa_diamond --model_path "$MODEL_PATH" --tensor_parallel_size 1
    ;;
  *)
    echo "Unsupported eval task: ${TASK_NAME}"
    exit 1
    ;;
esac
