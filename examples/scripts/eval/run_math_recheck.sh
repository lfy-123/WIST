conda activate wist
cd your_path_to_wist/WIST

INPUT_FILE=${1:-""}

if [[ -z "$INPUT_FILE" ]]; then
  echo "Usage: bash run_math_recheck.sh <path_to_results.json>"
  exit 1
fi

python -m openrlhf.evaluation.results_recheck --input_file "$INPUT_FILE"
