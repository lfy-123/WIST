source /path/to/your/anaconda3/etc/profile.d/conda.sh
conda activate wist
WORKSPACE_DIR="/path/to/your/WIST"
cd $WORKSPACE_DIR



max_levels=${1:-4}
model_name=${2:-"Qwen3-4B-Base"}
task=${3:-""}


if [[ "$task" == *Mathematics* ]]; then
  domain="Mathematics"
elif [[ "$task" == *Medicine* ]]; then
  domain="Medicine"
elif [[ "$task" == *Physics* ]]; then
  domain="Physics"
else
  domain="Mathematics"
fi

echo "max_levels is:$max_levels"
echo "model_name is:$model_name"
echo "domain is:$domain"

echo "Task name is:$task"

STORAGE_PATH="${WORKSPACE_DIR}/storage/${task}"
mkdir -p $STORAGE_PATH



logdir="${WORKSPACE_DIR}/logs/${task}"
mkdir -p ${logdir}

echo "logdir: ${logdir}/web.log"
python -m knowledge_tree.worker_web --storage_path $STORAGE_PATH \
    --max_levels $max_levels --model_name $model_name \
    --specified_domain $domain --poll_interval 1 > ${logdir}/web.log 2>&1
    
