set -ex

# ROLLOUT_MODEL_PATH=$1
# ROLLOUT_DATA_PATH=$2
# ROLLOUT_TEST_DATA_PATH=$3

ROLLOUT_MODEL_PATH="/mnt/shared-storage-user/llmrazor-share/model/Qwen2.5-Math-7B"
ROLLOUT_DATA_PATH="/mnt/shared-storage-user/huanghaian/code/verl/data/dapo_math/dapo-math-17k.jsonl"
ROLLOUT_TEST_DATA_PATH="/mnt/shared-storage-user/huanghaian/code/verl/data/dapo_math/aime-2024.jsonl"

# export XTUNER_USE_LMDEPLOY=1
export XTUNER_USE_SGLANG=1
export XTUNER_USE_FA3=0
export UVICORN_LOG_LEVEL="CRITICAl"
# export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export PYTHONPATH='.'
export ID_INPUT_OUTPUT=1
export XTUNER_MAX_CONCURRENT=16384

#export ID_INPUT_OUTPUT=1

OUTPUT_DIR='work_dirs/dapo_math_7B_newlmdeploy_nogroup_sglang'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

ray stop --force

python ci/scripts/test_dapo_trainer.py \
    --total-epochs 1 \
    --work-dir "$OUTPUT_DIR" \
    --model-path "$ROLLOUT_MODEL_PATH" \
    --data-path "$ROLLOUT_DATA_PATH" \
    --eval-data-path "$ROLLOUT_TEST_DATA_PATH" \
    --num-workers 8 \
    --gpus-per-node 8 \
    --rollout-global-batch-size 512 \
    --train-optimizer-steps 16 \
    --max-concurrent 16384 \
    --prompt-repeat-k 16 \
    --pack-max-length 32768 \
    --max-prompt-length 2048 \
    --max-response-length 8192 \
    --optimizer-disable-foreach \
    2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"