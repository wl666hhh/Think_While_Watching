#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${EVAL_MODEL_NAME:=TWW}"
: "${EVAL_SKIP_EXISTING:=1}"
export CUDA_VISIBLE_DEVICES EVAL_MODEL_NAME EVAL_SKIP_EXISTING

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_PATH="${SCRIPT_DIR}/configs/inference_config.json"

DATA_PATH=""
MULTI_GPU=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_path)    DATA_PATH="$2"; shift 2 ;;
        --config_path)  CONFIG_PATH="$2"; shift 2 ;;
        --model_name)   EVAL_MODEL_NAME="$2"; export EVAL_MODEL_NAME; shift 2 ;;
        --multi_gpu)    MULTI_GPU=true; shift ;;
        *)              echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -n "$DATA_PATH" ]]; then
    : "${EVAL_WRITE_BACK_DIR:=$DATA_PATH}"
    export EVAL_WRITE_BACK_DIR
fi

echo "=========================================="
echo "  Think-While-Watching Inference"
echo "=========================================="
echo "  Config:       $CONFIG_PATH"
echo "  GPUs:         $CUDA_VISIBLE_DEVICES"
echo "  Model name:   $EVAL_MODEL_NAME"
[[ -n "$DATA_PATH" ]] && echo "  Data path:    $DATA_PATH"
echo "=========================================="

CMD_ARGS="--config_path $CONFIG_PATH --model_name $EVAL_MODEL_NAME"
[[ -n "$DATA_PATH" ]] && CMD_ARGS="$CMD_ARGS --data_path $DATA_PATH"

if $MULTI_GPU; then
    IFS=',' read -ra GPU_ARR <<< "$CUDA_VISIBLE_DEVICES"
    NPROC="${#GPU_ARR[@]}"
    MASTER_PORT=$(shuf -i 29500-29999 -n 1)

    echo "Launching with $NPROC processes (port: $MASTER_PORT)..."
    python -m accelerate.commands.launch \
        --num_processes "$NPROC" \
        --main_process_port "$MASTER_PORT" \
        --mixed_precision bf16 \
        "${SCRIPT_DIR}/inference/streaming_inference.py" \
        $CMD_ARGS
else
    echo "Launching single-GPU inference..."
    python "${SCRIPT_DIR}/inference/streaming_inference.py" $CMD_ARGS
fi

echo ""
echo "=========================================="
echo "  Inference complete!"
echo "=========================================="
