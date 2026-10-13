#!/usr/bin/env bash
# Train FluxSR on 8 GPUs with a global batch size of 8.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-dev}"
LMDB_PATH="${LMDB_PATH:-data/flux_pairs_10k}"
OUTPUT_DIR="${OUTPUT_DIR:-exp/fluxsr}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRAIN_ITERS="${TRAIN_ITERS:-20000}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

resume_args=()
if [[ -n "${RESUME_FROM:-}" ]]; then
    resume_args=(--checkpoint_path "$RESUME_FROM")
fi

"$PYTHON_BIN" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" train.py \
    --model_id "$MODEL_ID" \
    --use_flux_generated_data --lmdb_path "$LMDB_PATH" \
    --deg_file_path realesrgan_paper.yaml \
    --output_path "$OUTPUT_DIR" --log_path "$OUTPUT_DIR/tb_log" \
    --train_iters "$TRAIN_ITERS" --batch_size 1 --num_workers 4 \
    --seed 114 --lr 1e-5 --warmup_step 500 --max_grad_norm 10 \
    --max_checkpoint 2 --checkpoint_keep_every 5000 \
    --log_iters 1000 --visual_iters 1000 --status_iters 10 \
    --guidance 4.0 --conditioning_timestep 350 \
    --use_fp16 --gradient_checkpointing \
    --lora_rank 64 --lora_alpha 64 \
    --spatial_loss --spatial_loss_type mse \
    --percep_loss lpips --percep_weight 1 --tv_weight 1 --adl_weight 0.1 \
    --log_loss "${resume_args[@]}" "$@"
