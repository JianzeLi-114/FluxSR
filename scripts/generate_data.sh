#!/usr/bin/env bash
# Generate one 1024x1024 teacher pair per JSON prompt; rerunning resumes.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-dev}"
PROMPT_PATH="${PROMPT_PATH:-weights/fluxsr/prompts_10k.json}"
LMDB_PATH="${LMDB_PATH:-data/flux_pairs_10k}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

sample_args=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    sample_args=(--max_samples "$MAX_SAMPLES")
fi

"$PYTHON_BIN" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
    scripts/generate_noise_image_pairs.py \
    --model_id "$MODEL_ID" --prompt_path "$PROMPT_PATH" --folder "$LMDB_PATH" \
    --batch_size "$GEN_BATCH_SIZE" --decode_batch_size 1 \
    --latent_resolution 128 --latent_channel 16 \
    --num_inference_steps 50 --guidance_scale 3.5 --seed 10 \
    "${sample_args[@]}" "$@"
