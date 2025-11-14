#!/usr/bin/env bash
set -euo pipefail

# Configuration (can be overridden by environment variables)
MODELS=(
  "Qwen/Qwen3-1.7B"
  "Qwen/Qwen3-8B"
)
BACKENDS=(
  "mirage"
  "vllm"
)

BS="${BS:-2}"          # max_num_batched_tokens
MAXTOK="${MAXTOK:-300}" # max_tokens
GPU="${GPU:-6}"         # CUDA_VISIBLE_DEVICES

gpu_type=$(nvidia-smi --query-gpu=name --format=csv,noheader | awk '{print $2}' | head -n 1)
ROOT_DIR="/home/jianan/repos/vllm"
OFFLINE_py="${ROOT_DIR}/offline.py"

mkdir -p "${ROOT_DIR}/outputs" "${ROOT_DIR}/vllm_profile"

for model in "${MODELS[@]}"; do
  # Safe model slug for file/dir names
  model_slug="$(echo "${model}" | tr '/: ' '___')"

  # Per-model directories
  out_dir="${ROOT_DIR}/outputs/${model_slug}"
  profile_base="vllm_profile/${model_slug}"
  mkdir -p "${out_dir}" "${ROOT_DIR}/${profile_base}"

  for backend in "${BACKENDS[@]}"; do
    suffix="${model_slug}_${gpu_type}_${backend}_bs${BS}_${MAXTOK}_asyncsched"
    profiler_dir="${profile_base}/${suffix}"
    out_file="${out_dir}/output_${suffix}.txt"

    mkdir -p "${ROOT_DIR}/${profiler_dir}"

    echo "Running model='${model}' backend='${backend}' bs=${BS} maxtok=${MAXTOK}"
    echo "Command: "
    echo "CUDA_VISIBLE_DEVICES="${GPU}" VLLM_TORCH_PROFILER_DIR="./${profiler_dir}" VLLM_DISABLE_COMPILE_CACHE=1 python "${OFFLINE_py}" --model "${model}" --max-num-batched-tokens "${BS}" --compilation "${backend}" --max-tokens "${MAXTOK}" --async-scheduling &> "${out_file}""
    
    CUDA_VISIBLE_DEVICES="${GPU}" \
    VLLM_TORCH_PROFILER_DIR="./${profiler_dir}" \
    VLLM_DISABLE_COMPILE_CACHE=1 \
      python "${OFFLINE_py}" \
        --model "${model}" \
        --max-num-batched-tokens "${BS}" \
        --compilation "${backend}" \
        --max-tokens "${MAXTOK}" \
        --async-scheduling \
      &> "${out_file}"

    echo "Done -> ${out_file}"
  done
done


