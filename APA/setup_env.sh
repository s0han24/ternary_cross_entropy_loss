#!/bin/bash

# Use only GPU 0 (avoid GPU 1 which is currently in use)
export CUDA_VISIBLE_DEVICES=0

# pip (downloads, wheels, build artifacts)
export PIP_CACHE_DIR=/scratch/sujay/pip_cache
export PIP_BUILD_DIR=/scratch/sujay/pip_build

# Hugging Face transformers/datasets cache
export HF_HOME=/scratch/sujay/hf_cache
export TRANSFORMERS_CACHE=/scratch/sujay/hf_cache
export HF_DATASETS_CACHE=/scratch/sujay/hf_cache/datasets

# Torch hub / weights cache
export TORCH_HOME=/scratch/sujay/torch_cache

# Virtual environment path
export VENV_PATH=/scratch/sujay/bench_venv

# Activate virtual environment if it exists
if [ -d "$VENV_PATH" ]; then
    source $VENV_PATH/bin/activate
    echo "Activated virtual environment: $VENV_PATH"
    echo "Using GPU: $CUDA_VISIBLE_DEVICES"
else
    echo "Virtual environment not found at: $VENV_PATH"
fi