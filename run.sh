#!/bin/bash

python inference_17.py \
    --dataset_path /scratch/sujayb/anlp/clarifying_questions/data/ambigqa.dev_4h.clarify.jsonl \
    --model_path /scratch/sujayb/anlp/clarifying_questions/final_fix/final_kto_model_merged \
    --mode respond \
    --n_samples 1 \
    --test 1 \
    --batch_size 8

python inference_17.py \
    --dataset_path /scratch/sujayb/anlp/clarifying_questions/data/ambigqa.dev_4h.clarify.jsonl \
    --model_path /scratch/sujayb/anlp/clarifying_questions/final_fix/final_model_merged \
    --mode clarify_q \
    --n_samples 1 \
    --test 8 \
    --batch_size 8

python inference_17.py \
    --dataset_path /scratch/sujayb/anlp/clarifying_questions/data/ambigqa.dev_4h.clarify.jsonl \
    --model_path /scratch/sujayb/anlp/clarifying_questions/final_fix/final_model_merged \
    --mode clarify_a \
    --n_samples 1 \
    --test 8 \
    --batch_size 8

python inference_17.py \
    --dataset_path /scratch/sujayb/anlp/clarifying_questions/data/ambigqa.dev_4h.clarify.jsonl \
    --model_path /scratch/sujayb/anlp/clarifying_questions/final_fix/final_model_merged \
    --mode qa_output \
    --n_samples 1 \
    --test 8 \
    --batch_size 8