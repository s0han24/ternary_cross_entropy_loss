#!/bin/bash


model='/scratch/sujayb/anlp/clarifying_questions/data/llama2/gen_clarify_q/DPO_1/best_checkpoint'

python -m pipeline.stage_0 \
    --config-path=../configs \
    --config-name=main.yaml \
    seed=1234 \
    logging_level=DEBUG \
    model.name=$model