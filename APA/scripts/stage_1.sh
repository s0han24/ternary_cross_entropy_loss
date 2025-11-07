#!/bin/bash

model='/scratch/sujayb/anlp/clarifying_questions/DPO_1_merged'

python -m pipeline.stage_1 \
    --config-path=../configs \
    --config-name=main.yaml \
    seed=1234 \
    logging_level=DEBUG \
    model.name=$model \
    pipeline.stage_index=0 \
    path.output='/scratch/sujayb/anlp/clarifying_questions/APA/outputs/' \
    train.num_train_epochs=3 \
    train.learning_rate='1e-3' \
    path.data='/scratch/sujayb/anlp/clarifying_questions/APA/dataset/'
