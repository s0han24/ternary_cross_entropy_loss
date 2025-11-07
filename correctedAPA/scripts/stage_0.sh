model='meta-llama/Llama-3.2-3B-Instruct'

python -m pipeline.stage_0 \
    --config-path=../configs \
    --config-name=main.yaml \
    seed=1234 \
    logging_level=DEBUG \
    model.name=$model