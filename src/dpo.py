import os
import json
import random
import argparse

import torch
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from accelerate import Accelerator, PartialState
from transformers import BitsAndBytesConfig

from utils import gen_clarify_q_prompt

def preprocess(ex, mode):
    dpo_data = []
    for chosen in ex['dpo']['clarifications']:
        for rejected in ex['dpo']['clarifications']:
            if chosen['rank'] >= rejected['rank']:
                continue
            if mode == 'gen_clarify_q':
                dpo_data.append({
                    'prompt': gen_clarify_q_prompt(
                        qa_input=ex['question'],
                        clarify_q=None,
                    ),
                    'chosen': chosen['question'] + '\n',
                    'rejected': rejected['question'] + '\n',
                })
            else:
                raise ValueError
    return dpo_data


def main(args):
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision="bf16",  # Use bfloat16 mixed precision
        log_with="all"
    )

    if args.model == 'llama2':
        args.base_model = 'meta-llama/Llama-3.1-8B-Instruct'
    elif args.model == 'gemma':
        args.base_model = 'google/gemma-7b'
    else:
        raise ValueError('Invalid base_model')

    experiment_dir = os.path.join(
        args.output_dir,
        args.model,
        args.mode,
        args.experiment_name,
    )
    os.makedirs(experiment_dir, exist_ok=True)

    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), indent=2))

    train_data = {}
    for path in args.train_paths:
        with open(path, 'r') as f:
            data = [json.loads(l) for l in f]
            train_data[os.path.split(path)[-1]] = [
                sft_ex for ex in data for sft_ex in preprocess(ex, mode=args.mode)
            ][:args.test]
    dev_data = {}
    for path in args.dev_paths:
        with open(path, 'r') as f:
            data = [json.loads(l) for l in f]
            dev_data[os.path.split(path)[-1]] = [
                sft_ex for ex in data for sft_ex in preprocess(ex, mode=args.mode)
            ][:args.test]

    for name, data in train_data.items():
        with open(os.path.join(experiment_dir, 'train.sft.' + name), 'w') as f:
            for ex in data:
                f.write(json.dumps(ex) + '\n')
    for name, data in dev_data.items():
        with open(os.path.join(experiment_dir, 'dev.sft.' + name), 'w') as f:
            for ex in data:
                f.write(json.dumps(ex) + '\n')

    train_data = Dataset.from_list([ex for data in train_data.values() for ex in data])
    dev_data = Dataset.from_list([ex for data in dev_data.values() for ex in data])

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_bos_token=False

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",  # Let the model decide optimal device mapping
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        max_memory={i: "24GiB" for i in range(torch.cuda.device_count())},  # Limit memory per GPU
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Set static graph for DDP compatibility
    if hasattr(model, '_set_static_graph'):
        model._set_static_graph()

    if args.checkpoint:
        model = PeftModel.from_pretrained(
            model,
            os.path.join(args.checkpoint),
        )
        model = model.merge_and_unload()

    # Put model in training mode
    model.train()

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )



    # Calculate per-device batch size
    per_device_batch_size = max(1, args.batch_size // (torch.cuda.device_count() * args.grad_accum_steps))
    effective_batch_size = per_device_batch_size * torch.cuda.device_count() * args.grad_accum_steps
    
    print(f'Device Count={torch.cuda.device_count()}')
    print(f'Input Batch Size={args.batch_size}')
    print(f'Grad Accum Steps={args.grad_accum_steps}')
    print(f'Per Device Batch Size={per_device_batch_size}')
    print(f'Effective Total Batch Size={effective_batch_size}')
    print(f'Distributed Type={accelerator.distributed_type}')
    print(f'Number of Processes={accelerator.num_processes}')
    training_args = DPOConfig(
        output_dir=experiment_dir,
        eval_strategy='steps',
        eval_steps=args.eval_steps,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum_steps,
        save_steps=args.save_steps or args.eval_steps,
        save_total_limit=None,
        logging_steps=50,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        torch_compile=False,  # Disable torch compile to save memory
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        bf16=True,  # Use bfloat16 mixed precision
        optim="adamw_torch_fused",  # Use fused optimizer for better memory efficiency
    )

    trainer = DPOTrainer(
        model,
        ref_model=None,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
    )

    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=False,
        enable_mem_efficient=False
    ):
        trainer.train(resume_from_checkpoint=args.resume_from)
    with open(os.path.join(experiment_dir, 'log_history.json'), 'w') as f:
        f.write(json.dumps(trainer.state.log_history, indent=2))
    trainer.model.save_pretrained(os.path.join(experiment_dir, 'best_checkpoint'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_paths', nargs='+', default=["/scratch/sujayb/anlp/clarifying_questions/data/splits/final_train.jsonl"])
    parser.add_argument('--dev_paths', nargs='+', default=["/scratch/sujayb/anlp/clarifying_questions/data/splits/final_dev.jsonl"])
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--resume_from', default=None)
    parser.add_argument('--random_seed', type=int, default=88888888)
    parser.add_argument(
        '--output_dir',
        default='../data/'
    )
    parser.add_argument('--test', type=int, default=None)

    parser.add_argument('--model', default='llama2')
    parser.add_argument('--mode', default="gen_clarify_q")
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_prompt_length', type=int, default=128)

    # General Training Hyperparameters
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--eval_steps', type=int, default=750)
    parser.add_argument('--save_steps', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=5e-7)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.1)

    # Lora Hyperparameters
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_bias', default="none")

    cli_args = parser.parse_args()
    random.seed(cli_args.random_seed)
    main(cli_args)

