import os
import json
import random
import argparse
from collections import Counter

import torch
from trl import SFTTrainer, SFTConfig
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from accelerate import Accelerator, PartialState

from utils import gen_direct_qa_output_prompt, gen_clarify_q_prompt, gen_clarify_a_prompt, gen_qa_output_prompt
from utils import QA_OUTPUT, CLARIFY_Q, CLARIFY_A

def preprocess(ex, mode):
    sft_examples = []
    if mode == 'gen_clarify_q':
        sft_examples.append(gen_clarify_q_prompt(
            qa_input=ex['question'],
            clarify_q=ex['clarification']['question'],
        ))
    elif mode == 'gen_clarify_a':
        for answer in ex['clarification']['answers']:
            sft_examples.append(gen_clarify_a_prompt(
                qa_input=ex['question'],
                clarify_q=ex['clarification']['question'],
                qa_output=answer['response'],
                clarify_a=answer['answer'],
            ))
    elif mode == 'gen_qa_output':
        for answer in ex['clarification']['answers']:
            sft_examples.append(gen_qa_output_prompt(
                qa_input=ex['question'],
                clarify_q=ex['clarification']['question'],
                clarify_a=answer['answer'],
                qa_output=answer['response'],
            ))
    elif mode == 'gen_direct_qa_output':
        for answer in ex['clarification']['answers']:
            sft_examples.append(gen_direct_qa_output_prompt(
                qa_input=ex['question'],
                qa_output=answer['response'],
            ))
    else:
        raise ValueError

    return sft_examples


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision="fp16",  # Enable mixed precision training
        log_with="all"  # Enable all logging
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
            ]
    dev_data = {}
    for path in args.dev_paths:
        with open(path, 'r') as f:
            data = [json.loads(l) for l in f]
            dev_data[os.path.split(path)[-1]] = [
                sft_ex for ex in data for sft_ex in preprocess(ex, mode=args.mode)
            ]

    if args.test:
        train_data = {k: sorted(v, key=len, reverse=True)[:args.test] for k, v in train_data.items()}
        dev_data = {k: sorted(v, key=len, reverse=True)[:args.test] for k, v in dev_data.items()}


    for name, data in train_data.items():
        with open(os.path.join(experiment_dir, 'train.sft.' + name), 'w') as f:
            for ex in data:
                f.write(json.dumps(ex) + '\n')
    for name, data in dev_data.items():
        with open(os.path.join(experiment_dir, 'dev.sft.' + name), 'w') as f:
            for ex in data:
                f.write(json.dumps(ex) + '\n')

    train_data = Dataset.from_list([{'text': ex} for data in train_data.values() for ex in data])
    dev_data = {
        name: Dataset.from_list([{'text': ex} for ex in data]) for name, data in dev_data.items()
    }

    per_device_batch_size = args.batch_size // torch.cuda.device_count() // args.grad_accum_steps
    print(f'Device Count={torch.cuda.device_count()}')
    print(f'Batch Size={args.batch_size}')
    print(f'Grad Accum Steps={args.grad_accum_steps}')
    print(f'Per Device Batch Size={per_device_batch_size}')
    print(f'Distributed training: {accelerator.distributed_type}')
    print(f'Number of processes: {accelerator.num_processes}')
    if args.batch_size % (torch.cuda.device_count() * args.grad_accum_steps):
        raise ValueError(
            'Invalid Batch Size={} and Device Count={} and Grad Accum Steps={}'.format(
                args.batch_size,
                torch.cuda.device_count(),
                args.grad_accum_steps
                )
            )
    
    # Use SFTConfig instead of TrainingArguments
    training_args = SFTConfig(
        completion_only_loss=True,
        output_dir=experiment_dir,
        eval_strategy='epoch',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.grad_accum_steps,
        save_strategy='epoch',
        save_total_limit=None,
        logging_steps=50,
        packing=False,
        dataset_text_field="text",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map={'':PartialState().process_index},
        torch_dtype=torch.float16  # Use fp16 for more memory efficiency
    )
    # Set static graph for DDP compatibility
    if hasattr(model, '_set_static_graph'):
        model._set_static_graph()
    
    if args.checkpoint:
        model = PeftModel.from_pretrained(
            model,
            os.path.join(args.checkpoint),
            is_trainable=True
        )
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type="CAUSAL_LM",
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token


    if args.mode == 'gen_clarify_q':
        response_template = CLARIFY_Q
    elif args.mode == 'gen_clarify_a':
        response_template = CLARIFY_A
    elif args.mode == 'gen_qa_output':
        response_template = QA_OUTPUT
    elif args.mode == 'gen_direct_qa_output':
        response_template = QA_OUTPUT
    else:
        raise ValueError

    response_template_ids = tokenizer.encode('\n' + response_template, add_special_tokens=False)[2:]
    # data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    data_collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.eos_token)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        data_collator=data_collator,
        peft_config=peft_config,
    )


    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=False,
        enable_mem_efficient=False
    ):
        trainer.train()

    if accelerator.is_local_main_process:
        with open(os.path.join(experiment_dir, 'log_history.json'), 'w') as f:
            f.write(json.dumps(trainer.state.log_history, indent=2))
        trainer.model.save_pretrained(os.path.join(experiment_dir, 'best_checkpoint'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='llama2')
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--train_paths', nargs='+', default=["/scratch/sujayb/anlp/clarifying_questions/data/splits/final_train.jsonl"])
    parser.add_argument('--dev_paths', nargs='+', default=["/scratch/sujayb/anlp/clarifying_questions/data/splits/final_dev.jsonl"])
    parser.add_argument('--test', type=int, default=None)
    parser.add_argument('--mode', default="gen_clarify_q")
    parser.add_argument(
        '--output_dir',
        default='../data/'
    )
    parser.add_argument('--random_seed', type=int, default=88888888)


    # General Training Hyperparameters
    parser.add_argument('--epochs', type=float, default=5.0)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=32)  # Reduced from 256
    parser.add_argument('--grad_accum_steps', type=int, default=8)  # Increased to maintain effective batch size
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